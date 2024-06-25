import gc

import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from utils import TORCH_DTYPES

class GradBufferConfig:
    def __init__(
            self,
            model_name,
            layers,
            dataset_name,
            dataset_split=None,
            dataset_config=None,
            buffer_size=256,
            min_capacity=128,
            model_batch_size=8,
            samples_per_seq=None,
            max_seq_length=None,
            act_size=None,
            shuffle_buffer=True,
            seed=None,
            device='cuda',
            dtype=torch.float32,
            buffer_device=None,
            offload_device=None,
            refresh_progress=False,
    ):
        """
        :param model_name: the hf model name
        :param layers: which layers to get activations from, passed as a list of ints
        :param dataset_name: the name of the hf dataset to use
        :param dataset_split: the split of the dataset to use
        :param dataset_config: the config to use when loading the dataset
        :param buffer_size: the size of the buffer, in number of activations
        :param min_capacity: the minimum guaranteed capacity of the buffer, in number of activations, used to determine
        when to refresh the buffer
        :param model_batch_size: the batch size to use in the model when generating activations
        :param samples_per_seq: the number of activations to randomly sample from each sequence. If None, all
        activations will be used
        :param max_seq_length: the maximum sequence length to use when generating activations. If None, the sequences
        will not be truncated
        :param act_size: the size of the activations vectors. If None, it will guess the size from the model's cfg
        :param shuffle_buffer: if True, the buffer will be shuffled after each refresh
        :param seed: the seed to use for dataset shuffling and activation sampling
        :param device: the device to use for the model
        :param dtype: the dtype to use for the buffer and model
        :param buffer_device: the device to use for the buffer. If None, it will use the same device as the model
        :param offload_device: the device to offload the model to when not generating activations. If None, offloading
        is disabled. If using this, make sure to use a large enough buffer to avoid frequent offloading
        :param refresh_progress: If True, a progress bar will be displayed when refreshing the buffer
        """

        assert isinstance(layers, list) and len(layers) > 0, "Layers must be a non-empty list of ints"

        self.model_name = model_name
        self.layers = layers
        self.dataset_name = dataset_name
        self.act_sites = ('hook_mlp_out', 'hook_attn_out')
        self.act_names = [f'blocks.{layer}.{site}' for layer in layers for site in self.act_sites]
        self.dataset_split = dataset_split
        self.dataset_config = dataset_config
        self.buffer_size = buffer_size
        self.min_capacity = min_capacity
        self.model_batch_size = model_batch_size
        self.samples_per_seq = samples_per_seq
        self.max_seq_length = max_seq_length
        self.act_size = act_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.device = device
        self.dtype = TORCH_DTYPES[dtype] if isinstance(dtype, str) else dtype
        self.buffer_device = buffer_device or device
        self.offload_device = offload_device
        self.refresh_progress = refresh_progress
        self.final_layer = max(layers)  # the final layer that needs to be run


class GradBuffer:
    """
    A data buffer to store generate and store activations and gradients used for training the attribution autoencoder.
    """

    def __init__(self, cfg: GradBufferConfig, hf_model=None):
        self.cfg = cfg

        if cfg.seed:
            torch.manual_seed(cfg.seed)

        # pointer to the current position in the dataset
        self.dataset_pointer = 0

        # load the dataset into a looping data loader
        if cfg.dataset_config:
            dataset = datasets.load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
        else:
            dataset = datasets.load_dataset(cfg.dataset_name, split=cfg.dataset_split)

        self.data_loader = DataLoader(
            dataset['text'],
            batch_size=cfg.model_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8
        )
        self.data_generator = iter(self.data_loader)

        # load the model into a HookedTransformer
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name=cfg.model_name,
            hf_model=hf_model,
            device=cfg.device,
            dtype=cfg.dtype
        )

        # if the act_size is not provided, use the size from the model's cfg
        self.cfg.act_size = self.model.cfg.d_model

        # if the buffer is on the cpu, pin it to memory for faster transfer to the gpu
        pin_memory = cfg.buffer_device == 'cpu'

        # the buffer to store activations in, with shape (size, len(layers), len(act_sites), act_size)
        self.act_buffer = torch.zeros(
            (cfg.buffer_size, len(cfg.layers), len(cfg.act_sites), cfg.act_size),
            dtype=cfg.dtype,
            pin_memory=pin_memory,
            device=cfg.buffer_device
        )

        # the buffer to store gradients in, with shape (size, len(layers), act_size)
        # len(act_sites) is not needed because the grads are the same for both the attn and mlp outputs
        # TODO: currently stores many copies of the same gradients to make things easier, should be optimized
        self.grad_buffer = torch.zeros(
            (cfg.buffer_size, len(cfg.layers), cfg.act_size),
            dtype=cfg.dtype,
            pin_memory=pin_memory,
            device=cfg.buffer_device
        )

        # pointer to read/write location in the buffer, reset to 0 after refresh is called
        # starts at buffer_size to be fully filled on first refresh
        self.buffer_pointer = cfg.buffer_size

        # initial buffer fill
        self.refresh()

    def refresh(self):
        """
        Whenever the buffer is refreshed, we remove the first `buffer_pointer` activations that were used, shift the
        remaining activations to the start of the buffer, and then fill the rest of the buffer with `buffer_pointer` new
        activations from the model.
        """

        # shift the remaining activations/grads to the start of the buffer
        self.act_buffer = torch.roll(self.act_buffer, -self.buffer_pointer, 0)
        self.grad_buffer = torch.roll(self.grad_buffer, -self.buffer_pointer, 0)

        # if offloading is enabled, move the model to `cfg.device` before generating activations
        if self.cfg.offload_device:
            self.model.to(self.cfg.device)

        # start a progress bar if `refresh_progress` is enabled
        if self.cfg.refresh_progress:
            pbar = tqdm(total=self.buffer_pointer)

        # fill the rest of the buffer with `buffer_pointer` new activations from the model
        while self.buffer_pointer > 0:
            # get the next batch of seqs
            try:
                seqs = next(self.data_generator)
            except StopIteration:
                self.reset_dataset()
                seqs = next(self.data_generator)

            seqs = self.model.tokenizer(
                seqs,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.cfg.max_seq_length
            ).to(self.cfg.device)['input_ids']

            self.model.zero_grad()

            # run the seqs through the model to get the activations
            # out - [batch, pos, n_vocab]
            out, cache = self.model.run_with_cache(seqs, names_filter=self.cfg.act_names)

            # compute gradients w.r.t. the model's loss
            input = out[:, :-1].flatten(0, 1)
            target = seqs[:, 1:].flatten()
            model_loss = F.cross_entropy(input, target)
            model_loss.backward()

            # clean up logits in order to free the graph memory
            del out
            torch.cuda.empty_cache()

            # store the activations and gradients in the buffer
            acts = torch.stack([torch.stack([cache[f"blocks.{layer}.{site}"] for layer in self.cfg.layers], dim=-2) for site in self.cfg.act_sites], dim=-2)
            grads = torch.stack([self.model.blocks[layer].mlp.b_out.grad for layer in self.cfg.layers], dim=-2) # (layer, act_size)

            # (batch, pos, layer, act_site, act_size) -> (batch*samples_per_seq, layer, act_site, act_size)
            if self.cfg.samples_per_seq:
                acts = acts[:, torch.randperm(acts.shape[-3])[:self.cfg.samples_per_seq]].flatten(0, 1)
            else:
                acts = acts.flatten(0, 1)

            write_pointer = self.cfg.buffer_size - self.buffer_pointer

            new_acts = min(acts.shape[0], self.buffer_pointer)  # the number of acts to write, capped by buffer_pointer
            self.act_buffer[write_pointer:write_pointer + new_acts].copy_(acts[:new_acts], non_blocking=True)
            del acts

            self.grad_buffer[write_pointer:write_pointer + new_acts].copy_(grads.unsqueeze(dim=0).expand(new_acts, -1, -1), non_blocking=True)

            # update the buffer pointer by the number of activations we just added
            self.buffer_pointer -= new_acts

            # update the progress bar
            if self.cfg.refresh_progress:
                pbar.update(new_acts)

        # close the progress bar
        if self.cfg.refresh_progress:
            pbar.close()

        # sync the buffer to ensure async copies are complete - this is needed because of the use of pinned memory/non_blocking=True
        torch.cpu.synchronize()

        # if shuffle_buffer is enabled, shuffle the buffers
        if self.cfg.shuffle_buffer:
            perm = torch.randperm(self.cfg.buffer_size)
            self.act_buffer = self.act_buffer[perm]
            self.grad_buffer = self.grad_buffer[perm]

        # if offloading is enabled, move the model back to `cfg.offload_device`, and clear the cache
        if self.cfg.offload_device:
            self.model.to(self.cfg.offload_device)
            torch.cuda.empty_cache()

        gc.collect()

        assert self.buffer_pointer == 0, "Buffer pointer should be 0 after refresh"

    @torch.no_grad()
    def next(self, batch: int = None):
        # if this batch read would take us below the min_capacity, refresh the buffer
        if self.will_refresh(batch):
            with torch.enable_grad():
                self.refresh()

        if batch is None:
            acts = self.act_buffer[self.buffer_pointer]
            grads = self.grad_buffer[self.buffer_pointer]
        else:
            acts = self.act_buffer[self.buffer_pointer:self.buffer_pointer + batch]
            grads = self.grad_buffer[self.buffer_pointer:self.buffer_pointer + batch]

        self.buffer_pointer += batch or 1

        return acts, grads

    def reset_dataset(self):
        """
        Reset the buffer to the beginning of the dataset without reshuffling.
        """
        self.data_generator = iter(self.data_loader)

    def will_refresh(self, batch: int = None):
        return self.cfg.buffer_size - (self.buffer_pointer + (batch or 1)) < self.cfg.min_capacity
