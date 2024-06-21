import torch


def reconstruction(x, y):
    """
    The MSE loss of the reconstruction
    :param x: The input tensor
    :param y: The reconstructed tensor
    """
    return ((y - x) ** 2).mean()


def act_sparsity(f):
    """
    The sparsity loss of the activations
    :param f: The feature activations
    """
    return f.abs().sum(-1).mean()


def grad_sparsity(f, grad, dictionary):
    """
    The sparsity loss of the gradients
    :param f: The feature activations
    :param grad: The gradient of the LLM's loss w.r.t. the activations
    :param dictionary: The autoencoder dictionary, i.e. the decoder weights
    """
    return (f * (grad @ dictionary)).abs().sum(-1).mean()


def unexplained(x, y, grad):
    """
    The contribution of the unexplained variance to the loss of the LLM
    :param x: The input tensor
    :param y: The reconstructed tensor
    :param grad: The gradient of the LLM's loss w.r.t. the activations
    """
    return ((x - y) * grad).sum(-1).abs().mean()
