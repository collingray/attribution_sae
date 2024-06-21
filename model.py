from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttributionSAEConfig:
    n_dim: int
    m_dim: int
    device: torch.device
    dtype: torch.dtype


class AttributionSAE(nn.Module):
    def __init__(self, cfg: AttributionSAEConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cfg = cfg

        self.W_e = nn.Linear(cfg.n_dim, cfg.m_dim, bias=False, device=cfg.device, dtype=cfg.dtype)
        self.b_e = nn.Parameter(torch.zeros(cfg.m_dim, device=cfg.device, dtype=cfg.dtype))
        self.W_d = nn.Linear(cfg.m_dim, cfg.n_dim, bias=False, device=cfg.device, dtype=cfg.dtype)
        self.b_d = nn.Parameter(torch.zeros(cfg.n_dim, device=cfg.device, dtype=cfg.dtype))

    def encode(self, x):
        return F.relu(self.W_e(x) + self.b_e)

    def decode(self, x):
        return self.W_d(x) + self.b_d

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f
