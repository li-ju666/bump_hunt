"""Mask-conditioned velocity field: residual MLP with adaptive layer norm on the time
embedding, the ternary mask one-hot encoded and concatenated to the input."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None, :] * 1000.0
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ada = nn.Linear(hidden, 2 * hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h, temb):
        scale, shift = self.ada(temb).chunk(2, dim=-1)
        u = self.norm(h) * (1.0 + scale) + shift
        return h + self.fc2(F.silu(self.fc1(u)))


class VelocityNet(nn.Module):
    def __init__(self, d, hidden=256, n_blocks=4, t_dim=256):
        super().__init__()
        self.d, self.t_dim = d, t_dim
        self.inp = nn.Linear(d + 3 * d, hidden)
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.blocks = nn.ModuleList([Block(hidden) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, t, mask):
        """z (B, d), t (B,), mask (B, d) in {-1, 0, 1} -> velocity (B, d)."""
        onehot = torch.stack([(mask == -1), (mask == 0), (mask == 1)], dim=-1)
        onehot = onehot.reshape(*z.shape[:-1], 3 * self.d).to(z.dtype)
        h = self.inp(torch.cat([z, onehot], dim=-1))
        temb = self.t_mlp(sinusoidal(t, self.t_dim))
        for blk in self.blocks:
            h = blk(h, temb)
        return self.out(self.out_norm(h))
