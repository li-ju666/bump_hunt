"""Preprocessing between physical (theta, counts) and the standardised joint state z.

Counts are dequantised with fresh U[0, 1) noise, variance-stabilised with log1p, and
standardised per bin; parameters are standardised per coordinate. Both maps are
invertible and their log-Jacobians are available, so learned densities can be compared
with exact ones in physical units.
"""

import numpy as np
import torch

from .simulator import D_THETA, D_X


class Transform:
    def __init__(self, theta, x, rng):
        x = np.asarray(x, dtype=float)
        xt = np.log1p(x + rng.uniform(0.0, 1.0, x.shape))
        self.theta_mean = torch.tensor(theta.mean(0), dtype=torch.float32)
        self.theta_std = torch.tensor(theta.std(0), dtype=torch.float32)
        self.x_mean = torch.tensor(xt.mean(0), dtype=torch.float32)
        self.x_std = torch.tensor(xt.std(0), dtype=torch.float32)

    # ---- state dict for checkpoints
    def state_dict(self):
        return {k: getattr(self, k) for k in ["theta_mean", "theta_std", "x_mean", "x_std"]}

    @classmethod
    def from_state_dict(cls, sd):
        obj = cls.__new__(cls)
        for k, v in sd.items():
            setattr(obj, k, v)
        return obj

    def to(self, device):
        for k in ["theta_mean", "theta_std", "x_mean", "x_std"]:
            setattr(self, k, getattr(self, k).to(device))
        return self

    # ---- parameters
    def theta_fwd(self, theta):
        return (torch.as_tensor(theta, dtype=torch.float32) - self.theta_mean) / self.theta_std

    def theta_inv(self, z):
        return z * self.theta_std + self.theta_mean

    def log_jac_theta(self):
        """log |d z_theta / d theta|, constant."""
        return -torch.log(self.theta_std).sum()

    # ---- observations
    def dequantise(self, x, gen=None):
        x = torch.as_tensor(x, dtype=torch.float32)
        return x + torch.rand(x.shape, generator=gen, device=x.device)

    def xtilde_fwd(self, xt):
        """Dequantised counts -> standardised."""
        return (torch.log1p(xt) - self.x_mean) / self.x_std

    def x_fwd(self, x, gen=None):
        """Integer counts -> standardised, with fresh dequantisation noise."""
        return self.xtilde_fwd(self.dequantise(x, gen))

    def xtilde_inv(self, z):
        return torch.expm1(z * self.x_std + self.x_mean)

    def x_inv(self, z):
        """Standardised -> integer counts (floor of the dequantised value)."""
        return torch.clamp(torch.floor(self.xtilde_inv(z)), min=0.0)

    def log_jac_x(self, xt):
        """log |d z_x / d x~| at dequantised counts x~, summed over bins."""
        return (-torch.log1p(xt) - torch.log(self.x_std)).sum(-1)

    # ---- joint
    def joint_fwd(self, theta, x, gen=None):
        return torch.cat([self.theta_fwd(theta), self.x_fwd(x, gen)], dim=-1)

    @staticmethod
    def split(z):
        return z[..., :D_THETA], z[..., D_THETA:]
