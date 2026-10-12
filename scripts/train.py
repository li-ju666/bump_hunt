"""Simulate a training set from the prior and train one OneFlowSBI model.

    uv run python scripts/train.py --budget 30000 --seed 0
"""

import argparse

import numpy as np

from oneflow.simulator import make_dataset
from oneflow.train import train

p = argparse.ArgumentParser()
p.add_argument("--budget", type=int, default=30_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=50_000)
p.add_argument("--batch", type=int, default=512)
p.add_argument("--device", default="auto")
p.add_argument("--patience", type=int, default=8, help="early stop after this many evaluations without improvement; 0 = off")
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--n-blocks", type=int, default=4)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--posterior-heavy", action="store_true", help="mask mix weighted towards posterior queries")
p.add_argument("--no-structured", action="store_true", help="paper's default mask mix only")
p.add_argument("--out", default=None)
a = p.parse_args()

out = a.out or f"runs/b{a.budget}_s{a.seed}" + ("_nostruct" if a.no_structured else "")
rng = np.random.default_rng(1000 + a.seed)
theta, x = make_dataset(a.budget, rng)
weights = None
if a.no_structured:
    weights = {"joint": 0.15, "posterior": 0.20, "likelihood": 0.20, "marg_x": 0.15, "marg_theta": 0.10, "partial": 0.20}
if a.posterior_heavy:
    weights = {"joint": 0.10, "posterior": 0.35, "likelihood": 0.15, "marg_x": 0.05, "marg_theta": 0.05, "partial": 0.10, "structured": 0.20}
train(theta, x, out, steps=a.steps, batch=a.batch, lr=a.lr, seed=a.seed, device=a.device, mask_weights=weights,
      hidden=a.hidden, n_blocks=a.n_blocks, patience=a.patience or None)
