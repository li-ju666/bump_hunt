"""Precompute everything the interactive results section needs from one trained model.

    uv run python scripts/export_interactive.py --run runs/b30000_s0

Writes results/interactive_data.json: for a grid of true (mu, M) scenarios, the
observation, exact expected counts, learned and exact posteriors (also with a window of
bins absent), learned and simulated likelihood bands, and the flagging scores.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from oneflow.evaluate import auroc
from oneflow.flow import Flow
from oneflow.reference import GridReference
from oneflow.simulator import EDGES, K, expected_counts, make_dataset, simulate_hist

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/b30000_s0")
p.add_argument("--out", default="results/interactive_data.json")
p.add_argument("--n-post", type=int, default=800)
a = p.parse_args()

rng = np.random.default_rng(1)
flow = Flow(Path(a.run) / "model.pt")
ref = GridReference(n_mu=61, n_M=141, n_lam=41)
MUS, MS, LAM = [0.0, 0.5, 1.0, 2.0], [400.0, 600.0, 800.0], 150.0
QS = [0.05, 0.5, 0.95]


def window_for(M):
    """Eight bins centred on the true mass."""
    i = int(np.searchsorted(EDGES, M) - 1)
    return max(0, i - 4), min(K, i + 4)


def offpeak_for(M):
    """Eight bins well away from the peak: twelve bins below it, or above it when the
    mass is near the lower edge of the window."""
    i = int(np.searchsorted(EDGES, M) - 1)
    s = i - 16
    if s < 0:
        s = i + 8
    return s, s + 8


def q(arr, nd=3):
    return np.round(np.asarray(arr, float), nd).tolist()


scenarios = []
for mu in MUS:
    for M in MS:
        theta = np.array([mu, M, LAM])
        x = simulate_hist(theta, rng)
        s, e = window_for(M)
        masks = {"none": None, "peak": (s, e), "offpeak": offpeak_for(M)}
        post, exact = {}, {}
        for name, w in masks.items():
            absent = np.zeros(K, bool)
            if w:
                absent[w[0]:w[1]] = True
            smp = flow.posterior(x, a.n_post, absent_bins=absent if w else None, lam="absent")
            post[name] = q(smp[:, :2], 3)
            exact[name] = q(ref.posterior_mu_M(x, keep=None if not w else ~absent), 5)
        gen = flow.likelihood_samples(theta, 1000)
        sim = np.array([simulate_hist(theta, rng) for _ in range(1000)])
        scenarios.append({
            "mu": mu, "M": M, "lam": LAM, "x": x.tolist(),
            "nu": q(expected_counts(theta)), "nu0": q(expected_counts(theta * [0, 1, 1])),
            "windows": {"peak": [s, e], "offpeak": list(offpeak_for(M))},
            "post": post, "exact": exact,
            "lik_learned": {str(qq): q(np.quantile(gen, qq, 0), 2) for qq in QS},
            "lik_sim": {str(qq): q(np.quantile(sim, qq, 0), 2) for qq in QS},
        })
        print(f"scenario mu={mu} M={M:.0f} done", flush=True)

_, x_in = make_dataset(200, rng)
_, x_out = make_dataset(200, rng, bkg="power")
lq_in, lq_out = flow.log_marginal(x_in), flow.log_marginal(x_out)
le_in = np.array([ref.log_evidence(x) for x in x_in])
le_out = np.array([ref.log_evidence(x) for x in x_out])

data = {
    "edges": q(EDGES, 2), "mu_grid": q(ref.mu, 3), "M_grid": q(ref.M, 2),
    "scenarios": scenarios,
    "flag": {"learned_in": q(lq_in, 2), "learned_out": q(lq_out, 2), "exact_in": q(le_in, 2), "exact_out": q(le_out, 2),
             "auroc_learned": auroc(lq_in, lq_out), "auroc_exact": auroc(le_in, le_out)},
    "model": {**flow.config, "best_step": flow.best_step},
}
Path(a.out).write_text(json.dumps(data, separators=(",", ":")))
print("wrote", a.out, f"{Path(a.out).stat().st_size / 1e6:.1f} MB")
