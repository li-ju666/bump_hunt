"""One figure showing OneFlowSBI on the bump hunt, against the exact answers.

    uv run python scripts/train.py --budget 30000      # ~10 min
    uv run python scripts/demo.py --run runs/b30000_s0
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oneflow.evaluate import auroc
from oneflow.flow import Flow
from oneflow.reference import GridReference
from oneflow.simulator import EDGES, expected_counts, make_dataset, simulate_hist

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/b30000_s0")
p.add_argument("--out", default="results/demo.png")
a = p.parse_args()

rng = np.random.default_rng(1)
flow, ref = Flow(Path(a.run) / "model.pt"), GridReference()
theta = np.array([1.0, 600.0, 150.0])
x = simulate_hist(theta, rng)

fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))

# (a) the observation
c = np.sqrt(EDGES[1:] * EDGES[:-1])
ax[0].errorbar(c, x, np.sqrt(x), fmt="k.", ms=4, label="one observation")
ax[0].stairs(expected_counts(theta), EDGES, label=r"expected, $\mu=1$, $M=600$")
ax[0].stairs(expected_counts(theta * [0, 1, 1]), EDGES, ls="--", label="background only")
ax[0].set(xscale="log", yscale="log", xlabel="dimuon mass [GeV]", ylabel="events / bin", title="(a) the data")
ax[0].legend(fontsize=8)

# (b) posterior: learned samples (lambda absent) on exact contours
post = flow.posterior(x, 1000, lam="absent")
ax[1].contour(ref.mu, ref.M, ref.posterior_mu_M(x), levels=6, colors="C0", linewidths=1)
ax[1].scatter(post[:, 0], post[:, 1], s=3, alpha=0.3, c="C1", label="OneFlowSBI samples")
ax[1].plot(theta[0], theta[1], "r+", ms=12, mew=2, label="truth")
ax[1].plot([], [], "C0", label="exact posterior")
ax[1].set(xlim=(0, 3), ylim=(300, 1000), xlabel=r"signal strength $\mu$", ylabel="M [GeV]",
          title=r"(b) posterior $p(\mu, M \mid x)$, $\lambda$ marginalised")
ax[1].legend(fontsize=8)

# (c) likelihood: same model, generated histograms vs the simulator
gen = flow.likelihood_samples(theta, 1000)
sim = np.array([simulate_hist(theta, rng) for _ in range(1000)])
ax[2].fill_between(c, np.quantile(gen, 0.05, 0), np.quantile(gen, 0.95, 0), step="mid", alpha=0.35, color="C1",
                   label="OneFlowSBI 90% band")
ax[2].fill_between(c, np.quantile(sim, 0.05, 0), np.quantile(sim, 0.95, 0), step="mid", alpha=0.25, color="C0",
                   label="simulator 90% band")
ax[2].plot(c, expected_counts(theta), "k--", drawstyle="steps-mid", lw=1, label="exact mean")
ax[2].set(xscale="log", yscale="log", xlabel="dimuon mass [GeV]", ylabel="events / bin",
          title=r"(c) likelihood $p(x \mid \theta)$")
ax[2].legend(fontsize=8)

# (d) flagging: learned log p(x) on simulator data vs shifted-simulator data
_, x_in = make_dataset(150, rng)
_, x_out = make_dataset(150, rng, bkg="power")
lq_in, lq_out = flow.log_marginal(x_in), flow.log_marginal(x_out)
ax[3].hist(lq_in, bins=30, alpha=0.6, label="simulator data")
ax[3].hist(lq_out, bins=30, alpha=0.6, label="shifted simulator (power law)")
ax[3].set(xlabel=r"learned $\log p(x)$, $\theta$ absent", ylabel="observations",
          title=f"(d) flagging misspecified data, AUROC {auroc(lq_in, lq_out):.3f}")
ax[3].legend(fontsize=8)

fig.suptitle("One OneFlowSBI model, four queries by changing the mask; blue = exact answer", y=1.02)
fig.tight_layout()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=130, bbox_inches="tight")
print("saved", a.out)
