"""Self-check and figure for the minimal bump hunt. The model lives in oneflow/simulator.py."""

import numpy as np
from oneflow.simulator import *          # noqa: F401,F403
from oneflow.reference import GridReference

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    theta_true = np.array([1.0, 600.0, 150.0])

    nu = expected_counts(theta_true)
    sims = np.array([simulate_hist(theta_true, rng) for _ in range(4000)])
    z = (sims.mean(0) - nu) / (sims.std(0) / np.sqrt(len(sims)))
    print(f"expected selected events {nu.sum():.1f}, simulated mean {sims.sum(1).mean():.1f}")
    print(f"expected vs simulated per bin, max |z| = {np.abs(z).max():.2f} (should be O(1))")

    x = simulate_hist(theta_true, rng)
    ref = GridReference()
    post = ref.posterior_mu_M(x)
    p_mu, p_M = post.sum(0) * ref.d[1], post.sum(1) * ref.d[0]
    print(f"observation: {x.sum()} events; log L at truth {log_lik(x, theta_true):.1f}, "
          f"at mu=0 {log_lik(x, theta_true * [0, 1, 1]):.1f}")
    print(f"posterior mean mu = {np.sum(ref.mu * p_mu) * ref.d[0]:.2f} (truth {theta_true[0]}), "
          f"M = {np.sum(ref.M * p_M) * ref.d[1]:.0f} (truth {theta_true[1]:.0f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    c = np.sqrt(EDGES[1:] * EDGES[:-1])
    nu0 = expected_counts(theta_true * [0, 1, 1])
    ax[0].errorbar(c, x, np.sqrt(x), fmt="k.", ms=4, label="one observation")
    ax[0].stairs(nu, EDGES, label=r"expected, $\mu=1$, $M=600$")
    ax[0].stairs(nu0, EDGES, ls="--", label="background only")
    ax[0].set(xscale="log", yscale="log", xlabel=r"dimuon mass $m$ [GeV]", ylabel="events / bin",
              title="the data: a bump on a falling background")
    ax[0].legend(fontsize=8)
    ax[1].contourf(ref.mu, ref.M, post, levels=20, cmap="Blues")
    ax[1].plot(theta_true[0], theta_true[1], "r+", ms=12, mew=2, label="truth")
    ax[1].set(xlabel=r"signal strength $\mu$", ylabel=r"$M$ [GeV]",
              title=r"exact posterior $p(\mu, M \mid x)$, $\lambda$ marginalised")
    ax[1].legend()
    x_ood = simulate_hist(theta_true, rng, bkg="power")
    ax[2].stairs(x, EDGES, label="simulator (exponential background)")
    ax[2].stairs(x_ood, EDGES, label="shifted simulator (power-law background)")
    ax[2].set(xscale="log", yscale="log", xlabel=r"dimuon mass $m$ [GeV]", ylabel="events / bin",
              title="in-distribution vs. misspecified data")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("minimal_check.png", dpi=130)
    print("saved minimal_check.png")
