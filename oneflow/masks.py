"""Ternary query masks over the joint state (mu, M, lambda, x_1..x_K).

    -1 absent (marginalised), 0 generated, 1 observed.

Families and default weights follow the OneFlowSBI paper (joint, posterior, likelihood,
two marginals, random partial), plus a 'structured' family of physics queries:
posterior with a contiguous window of absent bins, posterior given lambda, the
(mu, M) marginal posterior with lambda absent, and the null predictive p(x | mu)
with M and lambda absent.
"""

import numpy as np

from .simulator import D_THETA, D_X

ABSENT, GEN, OBS = -1, 0, 1
D = D_THETA + D_X

DEFAULT_WEIGHTS = {
    "joint": 0.15, "posterior": 0.20, "likelihood": 0.20,
    "marg_x": 0.10, "marg_theta": 0.05, "partial": 0.15, "structured": 0.15,
}


def canonical(name):
    m = np.full(D, GEN, dtype=np.int64)
    th, x = slice(0, D_THETA), slice(D_THETA, D)
    if name == "joint":
        pass
    elif name == "posterior":
        m[x] = OBS
    elif name == "likelihood":
        m[th] = OBS
    elif name == "marg_x":
        m[th] = ABSENT
    elif name == "marg_theta":
        m[x] = ABSENT
    elif name == "post_marg_lam":          # p(mu, M | x)
        m[x] = OBS
        m[2] = ABSENT
    elif name == "post_given_lam":         # p(mu, M | x, lambda)
        m[x] = OBS
        m[2] = OBS
    elif name == "null_pred":              # p(x | mu), M and lambda absent
        m[0] = OBS
        m[1:3] = ABSENT
    else:
        raise ValueError(name)
    return m


def missing_window(start, length):
    """Posterior mask with bins [start, start+length) absent."""
    m = canonical("posterior")
    m[D_THETA + start: D_THETA + start + length] = ABSENT
    return m


def _partial(rng):
    m = np.empty(D, dtype=np.int64)
    for sl in (slice(0, D_THETA), slice(D_THETA, D)):
        p = rng.dirichlet([1.0, 1.0, 1.0])
        m[sl] = rng.choice([ABSENT, GEN, OBS], size=sl.stop - sl.start, p=p)
    if not (m == GEN).any():
        m[rng.integers(D)] = GEN
    return m


def _structured(rng):
    kind = rng.integers(4)
    if kind == 0:
        length = rng.integers(4, 17)
        return missing_window(rng.integers(0, D_X - length + 1), length)
    return canonical(["post_given_lam", "post_marg_lam", "null_pred"][kind - 1])


def sample_masks(n, rng, weights=None):
    w = DEFAULT_WEIGHTS if weights is None else weights
    names = list(w)
    p = np.array([w[k] for k in names]) / sum(w.values())
    fam = rng.choice(len(names), size=n, p=p)
    out = np.empty((n, D), dtype=np.int64)
    for i, f in enumerate(fam):
        name = names[f]
        if name == "partial":
            out[i] = _partial(rng)
        elif name == "structured":
            out[i] = _structured(rng)
        else:
            out[i] = canonical(name)
    return out
