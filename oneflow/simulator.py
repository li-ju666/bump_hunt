"""
The minimal dimuon bump hunt: simulator and exact likelihood.

theta = (mu, M, lambda)
    mu      signal strength                                   U(0, 3)
    M       Z' mass [GeV]                                     U(300, 1000)
    lambda  background slope [GeV], nuisance                  N(150, 15^2)

One observation:
    N_b ~ Poisson(B),       m ~ m_gen + Exponential(mean lambda)      background
    N_s ~ Poisson(mu * S),  m ~ Normal(M, (w M)^2)                    signal
    keep m_lo <= m < m_hi,  x = histogram(m, log-spaced edges)

Exact likelihood: x_k ~ Poisson(nu_k), nu_k = B P_b(k) + mu S P_s(k), with P_b, P_s
differences of the exponential / normal CDF at the bin edges.
"""

import numpy as np
from scipy.special import ndtr, gammaln

M_GEN, M_LO, M_HI = 200.0, 250.0, 1200.0
K = 48
B, S = 2000.0, 50.0
W = 0.02
POWER = 4.0
MU_MAX, M_RANGE = 3.0, (300.0, 1000.0)
LAM_MEAN, LAM_SD = 150.0, 15.0

EDGES = np.geomspace(M_LO, M_HI, K + 1)
PARAM_NAMES = ["mu", "M", "lambda"]
D_THETA, D_X = 3, K


def sample_prior(n, rng):
    return np.column_stack([rng.uniform(0.0, MU_MAX, n),
                            rng.uniform(*M_RANGE, n),
                            rng.normal(LAM_MEAN, LAM_SD, n)])



def simulate_events(theta, rng, bkg="exp"):
    mu, M, lam = theta
    n_b = rng.poisson(B)
    if bkg == "exp":
        m_b = M_GEN + rng.exponential(lam, n_b)
    elif bkg == "power":
        m_b = M_GEN * (1.0 + rng.pareto(POWER - 1.0, n_b))
    else:
        raise ValueError(bkg)
    n_s = rng.poisson(mu * S)
    m_s = rng.normal(M, W * M, n_s)
    m = np.concatenate([m_b, m_s])
    return m[(m >= M_LO) & (m < M_HI)]


def simulate_hist(theta, rng, **kw):
    return np.histogram(simulate_events(theta, rng, **kw), bins=EDGES)[0]


def make_dataset(n, rng, **kw):
    """theta (n, 3) and integer counts x (n, K)."""
    theta = sample_prior(n, rng)
    x = np.array([simulate_hist(t, rng, **kw) for t in theta], dtype=np.int64)
    return theta, x


def bkg_probs(lam):
    lam = np.asarray(lam, dtype=float)[..., None]
    return np.diff(1.0 - np.exp(-(EDGES - M_GEN) / lam), axis=-1)


def sig_probs(M):
    M = np.asarray(M, dtype=float)[..., None]
    return np.diff(ndtr((EDGES - M) / (W * M)), axis=-1)


def expected_counts(theta):
    mu, M, lam = theta
    return B * bkg_probs(lam) + mu * S * sig_probs(M)


def poisson_loglik(x, nu):
    nu = np.maximum(nu, 1e-300)
    return np.sum(x * np.log(nu) - nu - gammaln(x + 1.0), axis=-1)


def log_lik(x, theta, keep=None):
    """Exact log p(x | theta); keep = boolean mask of bins to use (absent bins dropped)."""
    nu = expected_counts(theta)
    if keep is not None:
        return poisson_loglik(x[keep], nu[keep])
    return poisson_loglik(x, nu)

