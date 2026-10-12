"""Exact reference distributions on a (mu, M, lambda) grid.

The (mu, M) posterior marginal (also with absent bins) and the evidence p(x).
"""

import numpy as np
from scipy.special import logsumexp

from .simulator import (B, S, K, MU_MAX, M_RANGE, LAM_MEAN, LAM_SD,
                        bkg_probs, sig_probs, poisson_loglik)


class GridReference:
    def __init__(self, n_mu=61, n_M=281, n_lam=61, lam_halfwidth=4.5):
        self.mu = np.linspace(0.0, MU_MAX, n_mu)
        self.M = np.linspace(*M_RANGE, n_M)
        self.lam = LAM_MEAN + LAM_SD * np.linspace(-lam_halfwidth, lam_halfwidth, n_lam)
        self.d = np.array([self.mu[1] - self.mu[0], self.M[1] - self.M[0], self.lam[1] - self.lam[0]])
        self.bkg = B * bkg_probs(self.lam)          # (n_lam, K)
        self.sig = S * sig_probs(self.M)            # (n_M, K)
        # log prior density on the grid (mu, M uniform; lambda Gaussian)
        self.log_prior = (-0.5 * ((self.lam - LAM_MEAN) / LAM_SD) ** 2
                          - np.log(LAM_SD * np.sqrt(2 * np.pi))
                          - np.log(MU_MAX) - np.log(M_RANGE[1] - M_RANGE[0]))
        self.log_cell = np.log(self.d).sum()

    def log_lik(self, x, keep=None):
        """log p(x | theta) on the grid, shape (n_mu, n_M, n_lam)."""
        x = np.asarray(x, dtype=float)
        keep = np.ones(K, bool) if keep is None else np.asarray(keep, bool)
        xk, sig = x[keep], self.sig[:, keep]
        out = np.empty((len(self.mu), len(self.M), len(self.lam)))
        for j in range(len(self.lam)):
            nu = self.bkg[j, keep][None, None, :] + self.mu[:, None, None] * sig[None, :, :]
            out[:, :, j] = poisson_loglik(xk, nu)
        return out

    def log_post(self, x, keep=None):
        """Unnormalised log posterior on the grid."""
        return self.log_lik(x, keep) + self.log_prior[None, None, :]

    def log_evidence(self, x, keep=None):
        """log p(x) = log int p(x | theta) pi(theta) dtheta, by the grid sum."""
        return logsumexp(self.log_post(x, keep)) + self.log_cell

    def posterior_mu_M(self, x, keep=None):
        """p(mu, M | x) with lambda integrated out, shape (n_M, n_mu), unit mass on the grid."""
        lp = self.log_post(x, keep)
        lpm = logsumexp(lp, axis=2).T
        p = np.exp(lpm - logsumexp(lpm))
        return p / (p.sum() * self.d[0] * self.d[1])
