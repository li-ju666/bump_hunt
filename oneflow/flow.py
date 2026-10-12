"""Querying a trained OneFlowSBI model: masked sampling by ODE integration, and the
log density of an observation (all parameters absent) by the instantaneous change of
variables, in physical units."""

import numpy as np
import torch
from torch.func import jacrev, vmap

from .data import Transform
from .masks import canonical, GEN, OBS, ABSENT, D
from .model import VelocityNet
from .simulator import D_THETA, D_X

LOG_2PI = float(np.log(2 * np.pi))


class Flow:
    def __init__(self, ckpt, device="cpu"):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False) if not isinstance(ckpt, dict) else ckpt
        c = ck["config"]
        self.device = torch.device(device)
        self.model = VelocityNet(c["d"], c["hidden"], c["n_blocks"]).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tf = Transform.from_state_dict(ck["transform"]).to(self.device)
        self.config = c
        self.best_step = ck.get("best_step")
        self.stopped_at = ck.get("stopped_at")

    # ------------------------------------------------------------------ core ODE
    @torch.no_grad()
    def sample_z(self, mask, cond, steps=20, seed=None):
        """Euler-integrate the generated coordinates from noise at t=0 to t=1.
        mask (n, D) long; cond (n, D) standardised values, used on observed coords.
        seed fixes the base noise, so samples vary continuously with the conditioning."""
        m_gen = (mask == GEN).float()
        m_obs = (mask == OBS).float()
        gen = None if seed is None else torch.Generator(device=self.device).manual_seed(int(seed))
        z = m_gen * torch.randn(cond.shape, generator=gen, device=self.device) + m_obs * cond
        h = 1.0 / steps
        for i in range(steps):
            t = torch.full((z.shape[0],), i * h, device=self.device)
            z = z + h * m_gen * self.model(z, t, mask)
        return z

    def _v_div(self, z, t, mask, m_gen):
        """Velocity and its exact divergence over generated coordinates (vectorised Jacobian)."""
        def f(zi, ti, mi):
            v = self.model(zi[None], ti[None], mi[None])[0]
            return v, v
        J, v = vmap(jacrev(f, has_aux=True))(z, t, mask)
        return v, (torch.diagonal(J, dim1=-2, dim2=-1) * m_gen).sum(-1)

    @torch.no_grad()
    def log_density_z(self, mask, z1, steps=100):
        """log q(z_G | z_O) for the generated block of z1 under mask: integrate the flow
        backwards from t=1 to t=0 with RK4, accumulating the divergence."""
        m_gen = (mask == GEN).float()
        z, a = z1.clone(), torch.zeros(z1.shape[0], device=self.device)
        h = -1.0 / steps

        def rhs(z, t):
            v, div = self._v_div(z, t, mask, m_gen)
            return m_gen * v, div

        for i in range(steps):
            t = torch.full((z.shape[0],), 1.0 + i * h, device=self.device)
            k1z, k1a = rhs(z, t)
            k2z, k2a = rhs(z + 0.5 * h * k1z, t + 0.5 * h)
            k3z, k3a = rhs(z + 0.5 * h * k2z, t + 0.5 * h)
            k4z, k4a = rhs(z + h * k3z, t + h)
            z = z + h / 6 * (k1z + 2 * k2z + 2 * k3z + k4z)
            a = a + h / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)
        log_p0 = (-0.5 * (z ** 2 + LOG_2PI) * m_gen).sum(-1)
        return log_p0 + a

    # --------------------------------------------------------------- helpers
    def _mask(self, m, n):
        return torch.as_tensor(np.broadcast_to(np.asarray(m), (n, D)).copy(), device=self.device)

    def _cond(self, theta, x, n, gen=None):
        th = torch.zeros(n, D_THETA, device=self.device)
        if theta is not None:
            th = self.tf.theta_fwd(torch.as_tensor(np.broadcast_to(np.asarray(theta, float), (n, D_THETA)).copy(), device=self.device))
        xz = torch.zeros(n, D_X, device=self.device)
        if x is not None:
            xt = self.tf.dequantise(torch.as_tensor(np.broadcast_to(np.asarray(x, float), (n, D_X)).copy(), device=self.device), gen)
            xz = self.tf.xtilde_fwd(xt)
        return torch.cat([th, xz], -1)

    # --------------------------------------------------------------- queries
    def posterior(self, x, n, absent_bins=None, lam="gen", steps=20, seed=None):
        """Samples of theta given counts x: (n, 3) for x of shape (K,), (B, n, 3) for (B, K).
        lam: 'gen' (generate and keep), 'absent' (marginalise; lambda column is NaN),
        or a float (condition on that value). absent_bins: boolean (K,) bins to drop."""
        x = np.asarray(x, float)
        single = x.ndim == 1
        xb = np.repeat(x[None] if single else x, n, axis=0)
        Bn = xb.shape[0]
        m = canonical("posterior")
        if absent_bins is not None:
            m[D_THETA:][np.asarray(absent_bins, bool)] = ABSENT
        theta_cond = None
        if lam == "absent":
            m[2] = ABSENT
        elif lam != "gen":
            m[2] = OBS
            theta_cond = np.array([0.0, 0.0, float(lam)])
        mask = self._mask(m, Bn)
        gen = None if seed is None else torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        cond = self._cond(theta_cond, xb, Bn, gen)
        z = self.sample_z(mask, cond, steps, seed)
        th = self.tf.theta_inv(z[:, :D_THETA]).cpu().numpy()
        if lam == "absent":
            th[:, 2] = np.nan
        return th if single else th.reshape(-1, n, D_THETA)

    def likelihood_samples(self, theta, n, steps=20, seed=None):
        """Integer histograms generated at theta, shape (n, K)."""
        mask = self._mask(canonical("likelihood"), n)
        cond = self._cond(theta, None, n)
        z = self.sample_z(mask, cond, steps, seed)
        return self.tf.x_inv(z[:, D_THETA:]).cpu().numpy()

    def log_marginal(self, x, n_deq=4, steps=100):
        """Learned log p(x) with all parameters absent, in count space: the flagging score.
        x (B, K) integer counts; logmeanexp over n_deq dequantisations."""
        x = torch.as_tensor(np.asarray(x, float), device=self.device)
        Bn = x.shape[0]
        mask = self._mask(canonical("marg_x"), Bn)
        out = []
        for _ in range(n_deq):
            xt = self.tf.dequantise(x)
            z1 = torch.cat([torch.zeros(Bn, D_THETA, device=self.device), self.tf.xtilde_fwd(xt)], -1)
            out.append(self.log_density_z(mask, z1, steps) + self.tf.log_jac_x(xt))
        return (torch.logsumexp(torch.stack(out, 0), 0) - np.log(n_deq)).cpu().numpy()
