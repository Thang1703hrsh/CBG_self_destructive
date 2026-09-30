"""Small FP64 solves for CBG_implementation_spec.md, sections 4--5.

The legacy dual/Rayleigh modules implement a different, unnormalized objective.
This module deliberately keeps that API separate from the specified log-eta loss.
"""
from dataclasses import dataclass
import math
import warnings

import torch
from torch import Tensor


@dataclass(frozen=True)
class SpecSpectrum:
    kernel: Tensor
    m: int
    mu: Tensor
    rho: Tensor
    nu: Tensor
    c: Tensor
    reference_c: Tensor
    log_eta: Tensor
    A: Tensor
    objective: str = "log_eta"
    solver: str = "exact"
    ritz_residual: Tensor | None = None

    @property
    def objective_value(self):
        return self.rho[0] if self.objective == "raw_rho" else self.log_eta

    @property
    def S(self):
        return self.rho[:self.c.shape[1]].sum()

    @property
    def N(self):
        return self.nu[:self.c.shape[1]].sum()

    def diagnostics(self):
        k = self.c.shape[1]
        result = dict(eta_k=self.log_eta.exp(), log_eta_k=self.log_eta,
                      rho_hat=self.rho[0], S_k=self.S, N_k=self.N, mu=self.mu,
                      trace_K_bb=self.kernel[self.m:, self.m:].trace(),
                      trace_K_hh=self.kernel[:self.m, :self.m].trace())
        if self.ritz_residual is not None:
            result["ritz_residual"] = self.ritz_residual
        result["rho_gap_applicable"] = float(self.solver == "exact")
        # Undo the 1/sqrt(batch size) row weighting for the S9 diagnostics.
        result["mean_safety_gradient_metric_norm"] = (self.m * self.kernel.diag()[:self.m]).clamp_min(0).sqrt().mean()
        result["mean_benign_gradient_metric_norm"] = ((len(self.kernel)-self.m) * self.kernel.diag()[self.m:]).clamp_min(0).sqrt().mean()
        for label, values in (("rho", self.rho), ("nu", self.nu)):
            # The implicit parameter-space spectrum has zeros after rank m.
            result[f"{label}_gap"] = (values[k-1] - (values[k] if k < len(values) else 0)
                                       if label != "rho" or self.solver == "exact" else values.new_zeros(()))
            result.update({f"{label}_{i+1}": value for i, value in enumerate(values)})
        return result


@torch.no_grad()
def solve_spec_spectrum(kernel: Tensor, m: int, k: int, *, kappa_mu: float,
                        fixed_mu: Tensor | None = None, objective="log_eta", solver="exact",
                        rayleigh_steps=0, rayleigh_step_size=None, generator=None) -> SpecSpectrum:
    """Exact reduced solve, with detached per-step mu and no hidden ridge.

    fixed_mu is a numerical-oracle option: finite differences of a stop-gradient
    objective must freeze mu at the center, rather than differentiate its rule.
    """
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    if objective not in {"log_eta", "raw_rho"} or (objective == "raw_rho" and k != 1):
        raise ValueError("raw_rho requires k=1; supported objectives are log_eta and raw_rho")
    if solver not in {"exact", "rayleigh_K"}:
        raise ValueError("solver must be exact or rayleigh_K")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (m, k)):
        raise ValueError("m and k must be integers")
    n = len(kernel) - m
    if not 1 <= k <= m or n < 1:
        raise ValueError("require 1 <= k <= m and a nonempty benign batch")
    if not math.isfinite(kappa_mu) or kappa_mu <= 0:
        raise ValueError("kappa_mu must be finite and positive")
    K = kernel.detach().double()
    if not bool(torch.isfinite(K).all()):
        raise FloatingPointError("nonfinite CBG Gram")
    scale = torch.linalg.matrix_norm(K).clamp_min(torch.finfo(K.dtype).tiny)
    if bool(torch.linalg.matrix_norm(K - K.T) > 1e-10 * scale):
        raise FloatingPointError("CBG Gram is asymmetric")
    K = (K + K.T) / 2
    if bool(torch.linalg.eigvalsh(K)[0] < -1e-10 * scale):
        raise FloatingPointError("CBG Gram is not positive semidefinite")
    hh, hb, bb = K[:m, :m], K[:m, m:], K[m:, m:]
    mu = kappa_mu * bb.trace() / n if fixed_mu is None else torch.as_tensor(fixed_mu).to(K).detach()
    if mu.numel() != 1 or not bool(torch.isfinite(mu)) or not bool(mu > 0):
        raise FloatingPointError("mu must be positive; the benign Gram trace may be zero")
    chol = torch.linalg.cholesky(bb + mu * torch.eye(n, device=K.device, dtype=K.dtype))
    solved = torch.cholesky_solve(hb.T, chol)
    M = (hh - hb @ solved) / mu
    M = (M + M.T) / 2
    residual = None
    if solver == "exact":
        rho, y = torch.linalg.eigh(M)
    else:
        if (isinstance(rayleigh_steps, bool) or not isinstance(rayleigh_steps, int) or rayleigh_steps < 1
            or rayleigh_step_size is None or not math.isfinite(rayleigh_step_size) or rayleigh_step_size <= 0):
            raise ValueError("Rayleigh ablation needs positive steps and explicit step size")
        Q = torch.linalg.qr(torch.randn(m, k, device=K.device, dtype=K.dtype, generator=generator), mode="reduced").Q
        # Gradient ascent of tr(Q.T M Q) with a QR retraction. Normalize M by
        # its Frobenius norm so the configured step is dimensionless. This is
        # an explicitly truncated block solver, not an exact eigenvector claim.
        scale_M = torch.linalg.matrix_norm(M).clamp_min(torch.finfo(K.dtype).tiny)
        for _ in range(rayleigh_steps):
            tangent = (M @ Q - Q @ (Q.T @ M @ Q)) / scale_M
            Q = torch.linalg.qr(Q + 2 * rayleigh_step_size * tangent, mode="reduced").Q
        small = Q.T @ M @ Q
        rho, rotation = torch.linalg.eigh((small + small.T) / 2)
        y = Q @ rotation
        residual = torch.linalg.matrix_norm(M @ y - y * rho) / scale_M
    nu, yp = torch.linalg.eigh(hh)
    rho, y = rho.flip(0), y.flip(1)
    nu, yp = nu.flip(0), yp.flip(1)
    if not bool(rho[k-1] > 0) or not bool(nu[k-1] > 0):
        raise FloatingPointError("CBG requires k strictly positive resolved safety modes")
    y = y[:, :k] / rho[:k].sqrt()
    yp = yp[:, :k] / nu[:k].sqrt()
    c = torch.cat((y / mu, -solved @ y / mu))
    cp = torch.cat((yp, K.new_zeros(n, k)))
    S, N = rho[:k].sum(), nu[:k].sum()
    log_eta = (mu * S).log() - N.log()
    if not bool(torch.isfinite(log_eta)) or not bool(0 < log_eta.exp() <= 1 + 1e-6):
        raise FloatingPointError("CBG eta is outside (0, 1]; inspect Gram/solve precision")
    for label, values in (("rho", rho), ("nu", nu)):
        if label == "rho" and solver != "exact":
            continue
        gap = values[k-1] - (values[k] if k < len(values) else 0)
        if bool(gap / values[k-1] < 1e-3):
            warnings.warn(f"small {label} gap at k: using a Clarke subgradient", RuntimeWarning)
    A = -(mu / S) * (c * rho[:k]) @ c.T + (cp * nu[:k]) @ cp.T / N
    if objective == "raw_rho":
        A = -mu * rho[0] * c @ c.T
    return SpecSpectrum(K, m, mu, rho, nu, c, cp, log_eta, A, objective, solver, residual)


@torch.no_grad()
def sign_scalarization(spectrum: SpecSpectrum, s: Tensor, t: Tensor):
    """Return c_w and [g_h; g_b] from section 5.1 (both frozen)."""
    q = spectrum
    k = q.c.shape[1]
    s, t = s.to(q.kernel), t.to(q.kernel)
    if s.shape != (k,) or t.shape != (k,) or not bool(((s.abs() == 1) & (t.abs() == 1)).all()):
        raise ValueError("s and t must be length-k Rademacher vectors")
    cs, ct = s / q.S.sqrt(), t / q.N.sqrt()
    if q.objective == "raw_rho":
        displacement = q.kernel @ q.c[:, 0]
        return q.c[:, 0], torch.cat((displacement[:q.m], -q.rho[0]*displacement[q.m:]))
    cw = q.c @ cs + q.reference_c @ ct
    displacement = q.kernel @ q.c
    reference = q.kernel[:q.m] @ q.reference_c
    gh = displacement[:q.m] @ cs - reference @ ct
    gb = -displacement[q.m:] @ (q.rho[:k] * cs)
    return cw, torch.cat((gh, gb))
