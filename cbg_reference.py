"""CBG v2 reference implementation (PyTorch), with an end-to-end self-test.

This file mirrors docs/CBG_v2_method_spec.md step by step on a tiny causal LM in float64.
It is a correctness reference, not a scalable implementation: on a real model the same
steps run with chunked per-example gradients, bf16 compute and fp32 master weights.

Run:  python cbg_reference.py        (all checks print PASS/FAIL, takes about a minute on CPU)

Pipeline of one CBG step (Algorithm 1 of the paper, v2)
  1. center pass     : forward at theta, sample y_hat, form F~ ; hooks store X (inputs) of every Linear in U
  2. backward of 1'F~: hooks store Delta (output grads); Gram K from (X, Delta)
  3. small solve     : mu, S_mu, eigenpairs, eta_k, G           (all (m+n) x (m+n) or smaller)
  4. direction       : sign pair (a, b); w = Lambda^{-1} C'b and d_Lambda = 2 theta . diag(C'GC) from (X, Delta)
  5. radius          : eps = r * sqrt(dim U / b'Kb)
  6. two passes      : sigma = +1, -1 at theta + sigma*eps*w ; grad of sigma*beta/eps * a'F~ + 1/2 (L_aln + alpha L_bgn)
  7. update          : g + beta * d_Lambda
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as Fn

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------------
# Tiny causal LM with the Llama layer pattern (q,k,v,o + SwiGLU gate/up/down, RMSNorm)
# ----------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)

    def forward(self, x):
        T = x.shape[1]
        y = self.n1(x)
        q, k, v = self.q_proj(y), self.k_proj(y), self.v_proj(y)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        mask = torch.ones(T, T, dtype=torch.bool).triu(1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        x = x + self.o_proj(att @ v)
        y = self.n2(x)
        return x + self.down_proj(Fn.silu(self.gate_proj(y)) * self.up_proj(y))


class TinyLM(nn.Module):
    def __init__(self, vocab=40, d=12, h=20, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.norm(x))


def admissible_linears(model: nn.Module) -> list[nn.Linear]:
    """U = weights of every nn.Linear inside the transformer blocks (not embeddings, norms, head)."""
    return [mod for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and name.startswith("blocks.")]


# ----------------------------------------------------------------------------------
# Batch layout: sequences = [y^h_0, y^s_0, ..., y^h_{m-1}, y^s_{m-1}, benign_0, ..., benign_{n-1}]
# ----------------------------------------------------------------------------------
@dataclass
class Batch:
    tokens: torch.Tensor          # (2m+n, T) prompt + response, equal length in this test
    prompt_len: int
    m: int
    n: int
    seq_example: list = field(default_factory=list)   # example index e of each sequence

    @property
    def E(self):
        return self.m + self.n


def make_batch(vocab, m, n, Lp, Lr, gen):
    toks = torch.randint(0, vocab, (2 * m + n, Lp + Lr), generator=gen)
    for i in range(m):                       # y^h and y^s share the prompt x_i
        toks[2 * i + 1, :Lp] = toks[2 * i, :Lp]
    seq_example = [i for i in range(m) for _ in range(2)] + [m + j for j in range(n)]
    return Batch(toks, Lp, m, n, seq_example)


def response_logprobs(logits, tokens, Lp, targets=None):
    """log p(target_t | prefix) at every response position; targets default to the tokens."""
    lp = logits[:, Lp - 1:-1].log_softmax(-1)
    tgt = tokens[:, Lp:] if targets is None else targets
    return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)          # (B, Lr)


def sample_y_hat(model, batch, gen):
    """Sampled-Fisher targets: y_hat_t ~ p_theta(. | x, y_<t) under teacher forcing on y (no grad)."""
    with torch.no_grad():
        logits = model(batch.tokens[2 * batch.m:])
        probs = logits[:, batch.prompt_len - 1:-1].softmax(-1)
        B, L, V = probs.shape
        return torch.multinomial(probs.reshape(-1, V), 1, generator=gen).reshape(B, L)


def measurements(logits, batch, y_hat):
    """F~ = (h / sqrt(m) ; F_b / sqrt(n)) in example order e = 0..m+n-1."""
    m, n, Lp = batch.m, batch.n, batch.prompt_len
    lp_s = response_logprobs(logits[:2 * m], batch.tokens[:2 * m], Lp).mean(-1)   # token-normalized
    h = lp_s[0::2] - lp_s[1::2]                                                    # harmful - safe
    F_b = response_logprobs(logits[2 * m:], batch.tokens[2 * m:], Lp, y_hat).sum(-1)
    return torch.cat([h / math.sqrt(m), F_b / math.sqrt(n)])


def base_losses(logits, batch, ref_logits_benign):
    """L_aln: SFT on y^s ; L_bgn: token-level KL(p_theta0 || p_theta) on benign data."""
    m, Lp = batch.m, batch.prompt_len
    l_aln = -response_logprobs(logits[1:2 * m:2], batch.tokens[1:2 * m:2], Lp).mean()
    lp = logits[2 * m:, Lp - 1:-1].log_softmax(-1)
    lp0 = ref_logits_benign[:, Lp - 1:-1].log_softmax(-1)
    l_bgn = (lp0.exp() * (lp0 - lp)).sum(-1).mean()
    return l_aln, l_bgn


# ----------------------------------------------------------------------------------
# Step 1-2: capture (X, Delta) and build the Gram
# ----------------------------------------------------------------------------------
class Capture:
    """Forward hooks keep each Linear's input X; backward hooks keep its output gradient Delta."""

    def __init__(self, linears):
        self.linears, self.X, self.D, self.handles = linears, {}, {}, []
        for lin in linears:
            self.handles.append(lin.register_forward_hook(self._fwd))
            self.handles.append(lin.register_full_backward_hook(self._bwd))

    def _fwd(self, mod, inp, out):
        self.X[mod] = inp[0].detach()

    def _bwd(self, mod, gin, gout):
        self.D[mod] = gout[0].detach()

    def remove(self):
        for h in self.handles:
            h.remove()


def per_example_grads(cap, lin, batch):
    """Layer block of every c_e: sum over the sequences of example e of Delta_s' X_s  -> (E, d_out, d_in)."""
    X, D = cap.X[lin], cap.D[lin]
    per_seq = torch.einsum("sto,sti->soi", D, X)
    out = torch.zeros(batch.E, *per_seq.shape[1:])
    out.index_add_(0, torch.tensor(batch.seq_example), per_seq)
    return out


def gram_from_capture(cap, batch, xi):
    """K_ee' = sum over layers of < (theta^2 + xi^2) . c_e , c_e' >   (float64)."""
    K = torch.zeros(batch.E, batch.E)
    for lin in cap.linears:
        Ce = per_example_grads(cap, lin, batch).flatten(1)
        lam_inv = (lin.weight.detach() ** 2 + xi ** 2).flatten()
        K += (Ce * lam_inv) @ Ce.T
    return 0.5 * (K + K.T)


# ----------------------------------------------------------------------------------
# Step 3: the small solve (identical math to verify_cbg_v2.py)
# ----------------------------------------------------------------------------------
@dataclass
class Spectral:
    mu: float
    sigma: torch.Tensor      # eigenvalues of S_mu, descending
    nu: torch.Tensor         # eigenvalues of K_hh, descending
    eta: float
    G: torch.Tensor          # d log eta_k / d K  (E x E)
    c_mu: float
    gap_sigma: float
    gap_nu: float


def spectral(K, m, n, k, kappa_mu, mu_mode="differentiated"):
    Khh, Khb, Kbb = K[:m, :m], K[:m, m:], K[m:, m:]
    mu = kappa_mu * torch.trace(Kbb).item() / n
    L = torch.linalg.cholesky(mu * torch.eye(n) + Kbb)
    Rb = torch.cholesky_solve(Khb.T, L)                      # (mu I + K_bb)^{-1} K_bh
    S = Khh - Khb @ Rb
    S = 0.5 * (S + S.T)
    sig, Phi = torch.linalg.eigh(S)
    nu, Phi2 = torch.linalg.eigh(Khh)
    sig, Phi, nu, Phi2 = sig.flip(0), Phi.flip(1), nu.flip(0), Phi2.flip(1)
    Z = Rb @ Phi[:, :k]                                      # z_j
    Psi = torch.cat([Phi[:, :k], -Z])                        # psi_j = (phi_j ; -z_j)
    Psi2 = torch.cat([Phi2[:, :k], torch.zeros(n, k)])       # psi'_j = (phi'_j ; 0)
    sS, sN = sig[:k].sum(), nu[:k].sum()
    G = Psi @ Psi.T / sS - Psi2 @ Psi2.T / sN
    c_mu = ((Z ** 2).sum() / sS).item()
    if mu_mode == "differentiated":
        G[m:, m:] += (kappa_mu / n) * c_mu * torch.eye(n)
    G = 0.5 * (G + G.T)
    gs = (sig[k - 1] - sig[k]).item() if k < m else float("inf")
    gn = (nu[k - 1] - nu[k]).item() if k < m else float("inf")
    return Spectral(mu, sig, nu, (sS / sN).item(), G, c_mu, gs, gn)


def sign_pair(G, s):
    """Eigen-sign pair with E[a b'] = G for Rademacher s."""
    gam, V = torch.linalg.eigh(G)
    sq = gam.abs().sqrt()
    return V @ (sq * s), V @ (gam.sign() * sq * s)


# ----------------------------------------------------------------------------------
# Step 4: direction w = Lambda^{-1} C' b and d_Lambda = 2 theta . diag(C' G C), layer by layer
# ----------------------------------------------------------------------------------
def direction_and_dlam(cap, batch, G, b, xi):
    w, dlam = {}, {}
    for lin in cap.linears:
        Ce = per_example_grads(cap, lin, batch)                 # (E, out, in); chunk rows on a real model
        th = lin.weight.detach()
        lam_inv = th ** 2 + xi ** 2
        w[lin] = lam_inv * torch.einsum("e,eoi->oi", b, Ce)
        M = torch.einsum("ef,foi->eoi", G, Ce)                  # sum_e' G_ee' c_e'
        dlam[lin] = 2 * th * (Ce * M).sum(0)
    return w, dlam


# ----------------------------------------------------------------------------------
# Steps 1-7: one CBG step (returns the gradient; the caller applies the optimizer)
# ----------------------------------------------------------------------------------
@dataclass
class CBGConfig:
    k: int = 2
    kappa_mu: float = 1.0
    xi: float = 1e-3
    radius: float = 1e-4
    alpha: float = 1.0
    beta: float = 0.1
    mu_mode: str = "differentiated"


def cbg_step_grad(model, ref_model, batch, cfg, gen, s=None, y_hat=None, return_parts=False):
    params = [p for p in model.parameters() if p.requires_grad]
    linears = admissible_linears(model)
    dimU = sum(l.weight.numel() for l in linears)
    if y_hat is None:
        y_hat = sample_y_hat(model, batch, gen)
    with torch.no_grad():
        ref_logits = ref_model(batch.tokens[2 * batch.m:])

    # 1-2. center pass with capture
    cap = Capture(linears)
    model.zero_grad(set_to_none=True)
    Ft = measurements(model(batch.tokens), batch, y_hat)
    Ft.sum().backward()
    cap.remove()
    K = gram_from_capture(cap, batch, cfg.xi)

    # 3. small solve
    sp = spectral(K, batch.m, batch.n, cfg.k, cfg.kappa_mu, cfg.mu_mode)

    # 4. direction and Lambda term
    if s is None:
        s = torch.randint(0, 2, (batch.E,), generator=gen).double() * 2 - 1
    a, b = sign_pair(sp.G, s)
    w, dlam = direction_and_dlam(cap, batch, sp.G, b, cfg.xi)

    # 5. radius from the Gram
    eps = cfg.radius * math.sqrt(dimU / (b @ K @ b).item())

    # 6. two perturbed passes (in-place on the master weights, exact restore from a copy)
    backup = {lin: lin.weight.detach().clone() for lin in linears}
    g = [torch.zeros_like(p) for p in params]
    g_curv = [torch.zeros_like(p) for p in params]
    for sigma in (+1.0, -1.0):
        with torch.no_grad():
            for lin in linears:
                lin.weight.copy_(backup[lin] + sigma * eps * w[lin])
        logits = model(batch.tokens)
        Omega = a @ measurements(logits, batch, y_hat)
        l_aln, l_bgn = base_losses(logits, batch, ref_logits)
        gc = torch.autograd.grad(sigma * cfg.beta / eps * Omega, params, retain_graph=True)
        gb = torch.autograd.grad(0.5 * (l_aln + cfg.alpha * l_bgn), params)
        for i in range(len(params)):
            g[i] += gc[i] + gb[i]
            g_curv[i] += gc[i]
    with torch.no_grad():
        for lin in linears:
            lin.weight.copy_(backup[lin])

    # 7. add beta * d_Lambda on U
    idx = {id(p): i for i, p in enumerate(params)}
    for lin in linears:
        g[idx[id(lin.weight)]] += cfg.beta * dlam[lin]
        g_curv[idx[id(lin.weight)]] += cfg.beta * dlam[lin]
    if return_parts:
        return g, dict(K=K, sp=sp, a=a, b=b, eps=eps, g_curv=g_curv, y_hat=y_hat, w=w, dlam=dlam, cap=cap)
    return g, sp


# ----------------------------------------------------------------------------------
# Measurement only (for D_geo evaluation batches)
# ----------------------------------------------------------------------------------
def measure_geometry(model, batch, cfg, gen):
    y_hat = sample_y_hat(model, batch, gen)
    cap = Capture(admissible_linears(model))
    model.zero_grad(set_to_none=True)
    measurements(model(batch.tokens), batch, y_hat).sum().backward()
    cap.remove()
    model.zero_grad(set_to_none=True)
    K = gram_from_capture(cap, batch, cfg.xi)
    sp = spectral(K, batch.m, batch.n, cfg.k, cfg.kappa_mu, cfg.mu_mode)
    Kbb = K[batch.m:, batch.m:]
    return dict(eta_k=sp.eta, log_eta_k=math.log(sp.eta), rho_mu=sp.sigma[0].item() / sp.mu,
                sigma_1k=sp.sigma[:cfg.k].sum().item(), nu_1k=sp.nu[:cfg.k].sum().item(),
                tr_Kbb=torch.trace(Kbb).item(), mu=sp.mu, c_mu=sp.c_mu,
                eff_rank_Kbb=(torch.trace(Kbb) ** 2 / torch.trace(Kbb @ Kbb)).item(),
                gap_sigma=sp.gap_sigma, gap_nu=sp.gap_nu)


# ----------------------------------------------------------------------------------
# Per-example variant (ablation x): each example at its own perturbation, no sign variance
# ----------------------------------------------------------------------------------
def per_example_curvature(model, batch, cfg, y_hat):
    """Returns d log eta_k / d theta estimated with per-example perturbations theta +- eps_e w_e.

    A real implementation applies x_e (theta +- eps_e w_e)' as an additive branch inside one batched
    pass; here each example is simply re-evaluated with functional_call.
    """
    from torch.func import functional_call
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    linears = admissible_linears(model)
    name_of = {mod: name for name, mod in model.named_modules()}
    dimU = sum(l.weight.numel() for l in linears)

    cap = Capture(linears)
    model.zero_grad(set_to_none=True)
    measurements(model(batch.tokens), batch, y_hat).sum().backward()
    cap.remove()
    model.zero_grad(set_to_none=True)
    K = gram_from_capture(cap, batch, cfg.xi)
    sp = spectral(K, batch.m, batch.n, cfg.k, cfg.kappa_mu, cfg.mu_mode)
    G = sp.G
    GKG = G @ K @ G
    Ce = {lin: per_example_grads(cap, lin, batch) for lin in linears}
    total = {n: torch.zeros_like(p) for n, p in params.items()}
    for e in range(batch.E):
        eps_e = cfg.radius * math.sqrt(dimU / GKG[e, e].item())
        w_e = {lin: (lin.weight.detach() ** 2 + cfg.xi ** 2) * torch.einsum("f,foi->oi", G[e], Ce[lin])
               for lin in linears}
        for sigma in (+1.0, -1.0):
            p_sig = {n: p for n, p in params.items()}
            for lin in linears:
                p_sig[name_of[lin] + ".weight"] = lin.weight + sigma * eps_e * w_e[lin]
            Fe = measurements(functional_call(model, p_sig, (batch.tokens,)), batch, y_hat)[e]
            grads = torch.autograd.grad(sigma / eps_e * Fe, list(params.values()))
            for (n, _), gr in zip(params.items(), grads):
                total[n] += gr
    _, dlam = direction_and_dlam(cap, batch, G, torch.zeros(batch.E), cfg.xi)
    for lin in linears:
        total[name_of[lin] + ".weight"] += dlam[lin]
    return [total[n] for n in params]


# ----------------------------------------------------------------------------------
# Autograd reference: exact d log eta_k / d theta by double backward (tiny models only)
# ----------------------------------------------------------------------------------
def reference_grad(model, batch, cfg, y_hat):
    params = [p for p in model.parameters() if p.requires_grad]
    linears = admissible_linears(model)
    Uw = [l.weight for l in linears]
    Ft = measurements(model(batch.tokens), batch, y_hat)
    rows = []
    for e in range(batch.E):
        ge = torch.autograd.grad(Ft[e], Uw, create_graph=True, retain_graph=True)
        rows.append(torch.cat([g.flatten() for g in ge]))
    C = torch.stack(rows)
    lam_inv = torch.cat([(wt ** 2 + cfg.xi ** 2).flatten() for wt in Uw])
    K = (C * lam_inv) @ C.T
    m, n, k = batch.m, batch.n, cfg.k
    Kbb = K[m:, m:]
    mu = cfg.kappa_mu * torch.trace(Kbb) / n
    if cfg.mu_mode == "detached":
        mu = mu.detach()
    S = K[:m, :m] - K[:m, m:] @ torch.linalg.solve(mu * torch.eye(n) + Kbb, K[m:, :m])
    S = 0.5 * (S + S.T)
    Khh = 0.5 * (K[:m, :m] + K[:m, :m].T)
    log_eta = (torch.log(torch.linalg.eigvalsh(S).flip(0)[:k].sum())
               - torch.log(torch.linalg.eigvalsh(Khh).flip(0)[:k].sum()))
    return torch.autograd.grad(log_eta, params), K.detach(), log_eta.item()


def flat(gs):
    return torch.cat([g.flatten() for g in gs])


def rel(a, b):
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()


# ----------------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    m, n = 3, 6
    model = TinyLM()
    ref_model = TinyLM()
    ref_model.load_state_dict(model.state_dict())
    with torch.no_grad():                        # move model away from theta_0 so L_bgn is non-trivial
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    batch = make_batch(40, m, n, Lp=4, Lr=5, gen=gen)
    results = []

    def check(name, val, tol):
        ok = val <= tol
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name:62s} {val:.3e}  (tol {tol:.0e})")

    for mu_mode in ("differentiated", "detached"):
        cfg = CBGConfig(mu_mode=mu_mode, radius=1e-5, beta=1.0)
        y_hat = sample_y_hat(model, batch, gen)
        ref, K_ref, log_eta_ref = reference_grad(model, batch, cfg, y_hat)
        ref = flat(ref)

        # Gram from hooks == Gram from per-example autograd
        g, parts = cbg_step_grad(model, ref_model, batch, cfg, gen, y_hat=y_hat, return_parts=True)
        check(f"[{mu_mode}] Gram from (X,Delta) vs autograd", rel(parts["K"], K_ref), 1e-12)
        check(f"[{mu_mode}] log eta_k vs autograd", abs(math.log(parts["sp"].eta) - log_eta_ref), 1e-12)

        # expectation over all 2^E sign draws of the curvature estimate + d_Lambda == exact gradient
        acc = torch.zeros_like(ref)
        single = []
        for s in itertools.product([-1.0, 1.0], repeat=batch.E):
            _, pr = cbg_step_grad(model, ref_model, batch, cfg, gen, s=torch.tensor(s), y_hat=y_hat,
                                  return_parts=True)
            est = flat(pr["g_curv"])
            acc += est
            single.append(rel(est, ref))
        mean_est = acc / 2 ** batch.E
        check(f"[{mu_mode}] E_s[estimate] vs autograd d log eta / d theta", rel(mean_est, ref), 1e-6)
        print(f"      single-draw relative error: mean {sum(single) / len(single):.2f}")

        pe = flat(per_example_curvature(model, batch, cfg, y_hat))
        check(f"[{mu_mode}] per-example estimate (one pass pair) vs autograd", rel(pe, ref), 1e-6)

    # base-loss part of the two passes == gradient at theta (to O(eps^2))
    cfg = CBGConfig(radius=1e-5, beta=0.0)
    y_hat = sample_y_hat(model, batch, gen)
    g, parts = cbg_step_grad(model, ref_model, batch, cfg, gen, y_hat=y_hat, return_parts=True)
    params = [p for p in model.parameters() if p.requires_grad]
    with torch.no_grad():
        ref_logits = ref_model(batch.tokens[2 * m:])
    l_aln, l_bgn = base_losses(model(batch.tokens), batch, ref_logits)
    g0 = flat(torch.autograd.grad(l_aln + cfg.alpha * l_bgn, params))
    check("base-loss gradient from the two passes vs gradient at theta", rel(flat(g), g0), 1e-8)

    # radius identity and exact restore
    b, w = parts["b"], parts["w"]
    nrm = sum(((w[l] ** 2) / (l.weight.detach() ** 2 + cfg.xi ** 2)).sum() for l in w)
    check("||Lambda^{1/2} w||^2 == b'Kb", abs(nrm - b @ parts["K"] @ b).item() / (b @ parts["K"] @ b).item(), 1e-12)
    before = flat([p.detach() for p in params])
    cbg_step_grad(model, ref_model, batch, CBGConfig(), gen)
    check("weights restored exactly after the perturbed passes", (flat([p.detach() for p in params]) - before).abs().max().item(), 0.0)

    # benign-inflation invariance of eta_k (value) with mu moving with K_bb
    cfg = CBGConfig()
    y_hat = sample_y_hat(model, batch, gen)
    _, K, _ = reference_grad(model, batch, cfg, y_hat)
    Ks = K.clone(); Ks[m:, :] *= 3; Ks[:, m:] *= 3
    e1 = spectral(K, m, n, cfg.k, cfg.kappa_mu).eta
    e2 = spectral(Ks, m, n, cfg.k, cfg.kappa_mu).eta
    check("eta_k invariant to benign inflation (mu recomputed)", abs(e1 - e2) / e1, 1e-12)

    # measurement API
    geo = measure_geometry(model, batch, CBGConfig(), gen)
    check("measure_geometry returns eta_k in (0, 1]", 0.0 if 0 < geo["eta_k"] <= 1 else 1.0, 0.0)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
