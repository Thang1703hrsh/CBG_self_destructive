# CBG v2 — Method Specification for Implementation

This document is the coding reference for **Collateral Barrier Geometry (CBG), version 2**, the method of Section 4 and Appendices A–B of `CBG_v2.tex`. It lists every quantity the training step computes, its shape, the exact formula, and the order of operations.

`cbg_reference.py` implements every step below on a tiny causal LM in float64. It checks itself against an autograd double-backward reference and passes all 13 checks (§14). When the text and your code disagree, compare against that file.

**Conventions**
- `E = m + n` examples per CBG step. Index `e ∈ {0..m−1}` is a safety pair; `e ∈ {m..E−1}` is a benign sequence.
- **Small matrices** (anything `E×E` or smaller) are float64. **Model tensors** follow the training precision.
- `θ` means the **fp32 master weights**.
- `U` means the admissible (measured) weights.
- **[est.]** marks pass-count estimates, not measurements.

---

## 0. One CBG step at a glance

| # | Stage | Input | Output | Cost [est.] |
|---|---|---|---|---|
| 1 | Center forward | batch, θ | logits; sampled `ŷ`; `F̃ ∈ R^E`; stored inputs `X` of every Linear in `U` | 1 forward |
| 2 | Center backward of `1ᵀF̃` | graph | stored output grads `Δ`; Gram `K ∈ R^{E×E}` | 1 backward + per-example Gram |
| 3 | Small solve | `K` | `μ, σ, ν, η_k, G ∈ R^{E×E}` | negligible |
| 4 | Direction | `(X, Δ)`, `G`, sign draw | `a, b ∈ R^E`; `w` (model-shaped, on `U`); `d_Λ` (model-shaped, on `U`) | 1 chunked per-example pass, no backward |
| 5 | Radius | `b, K` | `ε = r·sqrt(dim U / bᵀKb)` | negligible |
| 6 | Two perturbed passes | θ ± εw, same batch and `ŷ` | `g = Σ_σ ∇[σ(β/ε)·aᵀF̃ + ½(L_aln + αL_bgn)]` | 2 forward + backward |
| 7 | Update | `g + β d_Λ` | optimizer step | — |

There is no `retain_graph`, no double backward and no Hessian–vector product call. Everything after stage 2 comes from the stored `(X, Δ)`, the small matrices and ordinary gradients.

---

## 1. Notation and shapes

| Symbol | Meaning | Shape / type |
|---|---|---|
| `m, n` | safety pairs / benign sequences per CBG step | int (default 8, 32) |
| `k` | number of spectral modes | int (default 4, `k ≤ m`) |
| `F̃` | weighted measurements `(h/√m ; F_b/√n)` | `R^E` |
| `c_e` | per-example gradient `∂F̃_e/∂θ` restricted to `U` | model-shaped (on `U`) |
| `C` | stack of `c_e` | `E × dim U` (never materialized) |
| `Λ⁻¹` | `θ² + ξ²` elementwise, on `U` | model-shaped |
| `K` | Gram `C Λ⁻¹ Cᵀ`, blocks `K_hh (m×m)`, `K_hb (m×n)`, `K_bb (n×n)` | `E×E`, fp64 |
| `μ` | damping `κ_μ · tr(K_bb) / n` | scalar |
| `S_μ` | Schur complement `K_hh − K_hb (μI+K_bb)⁻¹ K_hbᵀ` | `m×m` |
| `σ_j, φ_j` | eigenpairs of `S_μ`, descending | `m`, `m×m` |
| `ν_j, φ'_j` | eigenpairs of `K_hh`, descending | `m`, `m×m` |
| `η_k` | `σ_{1:k} / ν_{1:k} ∈ (0, 1]` | scalar |
| `z_j` | `(μI+K_bb)⁻¹ K_hbᵀ φ_j` | `n` |
| `ψ_j, ψ'_j` | `(φ_j ; −z_j)`, `(φ'_j ; 0)` | `E` |
| `E_b` | `0_m ⊕ I_n` | `E×E` |
| `G` | `∂ log η_k / ∂K` | `E×E` |
| `a, b` | eigen-sign pair with `E[abᵀ] = G` | `E` |
| `w` | `Λ⁻¹ Cᵀ b` | model-shaped (on `U`) |
| `d_Λ` | `2 θ ⊙ diag(Cᵀ G C)` | model-shaped (on `U`) |
| `r` | relative perturbation radius | scalar |

---

## 2. Data and measurements

### 2.1 Batch layout

The CBG batch has `2m + n` sequences, in this order:

```
[ x_0+y^h_0, x_0+y^s_0, x_1+y^h_1, x_1+y^s_1, ..., x_{m-1}+y^s_{m-1},  x_0^b+y_0, ..., x_{n-1}^b+y_{n-1} ]
seq_example = [0,0, 1,1, ..., m-1,m-1,  m, m+1, ..., m+n-1]
```

- Safety sequences have at most 256 tokens (prompt + response). Benign sequences have at most 128 tokens.
- Loss and measurement masks cover **response tokens only**.
- Padding is allowed (right padding plus an attention mask). Padded positions receive zero output gradient, so they contribute nothing to `Δᵀ X`.

### 2.2 Safety margin (token-normalized)

For pair `i`, with `mean_logp(y | x)` the mean log-probability over response tokens:

```
h_i(θ) = mean_logp(y^h_i | x_i) − mean_logp(y^s_i | x_i)          # rises as the model moves toward harm
F̃_i   = h_i / sqrt(m)
```

### 2.3 Benign measurement (sampled Fisher)

1. Under teacher forcing on `y_j`, sample `ŷ_{j,t} ~ p_θ(· | x_j, y_{j,<t})` at every response position. Sample under `no_grad` from the center-pass logits.
2. Keep `ŷ` **fixed for the whole step**, including both perturbed passes.
3. Compute

```
F_b,j(θ) = Σ_t log p_θ(ŷ_{j,t} | x_j, y_{j,<t})                 # sum, not mean
F̃_{m+j} = F_b,j / sqrt(n)
```

Do not use the observed `y_j` as the target: that gives the empirical Fisher, which is ablation (iv).

### 2.4 Base losses

- `L_aln`: SFT loss (mean cross-entropy) of `y^s` given `x`, on the safety sequences with `y^s`.
- `L_bgn`: token-level `KL(p_θ0 ‖ p_θ)` on the benign response positions, where `θ0` is the aligned model before defense training.
  - Compute the `θ0` logits once per step under `no_grad`.
  - Reuse them in both perturbed passes.
  - Use chunked log-softmax for large vocabularies.

---

## 3. Parameter sets

| Set | Definition | Rule |
|---|---|---|
| `U` (measured) | the `weight` of every `nn.Linear` inside the transformer blocks: q, k, v, o, gate, up, down | Never reduced. Embeddings, norm gains and `lm_head` are outside `U`. |
| `T` (trained) | the parameters the optimizer updates | May be all parameters, a saliency subset of `U`, or LoRA adapters. `K`, `η_k` and `w` are **always** computed on the full `U`. |

The gradient of `log η_k` reaches **every** trainable parameter, including those outside `U`, through the perturbed-pass gradients. `d_Λ` lives on `U` only.

---

## 4. Stages 1–2: capture and Gram

### 4.1 Capture

- A forward hook on every Linear in `U` stores its input `X` (shape `B×T×d_in`).
- A full backward hook stores its output gradient `Δ` (shape `B×T×d_out`).
- Run the center forward, then `backward()` of `F̃.sum()`. This all-ones seed is what makes the per-example gradients recoverable: examples do not interact in the forward pass, so the rows of `Δ` for example `e` are exactly `∂F̃_e/∂Z`.
- Store `(X, Δ)` in bf16. For Llama-2-7B this is about 5 MB per token by direct count (about 40 GB at 8,192 tokens) [est.]; offload to host memory layer by layer if needed.
- If activation checkpointing is on, hooks fire again during recomputation. Either disable checkpointing for the center pass or keep only the first capture.

### 4.2 Per-example gradient block of one layer

```
c_e^(ℓ) = Σ_{sequences s of example e} Δ_s^T X_s          # (d_out × d_in); a safety example sums its two sequences
```

### 4.3 Gram accumulation (chunk by output rows, accumulate in fp64)

```python
K = zeros(E, E, float64)
for layer in U:
    for rows in chunks(range(d_out), chunk_rows):
        Ce = stack([Δ_e[:, rows].T @ X_e for e in range(E)])        # (E, |rows|, d_in)
        lam_inv = θ[rows]**2 + ξ**2                                   # fp32
        A = Ce.flatten(1).float()
        K += ((A * lam_inv.flatten()) @ A.T).double()
K = (K + K.T) / 2
```

Peak memory is about `E · |rows| · d_in · bytes`. For example, `E = 40`, 2,048 rows, `d_in = 8,192` and fp32 give about 2.7 GB [est.].

Under tensor parallelism by output rows, each rank accumulates a partial `K`; all-reduce the `E×E` matrix.

---

## 5. Stage 3: the small solve (all fp64)

```python
def spectral(K, m, n, k, kappa_mu, mu_mode="differentiated"):
    Khh, Khb, Kbb = K[:m,:m], K[:m,m:], K[m:,m:]
    mu  = kappa_mu * trace(Kbb) / n                          # = kappa_mu × mean eigenvalue of K_bb
    L   = cholesky(mu*I_n + Kbb)
    Rb  = cholesky_solve(Khb.T, L)                           # (mu I + K_bb)^{-1} K_bh      (n × m)
    S   = sym(Khh - Khb @ Rb)
    sig, Phi   = eigh_desc(S)
    nu,  Phi2  = eigh_desc(Khh)
    Z    = Rb @ Phi[:, :k]                                   # z_j
    Psi  = vstack([Phi[:, :k], -Z])                          # psi_j   (E × k)
    Psi2 = vstack([Phi2[:, :k], zeros(n, k)])                # psi'_j  (E × k)
    sS, sN = sig[:k].sum(), nu[:k].sum()
    G = Psi @ Psi.T / sS - Psi2 @ Psi2.T / sN
    c_mu = (Z**2).sum() / sS                                 # = d log eta_k / d mu  (>= 0)
    if mu_mode == "differentiated":
        G[m:, m:] += (kappa_mu / n) * c_mu * I_n             # the E_b term
    return mu, sig, nu, sS / sN, sym(G), c_mu
```

The closed form behind this code:

```
G = (1/σ_{1:k}) Σ_{j≤k} ( ψ_j ψ_jᵀ + (κ_μ/n) ‖z_j‖² E_b )  −  (1/ν_{1:k}) Σ_{j≤k} ψ'_j ψ'_jᵀ
```

**Rules**

- `mu_mode = "differentiated"` is the default. `"detached"` exists for ablation (ix) only: with `μ` detached, inflating benign sensitivity lowers `η_k`.
- Log the spectral gaps `σ_k − σ_{k+1}` and `ν_k − ν_{k+1}`. Without a gap the gradient is a Clarke subgradient; this is not an error, but record it.
- Skip the CBG term for the step (and log it) if `σ_{1:k}` or `ν_{1:k}` is `≤ 1e-30` or non-finite.

**Escapes (optional, for diagnostics and attack S5.4)**

```
u_j = Λ⁻¹ Cᵀ ψ_j / sqrt(μ σ_j)
```

- efficiency `ρ_j = σ_j / μ`
- `C_h u_j = sqrt(σ_j/μ) φ_j`
- `C_b u_j = sqrt(μ/σ_j) z_j`
- `Q_b(u_j) = μ ‖z_j‖² / σ_j`

---

## 6. Stage 4: sign pair, direction `w` and `d_Λ`

### 6.1 Sign pair

```python
gam, V = eigh(G)
s  = rademacher(E)                    # new draw each step; fixed within the step
sq = sqrt(abs(gam))
a  = V @ (sq * s)
b  = V @ (sign(gam) * sq * s)         # E[a bᵀ] = G
```

### 6.2 Direction and Λ term (one chunked pass over `(X, Δ)`, no backward)

```python
for layer in U:
    for rows in chunks(...):
        Ce = stack([Δ_e[:, rows].T @ X_e for e in range(E)])      # (E, |rows|, d_in)
        lam_inv = θ[rows]**2 + ξ**2
        w[rows]    = lam_inv * einsum("e,eoi->oi", b, Ce)
        M          = einsum("ef,foi->eoi", G, Ce)                 # Σ_e' G_ee' c_e'
        d_lam[rows] = 2 * θ[rows] * (Ce * M).sum(0)
```

`w` alone needs only one weighted product per layer, `(θ²+ξ²) ⊙ (Δᵀ diag(b_tok) X)`, where `b_tok` repeats `b_e` over the tokens of example `e`. `d_Λ` is what needs the per-example form.

---

## 7. Stage 5: radius

```
‖Λ^{1/2} w‖² = bᵀ K b
ε = r · sqrt( dim U / (bᵀ K b) )     # RMS relative perturbation over U equals r
```

- Choose `r` with S0.2 (finite difference vs. exact HVP) at the precision you will train in.
- Guidance from the centered-difference error trade-off: try `{1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2}`.
  - bf16 autocast usually needs `r ≈ 1e-2`.
  - fp32 compute tolerates `r ≈ 1e-3`.

---

## 8. Stage 6–7: perturbed passes and gradient

```python
backup = {W: W.detach().clone() for W in U}               # or restore via W -= σεw if memory is tight (≤ few ulp error)
g = zeros_like(trainable)
for σ in (+1, -1):
    with no_grad():
        for W in U: W.copy_(backup[W] + σ * ε * w[W])     # fp32 master weights
    logits = model(batch)                                  # same batch, same ŷ, same dropout (off)
    Ft     = measurements(logits, ŷ)
    L_aln, L_bgn = base_losses(logits, ref_logits_benign)
    g += grad( σ * (β/ε) * (a @ Ft) + 0.5 * (L_aln + α * L_bgn) , trainable)
with no_grad():
    for W in U: W.copy_(backup[W])
g[U] += β * d_lam
optimizer.step(g)
```

Why this is correct:

- `Σ_σ σ(β/ε)∇Ω(θ+σεw) = β[∇Ω(θ+εw) − ∇Ω(θ−εw)]/ε`, and its expectation over the sign draw equals the curvature term of `β∇log η_k` up to `O(ε²)`.
- `½[∇L(θ+εw) + ∇L(θ−εw)] = ∇L(θ) + O(ε²)`.
- Total: `∇(L_aln + αL_bgn) + β∇log η_k`.

**Variance.** One sign draw has large relative error: 200–260% on the synthetic checks, although the mean is exact. Averaging `D` draws costs `2D` passes. The per-example variant (§10) has no sign variance.

---

## 9. Options on top of the step

| Option | What changes | Notes |
|---|---|---|
| Lazy CBG (`K_lazy`) | Run the full CBG step every `K_lazy` steps with `β_eff = K_lazy·β`; other steps optimize `L_aln + αL_bgn` only | With AdamW, scaling by `K_lazy` is not an exact compensation; test "no rescale" in S10. |
| CBG+UL | Add `γ_ul · L_ul` (SEAM's unlearning loss) on the harmful responses `y^h` of the same pairs | Reuse SEAM's released implementation of `L_ul` (layer-wise gradient ascent with log transform); `γ_ul` is TBD. |
| Trained subset `T ⊂ U` | Mask optimizer updates outside `T` | `K`, `w`, `ε` stay on the full `U`. |
| LoRA training | Base weights frozen; apply the perturbation as an additive branch `x(W0 + BA)ᵀ + σε·x wᵀ` | Chain `d_Λ` (a `W`-shaped gradient) to the adapters: `g_B += d_Λ Aᵀ·scale`, `g_A += Bᵀ d_Λ·scale`. `Λ` uses `W = W0 + BA`. |
| Several sign draws | Repeat stages 4–6 with new `s`; average `g` | Stages 1–3 are shared. |

---

## 10. Per-example variant (ablation x; deterministic)

For each example `e`:

```
w_e = Λ⁻¹ Cᵀ G 1_e
ε_e = r · sqrt(dim U / (G K G)_ee)
```

Each example's tokens are evaluated at `θ ± ε_e w_e`. Use one additive branch per linear layer, `x_e θᵀ ± ε_e x_e w_eᵀ`, and rebuild `w_e` chunk by chunk from `(X, Δ)` in both the forward and the backward of each perturbed pass. The pass-`σ` scalarization is `Σ_e σ β F̃_e / ε_e`, and `d_Λ` is unchanged.

Cost is about 14 G_p per step against about 8 G_p for the shared direction [est.]. `cbg_reference.py::per_example_curvature` implements it with `functional_call`.

---

## 11. Geometry measurement and logging

### 11.1 `measure_geometry(model, batch)`

Runs stages 1–3 only, with no update. Use it on the fixed `D_geo` batches (same examples and the same `ŷ` seed for every model).

It returns:
- `eta_k`, `log_eta_k`
- `rho_mu = σ_1/μ`
- `sigma_1k`, `nu_1k`
- `tr_Kbb`, `mu`, `c_mu`
- `eff_rank_Kbb = tr(K_bb)² / tr(K_bb²)`
- `gap_sigma`, `gap_nu`

Always report `μ, m, n` with any geometry number. **Never** report the undamped minibatch `ρ_esc` or `B_γ`; they are `+∞` and `0` by construction.

### 11.2 Log every CBG step

- the geometry fields of §11.1
- `ε`, `bᵀKb`
- `‖β d_Λ‖` and `‖β · curvature part‖` against `‖∇(L_aln + αL_bgn)‖`, i.e. whether the CBG term actually acts
- `L_aln`, `L_bgn`
- the fraction of `U` coordinates with `|θ_i| < ξ`
- skipped-step and NaN flags
- wall time per stage and peak memory (S12)

---

## 12. Numerical rules

1. Keep `K`, `S_μ`, `G`, `a`, `b` and all eigensolves in float64. Symmetrize `K`, `S_μ` and `G` after forming them.
2. Solve with a Cholesky factorization of `μI + K_bb` rather than an explicit inverse.
3. Apply perturbations to fp32 master weights. The compute precision of the perturbed passes must resolve relative changes of size `r` (bf16 autocast rounds weights and activations to about `2⁻⁸` relative), so choose `r` by S0.2.
4. Keep `ŷ`, the minibatch, the sign draw and dropout (off) identical across the center pass and both perturbed passes.
5. Restore the weights after the perturbed passes. The reference restores exactly from a copy; restoring by subtraction leaves an error of a few ulp.
6. Use one global `ξ` (default `1e-3`) and log the fraction of `|θ_i| < ξ`.

---

## 13. Configuration

```yaml
# ---- method (tuned on the validation protocol; see the training guide §2.3) ----
alpha: 1.0              # fixed (SEAM found alpha insensitive; re-check on pilot)
beta: 0.01              # TUNED: grid {1e-3, 1e-2, 1e-1, 1}; beta is not comparable to SEAM's beta
radius: 1.0e-2          # CALIBRATED by S0.2 at the training precision (not tuned on attacks)

# ---- fixed structure ----
m: 8
n: 32                   # n >= 4m
k: 4                    # k <= m
len_safety: 256         # tokens per sequence (prompt + response)
len_benign: 128
kappa_mu: 1.0           # fixed on the pilot (candidates 0.05, 0.2, 1.0); sensitivity in S10
xi: 1.0e-3
mu_mode: differentiated # ablation: detached
benign_metric: sampled_fisher   # ablation: empirical_fisher
estimator: shared_sign  # ablation: per_example
num_sign_draws: 1

# ---- execution ----
lazy_period: 1          # K_lazy; chosen by S0.6
trained_set: all        # all | saliency:<T_cov> | lora:<rank>
perturb_precision: bf16 # bf16 | tf32 | fp32, chosen by S0.2

# ---- optimizer (SEAM-matched) ----
optimizer: adamw
lr: 2.0e-5
steps: 500
scheduler: cosine
warmup_ratio: 0.1
weight_decay: 0.0
```

---

## 14. Tests

`python cbg_reference.py` runs on a 2-block Llama-pattern model in float64 and passes 13/13:

| Check | Result |
|---|---|
| Gram from `(X, Δ)` hooks vs per-example autograd | 3e-16 |
| `log η_k` vs autograd | 2e-16 |
| Mean over all `2^E` sign draws of (curvature estimate + `d_Λ`) vs exact `∇θ log η_k` from double backward, μ differentiated / detached | 5e-9 / 3e-9 |
| Per-example estimator vs exact gradient | 1.3e-9 / 1.5e-9 |
| Base-loss gradient from the two passes vs gradient at θ | 9e-10 |
| `‖Λ^{1/2}w‖² = bᵀKb` | 1e-16 |
| Weights restored exactly | 0 |
| `η_k` invariant to benign inflation | 0 |
| `measure_geometry` sanity | passes |

The comparison covers gradients on all parameters, including embeddings, norms and the head.

`verify_cbg_v2.py` checks the same algebra on an analytic model:
- escape formulas
- `c_μ`
- equivalence with the original paper's `A` matrix
- second-order finite differences
- invariances

**Tests to add in the LLM codebase**, all on a ≤1B model:
- **L1.** Hook Gram equals the Gram from a per-example autograd loop on 2+2 short sequences, in fp32.
- **L2.** The chunked `w` and `d_Λ` equal the dense formulas.
- **L3.** The finite-difference curvature term along a fixed `(a, b)` matches an exact HVP (`torch.func.jvp` over `grad`) as a function of `r` and precision. This is S0.2.
- **L4.** One full step with `β = 0` reproduces the plain `L_aln + αL_bgn` gradient.
- **L5.** Padding: adding padded tokens leaves `K` unchanged.
- **L6.** Activation checkpointing on or off gives the same `K`.

---

## 15. Cost and memory [est.; replace with S0.4 / S12 measurements]

`G_p` is one forward and backward pass on 4,096 tokens. The CBG batch (8 pairs × 2 × 256 + 32 × 128 = 8,192 tokens) costs about 2 G_p per pass.

| Stage | Shared direction | Per-example |
|---|---|---|
| Center pass | 2.0 | 2.0 |
| Gram (per-example forming) | ~0.8 | ~0.8 |
| `w`, `d_Λ` pass | ~0.8 | ~0.8 |
| Two perturbed passes | 4.0 | ~9.8 (branch products + rebuilding `w_e` in forward and backward) |
| `θ0` reference forward | ~0.3 | ~0.3 |
| **Total per CBG step** | **≈ 8 G_p** | **≈ 14 G_p** |
| With `lazy_period = 4` (plain steps ≈ 1.5–1.8 G_p) | ≈ 3–3.5 G_p per step | ≈ 4.5–5 G_p per step |

SEAM's Algorithm 1 evaluates 5–6 gradients per step on 8-example batches. The SEAM paper does not report tokens per batch, so the ratio to SEAM must come from measurement (S12).

---

## 16. Pitfalls checklist

- [ ] `F̃` uses weights `1/√m` and `1/√n`; the benign term is a **sum** over tokens and the safety term a **mean** difference.
- [ ] `ŷ` is sampled once per step and reused; it is not resampled in the perturbed passes.
- [ ] Backward seed is `F̃.sum()` (all ones). Any other seed breaks the per-example recovery.
- [ ] The two sequences of a safety pair are summed into one example.
- [ ] `U` excludes embeddings, norms and `lm_head`; `T` may differ from `U`, but `K` and `w` never shrink.
- [ ] `G` includes the `E_b` term, i.e. `μ` is differentiated.
- [ ] `d_Λ` is added with weight `β`, on `U` only.
- [ ] `ε` uses `bᵀKb`, not the norm of a detached `w` in bf16.
- [ ] Weights are restored after the perturbed passes; optimizer state is untouched by them.
- [ ] Geometry is always measured on the full `U` and reported with `μ, m, n`.
