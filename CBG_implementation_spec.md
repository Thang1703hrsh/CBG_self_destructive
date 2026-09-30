# CBG — Implementation Specification

Implementation spec for **Collateral Barrier Geometry (CBG)**, derived from the paper manuscript (`CBG_acl.tex`, Section 4, Appendix A–B).
Hand this file to Claude Code. The math below is authoritative; if code and this file disagree, the file wins unless a test in §9 shows the file is wrong.

---

## 0. Ground rules for the implementer

- **Do not change the objective.** The loss is `L_aln + α·L_bgn + β·log η_k` (§4). Simplifications are allowed only behind an explicit config flag and must be logged (see §8).
- **Verify before scaling.** Implement on a tiny model in float64 first and pass every test in §9 before touching a 7B model.
- **No invented hyperparameters.** Values marked `TBD` must come from a config file, not be hard-coded.
- Framework: PyTorch + Hugging Face `transformers`. Target models: Llama-2-7B-Chat (default), Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct.

---

## 1. Notation and shapes

| Symbol | Meaning | Shape / type |
|---|---|---|
| `θ` | all trainable parameters (fp32 master copy) | — |
| `U` | admissible update space: **all weight matrices of `nn.Linear` layers** in the transformer blocks | subset of `θ` |
| `m` | safety pairs per step (default 8) | int |
| `n` | benign sequences per step (default 32) | int |
| `E = m + n` | examples per step | int |
| `k` | number of spectral modes (default 4), `k ≤ m` | int |
| `ξ` | damping floor | float, `TBD` |
| `κ_μ` | damping scale | float, `TBD` |
| `r` | RMS relative perturbation radius | float, `TBD` |
| `α, β` | loss weights | float, `TBD` |
| `G_e` | per-example gradient of `F̃_e` w.r.t. parameters in `U` | one tensor per Linear weight |
| `Λ⁻¹` | elementwise `θ² + ξ²` on `U` | same shapes as weights |

### Data per step

- Safety batch: `m` triples `(x_i, y_i^s, y_i^h)` → **2m sequences** (prompt+safe, prompt+harmful). Default response length ≤ 512 tokens.
- Benign batch: `n` pairs `(x_j, y_j)` → **n sequences**. Default length ≤ 128 tokens.
- **Never pack several examples into one sequence.** Per-example gradients rely on examples not interacting inside the forward pass (§3.2).

---

## 2. Quantities computed in one forward pass

### 2.1 Safety margins `F_h ∈ R^m`

For example `i`, with teacher forcing and response-token masks:

```
h_i(θ) = (1/|y_i^h|) Σ_{t∈y_i^h} log p_θ(y^h_t | x_i, y^h_<t)
       − (1/|y_i^s|) Σ_{t∈y_i^s} log p_θ(y^s_t | x_i, y^s_<t)
```

Weighted: `F̃_h = F_h / sqrt(m)` (default `W_h = I/m`).

### 2.2 Benign sampled-Fisher measurements `F_b ∈ R^n`

Using the logits of the **same** forward pass at `θ`, under teacher forcing on the reference `y_j`:

```
ŷ_{j,t} ~ Categorical(softmax(logits_{j,t}))        # temperature 1, detached, sampled ONCE per step
F_{b,j}(ϑ) = Σ_t log p_ϑ(ŷ_{j,t} | x_j, y_{j,<t})   # SUM over response tokens, not mean
```

Weighted: `F̃_b = F_b / sqrt(n)` (default `W_b = I/n`).
The samples `ŷ` are **reused unchanged** in the perturbed passes (§5).

### 2.3 Behavior-preserving losses

- `L_aln` = mean token NLL of `y_i^s` given `x_i` (SFT on safe responses). It reuses the safe sequences of the safety batch.
- `L_bgn` = token-level KL divergence between the original aligned model `θ₀` and the current model on benign sequences. The paper says "KL divergence from θ₀"; implement `KL(p_θ₀ ‖ p_θ)` averaged over response tokens. **Confirm the direction with the author** and keep it configurable.
- Reference logits of `θ₀`: precompute and cache for the benign set, or run a frozen `θ₀` forward (~0.3 G per step).

---

## 3. Per-example Gram accumulation (center backward pass)

### 3.1 What to compute

Stack the weighted per-example Jacobian rows `C = [C_h; C_b]`, where row `e` is `G_e = ∇_θ F̃_e` restricted to `U`. We need the `E×E` Gram in the `Λ⁻¹` inner product:

```
K[e, e'] = Σ_{Linear layers L} Σ_{params p ∈ L} (θ_p² + ξ²) · G_e[p] · G_{e'}[p]
```

Split into blocks `K_hh (m×m)`, `K_hb (m×n)`, `K_bb (n×n)`.

### 3.2 How: hooks on every `nn.Linear` in `U`

For a Linear layer `Z = X Wᵀ`, let `X_s` (tokens×d_in) be the input rows of sequence `s` and `Δ_s` (tokens×d_out) the gradient of the backward seed w.r.t. `Z` on those rows. With seed `Σ_e F̃_e` (all-ones cotangent on `F̃`), rows belonging to example `e` carry exactly `∂F̃_e/∂Z`, because examples do not interact. Then:

```
G_e(W) = Σ_{sequences s of example e} Δ_sᵀ X_s        # (d_out × d_in)
```

- A **safety example owns two sequences** (safe and harmful). Sum both. The signs and `1/|y|` factors are already in `Δ`, via the backward through `h_i`.
- A benign example owns one sequence.
- Padding rows have `Δ = 0`, so they contribute nothing. Still mask them explicitly for safety.

Implementation pattern:

```python
# forward hook: save X (detached) per Linear in U
# tensor hook on output Z: receive Δ = dL/dZ during backward
def on_backward(layer, X, Δ):           # X: [B, T, d_in], Δ: [B, T, d_out]
    lam_inv = layer.weight.detach().float()**2 + xi**2          # [d_out, d_in]
    for chunk in chunks_over_output_rows(layer, max_bytes):     # bound memory
        Ge = per_example_grads(X, Δ, seq_to_example, rows=chunk) # [E, |chunk|, d_in] fp32
        Ge_w = Ge * lam_inv[chunk].sqrt()                        # Λ⁻¹ inner product
        K += Ge_w.flatten(1) @ Ge_w.flatten(1).T                 # accumulate in fp64
```

- Accumulate `K` in **float64**.
- Memory: holding all `G_e` for one 14336×4096 matrix at `E = 40` takes about 4.7 GB in bf16 (9.4 GB in fp32). Chunk over output rows.
- The ordinary `.grad` from this backward is **not** used. Call `torch.autograd.grad(..., retain_graph=True)` on the seed and discard the result, or zero the grads afterwards.
- **Retain the graph.** It is reused in §4.3.

---

## 4. Spectral solve (small matrices, float64)

### 4.1 Damping scale

```
μ = κ_μ · trace(K_bb) / n          # detached scalar
```

### 4.2 Reduced matrices

```
S_bb = μ I_n + K_bb                                   # n×n, SPD → Cholesky
M    = (1/μ) · (K_hh − K_hb S_bb⁻¹ K_hbᵀ)             # m×m, symmetrize: M = (M + Mᵀ)/2
```

- `ρ_1 ≥ … ≥ ρ_k` = top-k eigenvalues of `M` (must be > 0).
- `y_j` = the eigenvectors, rescaled so that `‖y_j‖² = 1/ρ_j`.
- `ν_1 ≥ … ≥ ν_k` = top-k eigenvalues of `K_hh`.
- `y'_j` = the eigenvectors, rescaled so that `‖y'_j‖² = 1/ν_j`.

### 4.3 Coefficients of the escape directions (never materialized individually)

For each mode `j ≤ k`:

```
a_j = y_j / μ                                          # length m
b_j = −(1/μ) S_bb⁻¹ K_hbᵀ y_j                          # length n
c_j  = [a_j ; b_j]        → u_j = Λ⁻¹ ⊙ (Cᵀ c_j)       # damped escape, u_jᵀ B_μ u_j = 1
c'_j = [y'_j ; 0_n]       → v_j = Λ⁻¹ ⊙ (Cᵀ c'_j)      # reference escape, v_jᵀ Λ v_j = 1
```

Displacements come from the Gram alone, with no extra passes:

```
C_h u_j = K_hh a_j + K_hb b_j          C_b u_j = K_hbᵀ a_j + K_bb b_j
C_h v_j = K_hh y'_j  (= ν_j y'_j)
```

### 4.4 Objective value

```
S_k = Σ_{j≤k} ρ_j ,   N_k = Σ_{j≤k} ν_j
log η_k = log(μ · S_k) − log(N_k)          # η_k ∈ (0, 1]; assert η_k ≤ 1 + 1e-6
```

Log per step: `η_k, S_k, N_k, μ, trace(K_bb), trace(K_hh)`, the full spectra of `M` and `K_hh`, and the gaps `ρ_k − ρ_{k+1}` and `ν_k − ν_{k+1}`. Warn when a relative gap is < 1e-3; the gradient is then only a Clarke subgradient.

---

## 5. Outer gradient

### 5.1 Random direction and scalarization

Draw Rademacher `s, t ∈ {±1}^k`.

**Coefficient of `w` (length E):**
```
c_w = Σ_j s_j c_j / sqrt(S_k) + Σ_j t_j c'_j / sqrt(N_k)
w   = Λ⁻¹ ⊙ (Cᵀ c_w)
```

**Cotangents of `Ω` (so that `Ω(ϑ) = g_hᵀ F̃_h(ϑ) + g_bᵀ F̃_b(ϑ)`):**
```
g_h =  Σ_j s_j (C_h u_j) / sqrt(S_k)  −  Σ_j t_j (C_h v_j) / sqrt(N_k)     # length m
g_b = −Σ_j s_j ρ_j (C_b u_j) / sqrt(S_k)                                  # length n
```

This implements `Ω = Σ_j s_j Φ_j/√S_k − Σ_j t_j Ψ_j/√N_k` with `Φ_j = (C_h u_j)ᵀF̃_h − ρ_j (C_b u_j)ᵀF̃_b` and `Ψ_j = (C_h v_j)ᵀF̃_h`.

### 5.2 Materialize `w` and the damping-gradient term (second hooked backward on the retained graph)

Run a backward on the retained graph with the **all-ones seed again**, so the hooks see the per-example `Δ`. In each Linear hook, from the per-example `G_e` of that layer (chunked as in §3):

```
w_L      = Λ⁻¹_L ⊙ Σ_e c_w[e] · G_e
diagA_L  = Σ_e G_e ⊙ (Σ_{e'} A[e,e'] · G_{e'})           # elementwise
d_Λ,L    = −2 · θ_L ⊙ diagA_L
```

with the `E×E` matrix

```
A = −(μ / S_k) Σ_j ρ_j c_j c_jᵀ  +  (1 / N_k) Σ_j ν_j c'_j c'_jᵀ
```

Derivation (for the record): `∇_θ(uᵀΛu) = −2 u⊙u⊙θ / (θ²+ξ²)²` and `u = Λ⁻¹⊙(Cᵀc)`, hence `∇_θ(uᵀΛu) = −2 θ ⊙ (Cᵀc)⊙(Cᵀc)`. So

```
d_Λ = −(μ/S_k) Σ_j ρ_j ∇(u_jᵀΛu_j) + (1/N_k) Σ_j ν_j ∇(v_jᵀΛv_j) = −2 θ ⊙ diag(Cᵀ A C)
```

`d_Λ` is nonzero only on `U`. Free the graph after this pass.

### 5.3 Step size of the perturbation

```
ε = r · sqrt(dim U) / ‖ w ⊙ (θ² + ξ²)^{-1/2} ‖₂        # RMS relative perturbation = r
```

### 5.4 Two perturbed passes (give CBG and base-loss gradients together)

```python
g = 0
for σ in (+1, -1):
    θ_σ = θ + σ ε w                         # on fp32 master weights; w is zero outside U
    forward joint batch at θ_σ with the SAME ŷ samples, same batch, dropout off
    loss_σ = σ * (β/ε) * Ω(θ_σ) + 0.5 * (L_aln(θ_σ) + α L_bgn(θ_σ))
    g += grad(loss_σ, all trainable params)
g += β * d_Λ
optimizer.step(g)
```

- `Ω(θ_σ) = g_hᵀ F̃_h(θ_σ) + g_bᵀ F̃_b(θ_σ)`, with `g_h, g_b` fixed from the center point.
- Summing over `σ` gives `β[∇Ω(θ+εw) − ∇Ω(θ−εw)]/ε + ½[∇L(θ+εw) + ∇L(θ−εw)]`. This estimates `β ∇log η_k + ∇(L_aln + α L_bgn)`, unbiased up to `O(ε²)` over the sign draw.
- These gradients apply to **all** trainable parameters, including those outside `U`. That is correct, because `η_k` depends on them too.

### 5.5 ⚠ Precision (important — stricter than the paper text)

The perturbation is about `r` relative per coordinate. **Two separate precision limits** can erase it:

1. **Weight storage.** Keep master weights in fp32 and apply `θ ± εw` in fp32.
2. **Matmul compute.** Under bf16 autocast, weights are cast to bf16 inside each matmul (2⁻⁸ ≈ 4e-3 relative resolution). TF32 rounds to about 2⁻¹¹ ≈ 5e-4.

So either run the two perturbed passes in true fp32 (`torch.backends.cuda.matmul.allow_tf32 = False`, no autocast), or choose `r` well above the compute resolution. In the synthetic check, bf16-rounded weights gave 190% / 35% / 6% error at relative radii 1e-4 / 1e-3 / 1e-2.

Tune `r` with test T6 (§9) on the target precision setting.

> Note for the paper: Section 4.3 currently says only "perturbations are applied to fp32 master weights". It should also say the perturbed forwards must not down-cast weights, or that `r` is chosen above the compute precision.

---

## 6. One full training step (summary)

```
Algorithm CBG-step(θ, safety batch, benign batch)
 1. Forward joint batch at θ (fp32 master weights; center pass may use bf16 autocast).
    Sample ŷ from benign logits. Build F̃_h, F̃_b.
 2. Hooked backward of Σ F̃ (retain graph): accumulate K = [[K_hh, K_hb],[K_hbᵀ, K_bb]] (fp64).
 3. μ ← κ_μ tr(K_bb)/n. Solve M and K_hh for top-k pairs. Compute S_k, N_k, log η_k,
    coefficients c_j, c'_j, displacements C_h u_j, C_b u_j, C_h v_j.
 4. Draw s, t. Form c_w, g_h, g_b, A.
 5. Second hooked backward on retained graph: w (on U), d_Λ (on U). Free graph.
 6. ε ← r·sqrt(dim U)/‖w/sqrt(θ²+ξ²)‖.
 7. For σ ∈ {+1,−1}: forward at θ+σεw (same ŷ, fp32 compute), backward of
    σ(β/ε)Ω + ½(L_aln + αL_bgn), accumulate into g.
 8. g ← g + β d_Λ. Optimizer step (AdamW, settings TBD).
```

**Cost** (pass-count estimate, not a measurement; `G` = one forward+backward on 4,096 tokens):

| Stage | Estimate |
|---|---|
| center forward+backward | ~3 G |
| Gram hooks | ~1 G extra |
| eigensolve | negligible |
| second hooked backward (w, d_Λ) | ~2 G + hook work |
| two perturbed passes | ~6 G (more if fp32 compute) |
| `θ₀` reference forward | ~0.3 G |

The total is about 12 G, versus about 5 G for SEAM. Measure the real numbers and report them per stage.

---

## 7. Suggested code layout

```
cbg/
  config.py          # dataclass: m, n, k, ξ, κ_μ, r, α, β, lr, steps, seq lens, flags (§8)
  data.py            # safety triples → 2 sequences + example ids; benign; masks; no packing
  measurements.py    # F̃_h, F̃_b (sampled Fisher), L_aln, L_bgn
  hooks.py           # forward/backward hooks on nn.Linear in U; per-example grads; chunking
  gram.py            # accumulate K; second pass computing w and d_Λ
  spectral.py        # μ, M, eigen-solves, coefficients, log η_k, A, c_w, g_h, g_b
  step.py            # CBG-step (§6)
  train.py           # loop, logging, checkpoints
  tests/             # §9
```

---

## 8. Config flags (debug / ablation only; log when used)

| Flag | Effect | Paper ablation |
|---|---|---|
| `objective = {log_eta, raw_rho}` | `log η_k` vs raw `ρ̂_μ = ρ_1` | (i) |
| `damping = {relative, euclidean}` | `Λ = diag(1/(θ²+ξ²))` vs `Λ = I` | (ii) |
| `k` | number of modes | (iii) |
| `fisher = {sampled, empirical}` | `ŷ` sampled vs observed `y_j` | (iv) |
| `n` at fixed token budget | minibatch bias | (v) |
| `solver = {exact, rayleigh_K}` | exact reduced solve vs K-step Rayleigh ascent | (vi) |
| `perturb_precision = {fp32, bf16}`, `r` | FD precision | (vii) |
| `drop_aln`, `drop_bgn`, `add_unlearn` | loss-term ablations | (viii) |
| `detach_lambda` | set `d_Λ = 0`; **deviates from the paper**, debug only | — |
| `num_sign_draws` | average several `(s, t)` draws per step (variance reduction; costs 2 G per extra draw) | — |

---

## 9. Required tests (pass all on a tiny model in float64 before scaling)

Use a tiny Llama-architecture model (e.g., 2 layers, hidden 32, SwiGLU, RMSNorm), with `torch.set_default_dtype(torch.float64)`. `cbg_checks.py` (already provided) contains NumPy reference versions of T2–T5 and T7.

| ID | Test | Pass criterion |
|---|---|---|
| T1 | Per-example grads from hooks (incl. 2-sequence safety examples) vs `torch.func.vmap(grad)` | max rel err < 1e-10 |
| T2 | Reduced solve (§4) vs dense `scipy.linalg.eigh(Gh, Gb + μΛ)` on the full parameter vector | top-k eigenvalues rel err < 1e-8; `u_jᵀ B_μ u_j = 1` |
| T3 | Randomized estimator: average of `2 H_Ω w + d_Λ` over **all** `2^{2k}` sign draws (exact HVP via `torch.func`) vs central-difference gradient of `log η_k(θ)` recomputed from scratch | rel err < 1e-6 |
| T4 | `d_Λ` from §5.2 vs autograd of `Σ ρ_j u_jᵀΛu_j` with `u_j` frozen | rel err < 1e-10 |
| T5 | Invariance: scale one SwiGLU up-projection row by `c` and the matching down-projection column by `1/c` (both in `U`) | model outputs identical; `η_k` unchanged (rel < 1e-8) with `damping=relative, ξ=0`; changes with `damping=euclidean` |
| T6 | Finite difference `(∇Ω(θ+εw) − ∇Ω(θ−εw))/ε` vs exact `2 H_Ω w`, for the chosen `r` and precision | report rel err vs `r`; pick `r` in the flat region |
| T7 | `0 < η_k ≤ 1`; `η_k` unchanged when `F̃_h` is scaled by a constant | exact up to fp tolerance |
| T8 | End-to-end: 20 steps on the tiny model; `log η_k` decreases; `L_aln`, `L_bgn` stay bounded | qualitative |

---

## 10. Logging and diagnostics (needed for Section 5 of the paper)

Log per step:
- `log η_k`, `S_k`, `N_k`, `μ`, `tr(K_bb)`, `tr(K_hh)`, and the top-10 spectra of `M` and `K_hh`
- eigen-gaps, `ε`, `‖w‖`, `‖d_Λ‖`, `L_aln`, `L_bgn`
- the time of each stage in §6, and peak memory

`N_k` and `tr(K_bb)` are the flattening / inflation diagnostics (paper hypothesis H4).

Provide a standalone `measure_geometry(model, batch)` that returns `η_k, ρ̂_μ, N_k, tr(K_bb)` without a training step. It is used at attack checkpoints for the geometry-predicts-outcome analysis.

---

## 11. Multi-GPU (after single-GPU correctness)

- Keep the **same joint batch on all ranks**, so the Gram needs every example's gradient.
- Shard parameters (FSDP or tensor parallel). Each rank accumulates its partial `K` from its parameter shard; `all_reduce(K, SUM)`.
- The spectral solve (tiny) is replicated. `w` and `d_Λ` are computed per shard.
- Do not split examples across ranks for the Gram pass.
