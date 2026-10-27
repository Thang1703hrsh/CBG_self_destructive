# Method-to-code map

This document describes the normalized method in the accompanying paper. The
older `cbg_version: v2` and `v5` Trainer branches optimize raw single-mode
Rayleigh objectives and are retained as experimental code, not as synonyms for
the paper method.

## Measurements and parameter spaces

`spec.measurements` implements Eqs. (2) and (3). Safety measurements are harmful
minus safe teacher-forced **mean response log probabilities**, weighted by
`1/sqrt(m)`. Benign measurements are **sums** of sampled token log probabilities,
weighted by `1/sqrt(n)`. Targets are drawn independently at teacher-forced benign
contexts, once per step, and frozen through the curvature calculation. The
sampled tokens do not replace the teacher-forced context tokens.

The base losses are token means: safe-response cross-entropy and
`KL(p_reference || p_current)` on benign response positions. The causal shift
uses logits at position `t-1` to predict token `t`; padding is excluded. Prompt
activations still contribute to the parameter Jacobian through attention.

`spec_linear_modules` chooses U: Linear **weight matrices** inside the configured
transformer block container (`model.layers` by default). Embeddings, biases,
normalization weights, and the output head are outside U. Trainable parameters
T may include those coordinates: the outer mixed-Hessian derivative covers T.
Frozen coordinates in U still participate in the search. They receive no new
optimizer gradients. Quantized models and LoRA wrappers are not supported by
the paper hook implementation.

## Gram and reduced solve

`SpecLinearHooks` stores layer inputs X and output cotangents Delta from the
backward of the sum of weighted measurements. Per-example weight gradients are
`Delta.T @ X`; a safety example combines its harmful and safe sequences before
forming any Gram entries. Microbatch contributions accumulate into the same
global example index.

`K = C @ diag(theta_U**2 + xi**2) @ C.T` is accumulated in FP64 using chunks of
output rows. The complete parameter-by-parameter Hessian and full per-example
parameter Jacobian are never materialized in production. `gram_max_bytes`
bounds contraction workspaces, not all activation, capture, or model storage.

`solve_spec_spectrum` implements Eqs. (7), (8), and (13):

```text
mu = kappa_mu * trace(K_bb) / n
Z = solve(K_bb + mu * I, K_hb.T)
S_mu = K_hh - K_hb @ Z
log_eta = log(sum_top_k_eigenvalues(S_mu)) - log(sum_top_k_eigenvalues(K_hh))
```

The solve uses Cholesky and symmetric FP64 eigensolvers, checks symmetry,
finiteness, and positive semidefiniteness, and computes the Gram cotangent G
analytically. The `mu_mode=differentiated` branch includes
`(kappa_mu/n) * ||Z @ Phi_k||_F**2 / sigma_sum` on the benign diagonal.
`SpecSpectrum.rho` stores the eigenvalues of `S_mu/mu`, so `mu * rho` recovers
the paper's sigma values. `A` is a legacy name for `-G`.

Zero benign trace or zero spectral sums raise `DegenerateGeometry`. Training
then falls back to the base losses and records `geometry_skipped`. Identical
safety pairs are recognized before contraction, avoiding artificial geometry
from cancellation noise. Tiny but nonzero safety spectra remain valid; no
absolute cutoff breaks the safety-scale invariance. Near a closed top-k gap,
the code warns that differentiability is not guaranteed. It does not claim
that an arbitrary combination of the two spectral subgradients is a Clarke
subgradient of their log ratio.

## Curvature and metric derivative

For `G = V diag(gamma) V.T`, draw a Rademacher vector s and form

```text
a = V @ (sqrt(abs(gamma)) * s)
b = V @ (sign(gamma) * sqrt(abs(gamma)) * s)
w = diag(theta_U**2 + xi**2) @ C.T @ b
g_geometry = 2 * H_(a.T @ F_tilde) @ w + d_metric
d_metric = 2 * theta_U * diag(C.T @ G @ C)
```

`eigen_sign_scalarization` returns `(b, a)`, in that order. Conditioned on the
center Gram, `E[a @ b.T] = G`. Signs, sampled targets, w, and G are detached.
`SpecLinearHooks.materialize` forms w and the metric derivative from captured
first-order quantities. `_hvp_pass` uses reverse-over-reverse differentiation;
it differentiates the scalar directional derivative only into T. The metric
term is added once, while curvature draws are averaged.

`cbg_step(..., curvature="hvp")` accumulates the gradient of Eq. (9) at the
**center** parameters. It neither clears `.grad` nor performs an optimizer step.
It restores model modes, precision settings, and temporary trainability flags.

## Finite differences and precision

`curvature="fd"` uses two perturbed backward passes, with
`epsilon = r * sqrt(dim(U)) / sqrt(b.T @ K @ b)`. This fixes the RMS relative
movement at r. The reference implementation `spec_training_gradient` uses
functional parameter replacements; the scalable implementation temporarily
perturbs live weights. Its default `restore="copy"` restores weights exactly
in a `finally` block, including on forward/backward failure.

The FD path also averages base-loss gradients at the two endpoints. Thus it
approaches Eq. (9) with O(epsilon squared) error rather than giving the exact
single-point base gradient at finite radius. Prefer HVP for the paper method.
`restore="subtract"` saves memory but restores only up to floating-point
roundoff. Failed calls can leave partial gradients; discard those before retrying.
Zero shared directions contribute no curvature and do not divide by zero.

The paper config uses FP32 master weights, BF16 center captures, an FP64 reduced
solve, and TF32 HVP compute. These choices introduce numerical approximation;
"exact HVP" distinguishes the differentiation algorithm from finite differences,
not exact real arithmetic. Use FP32 compute and model-dtype captures to diagnose
roundoff, and FP64 tiny models for numerical oracles. BF16 HVP compute is not the
default when precision is unspecified.

## Validation

- Explicit toy Jacobians check the hook Gram and metric derivative.
- Autodifferentiation through dense toy spectral objectives checks G and the
  damping derivative.
- Enumerating all signs checks the estimator's conditional expectation. A
  production HVP test compares the complete gradient, including non-U trainable
  coordinates, with direct autodifferentiation of Eq. (9).
- HVP/FD comparisons check curvature and microbatch normalization, frozen U
  coordinates, multiple draws, and gradient accumulation.
- Failure injection checks restoration at both FD endpoints. Data tests cover
  masks, prompt-disjoint splits, manifest hashes, and strict truncation.

These are implementation checks, not evidence that a large-model run reproduces
the paper's attack outcomes.
