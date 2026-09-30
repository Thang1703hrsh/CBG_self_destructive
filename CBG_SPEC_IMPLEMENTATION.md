# Implementation-spec CBG

`cbg_version: spec` implements `L_aln + alpha L_bgn + beta log eta_k` from
`CBG_implementation_spec.md`. V2/V5 retain their existing raw single-mode,
Euclidean-metric experiments. Their metrics and hyperparameters are not
interchangeable with the normalized method.

## Implementation

- `src/cbg/spec.py`: one joint teacher-forced forward; safety means divided by
  sqrt(m); detached sampled Fisher targets and response-token SUMS divided by
  sqrt(n); global response-token NLL/KL; fixed-sample centered differences.
- `src/cbg/spec_hooks.py`: U is every `nn.Linear.weight` inside `model.layers`.
  Embeddings, norms, biases, and the LM head remain trainable in the outer
  problem. Both sequences of each safety example are added before contraction.
  Output-row chunks bound per-example gradient workspace; Gram multiplication
  and accumulation use FP64. Prompt activations contribute; padding does not.
- `src/cbg/spec_spectral.py`: detached per-step `mu = kappa_mu tr(K_bb)/n`,
  Cholesky reduced solve, top-k spectra, normalized eigenvectors, randomized
  scalarization and the full weight-relative damping derivative.
- `src/core/cbg_trainer.py`: injects the explicit gradient into the standard
  Trainer optimizer/accumulation loop. Reference logits are frozen. Checkpoints
  fingerprint scientific settings and data and validate runtime settings on
  resume. Trainer saves/restores the global RNG used for samples and signs.
- `CbgJointBatchDataset`: independent safety/benign permutations with explicit
  m and n; one dataset item is one complete joint batch. Shorter pools cycle.

Each endpoint uses stateless FP32/FP64 parameter replacement on U, without
changing live weights or existing `.grad` buffers. Dropout and autocast are
disabled; TF32 is disabled during geometry, endpoint and reference evaluation. On failure,
temporary hooks, model modes, and compute settings are restored. Degenerate
positive-mode or damping assumptions raise rather than introduce a hidden ridge.

Diagnostics include eta, log eta, S, N, mu, both Gram traces, full reduced
spectra, top-k gaps, perturbation epsilon, direction/correction norms,
preservation losses, stage timing, peak CUDA allocation and enabled ablations.

## Run the offline checks

```powershell
$env:USE_TF = "0"
python -m unittest discover -s tests -p test_cbg_spec.py -v
python -m unittest discover -s tests -q
```

These checks create tiny models locally; they download no weights or data.
T1 compares hooks against individual gradients and `vmap(grad)` with padding,
unequal response lengths, two-sequence safety ownership, and output chunking.
T2 checks an independent dense generalized eigenproblem and normalization.
T3 enumerates every sign combination and compares against a finite difference
of every coordinate, including coordinates outside U. A separate two-layer
Llama test checks the complete averaged training estimator against a directly
differentiated log-eta objective. T4 checks the damping term with frozen escape
directions. T5 checks SwiGLU rescaling. T6 compares endpoint differences to an
exact HVP over several radii. T7 checks bounds and safety scaling. T8 runs 20
tiny-model optimizer steps with decreasing geometry and bounded NLL/KL.

Hugging Face's Llama RMSNorm and eager attention unconditionally cast intermediate
values to FP32, even for double weights. The FP64 numerical tests replace these
two formulas **only inside the test fixture**, preserving FP64 arithmetic. This
does not change production model code. Target-model FP32 radius qualification
remains necessary; passing FP64 tests does not choose a production radius.

Two test descriptions in the source specification need a precise interpretation:

1. T3 must freeze the center's mu and Fisher samples when recomputing finite
   differences. Recomputing the damping rule at each endpoint would differentiate
   through mu, contradicting its explicit stop-gradient definition in section 4.1.
2. T4 must differentiate the full weighted expression in section 5.2, including
   both u and v families and their signs/normalizers. The abbreviated test-table
   expression alone is not d_Lambda.

## Pilot entry point

```text
python -m src.train --method cbg --config config/cbg_spec_pilot.yaml
```

The YAML contains explicit **uncalibrated pilot choices**, not selected research
hyperparameters. Set the dataset path, model, optimizer settings, xi, kappa_mu,
r, alpha and beta for the intended experiment. Default KL is
`KL(reference || current)` as specified; `kl_direction` also supports the reverse
direction pending the author's decision. `per_device_train_batch_size` must be
1 because m and n are inside each joint dataset item. Accumulation averages
separate geometry estimates; it does not increase a single Gram batch.

The benign Fisher samples are reused at both endpoints and across sign draws.
Additional draws currently rebuild the center for bounded graph lifetime, so
their cost includes center work as well as two endpoint passes. Implemented
ablation flags include `spec_damping`, `fisher`, `detach_lambda`, `num_sign_draws`,
`spec_objective`, `drop_aln`, `spec_solver`, `spec_perturb_precision`, k, m, n,
alpha and beta. The scenario extension adds raw-rho, explicit truncated block
Rayleigh and BF16 endpoint ablations; the default remains exact log-eta with
true FP32 endpoints. See `CBG_SCENARIO_RUNBOOK.md` for qualification and use.

Use `src.cbg.measure_geometry(model, batch, config=SpecConfig(...), generator=...)`
for checkpoint geometry without an optimizer step. It returns eta_k, rho_hat,
N_k, trace_K_bb and the other scalar diagnostics. Seed the generator identically
for comparable checkpoint probes; an explicit fixed token sample tensor is
also accepted. Parameters in U must be differentiable even during measurement.

## Qualification boundaries

The local verification run passed all **123 tests**, including the real Trainer
integration with gradient accumulation and bitwise-identical checkpoint resume.
Environment: Python 3.13.5, PyTorch 2.9.1+cpu, Transformers 4.57.6, Accelerate
1.4.0. The environment differs from the older Torch/Transformers pins in
`requirements.txt`; those pinned versions have not been requalified here.
Accelerate was installed only under the ignored `tmp/cbg_test_deps` directory
for this run. With the project's dependencies installed normally, the test
commands above need no additional Python path. The Trainer integration test
explicitly skips if Accelerate is unavailable.

The supported spec runtime is single-process, single-model-device FP32 with
eager attention and no gradient checkpointing. A frozen reference may reside on
a separate device. No multi-GPU sharding, large-model memory budget, throughput,
attack outcome, or paper-quality result is established by the offline tests.
The hook memory limit bounds contraction workspace, not total model, activation,
logit, optimizer, or direction storage. Complete S0 target-model precision and
memory checks before scaling to 7B.

The existing Modal launchers explicitly select legacy V5 configs. They have not
been converted into claims of spec qualification. The legacy SEAM comparison
code and research figure utilities are separate from the new training method.
