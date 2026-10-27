# Repository review

The review used the accompanying 25-page paper, with particular attention to
Eqs. (2), (3), (7)--(10), (13), Algorithm 1, and Table 10. The implementation
map is in [method.md](method.md). No large-model training or external uploads
were performed during this cleanup.

## Changes

| Finding | Resolution |
| --- | --- |
| The README described SEAM as this project's paper and linked missing runbooks. | Replaced with CBG setup, method entry points, tested contracts, and attribution. |
| Historical `final` configs include trajectory penalties, cached gradients, and balancing outside Eq. (9). | Added `config/cbg_paper.yaml`; documented the distinct config families. Historical configs remain unchanged so old checkpoints are interpretable. |
| Failed FD passes could leave live parameters perturbed. | Restore in `finally`; regression coverage for failures at either endpoint and both restore modes. Partial gradients must still be discarded after failure. |
| HVP differentiation failed if a weight in U was frozen in T; metric injection could create gradients on frozen weights. | Temporarily enable the inner differentiation coordinates, backpropagate only to T, and restore trainability. Do not inject metric gradients into frozen weights. |
| A zero shared direction caused division by zero; an absolute spectral cutoff discarded small valid safety scales. | Skip zero curvature directions, retain their base update, recognize identical safety pairs explicitly, and retain small positive spectra. |
| A closed-gap warning overstated a Clarke-subgradient claim. | State that differentiability of the log ratio is not guaranteed, consistent with Appendix A.7. |
| Unspecified HVP precision defaulted to BF16 despite the code's own accuracy warning. | Default to FP32; the paper config explicitly requests TF32. |
| The utility gate used a hard-coded 20-step window. | Make the window configurable and set the paper recipe to 25, including resumed runs. |
| Legacy data preparation copied row splits and could leak prompts. | Add a separate paper split policy, reserve disjoint benign fine-tuning data, and allow exclusions from other baselines' training sets. Keep legacy split behavior explicit. |
| SEAM-compatible tokenization silently treated a prompt token as a response after truncation. | The paper format rejects these records. Legacy SEAM format retains its historical behavior. |
| A new paper run could use stale legacy pools. | Require the paper manifest schema, preparation identity, counts, and file hashes. |
| A logged exception kept its traceback and solver tensors alive under captured logging. | Log its string, allowing the failed solve's tensors to be released before retrying. |
| The Transformers parser assumed a mutable metadata copy. | Give it a private mutable copy without altering the original dataclass field. |
| Tracked baseline copies, results, generated configs, and personal launchers cluttered the release. | Preserve them under ignored `tmp/local-archive/` and remove them from the public source tree. Existing manuscript/figure deletions were left as found. |

The broad `*.json`/`*.jsonl` ignore rules were replaced with directory-specific
data/output exclusions, allowing future small fixtures and metadata to be
versioned. Added CPU CI, a test-discovery config, explicit core/test dependency
sets, and portable line-ending rules. The existing license is preserved.

## Verification and remaining limitations

Final local check: **213 passed, 4 skipped**, plus 59 passing subtests
(`python -m pytest -q --disable-warnings`). The skips are optional integrations
whose dependencies are absent locally; they remain enabled in CI when installed.
The paper-policy data-to-training smoke test completes two HVP optimizer steps
on a tiny local model. Python compilation, Bash launcher syntax checks,
documentation-link checks, and `git diff --check` also pass. The 196 archived
tracked files were compared with HEAD to verify their preserved contents.

Tests include independent explicit-Jacobian and spectral-autodiff oracles,
enumeration of all eigen-sign draws for the complete production HVP,
microbatch/HVP/FD comparisons, frozen-coordinate handling, and restoration on
injected failures. The training runner is exercised on tiny local Llama models,
including checkpoint/resume and utility gates. Prompt separation, strict masks,
and manifest changes have dedicated regressions.

The local environment has Python 3.15.1, a CPU-only PyTorch development build
(`2.16.0.dev20261026+cpu`), and Transformers 4.52.4. It lacks the optional
`datasets`, `peft`, and `accelerate` integration dependencies. The release
dependency pins and Python 3.11/3.12 CI configuration must also be validated in
that supported environment; adding CI does not mean a hosted run has occurred.

The method review does **not** establish reproduction of the reported 7B
experiments. In particular:

- The new single-point recipe must be retrained; historical trajectory/balanced
  checkpoints cannot be relabeled as Eq. (9) results. The paper's complete model
  revision and original data/run manifests are not supplied by this cleanup.
- A comparison's additional baseline-training prompt exclusions must be supplied
  before preparation. Without them, separation is checked against this release's
  pools, not unavailable baseline data.
- Strict response masks may expose truncated records in the public sources.
  Record and disclose any filtering decision; it changes the training sample.
- Historical attacks/evaluation helpers implement different batching and
  generation conventions. Distinguish Main from Direct-batch and padding-included
  from padding-masked attacks, as the paper does. Do not pool their results.
- The local normalized objective is not a bound on arbitrary attack trajectories
  or a robustness guarantee. BF16 capture/TF32 compute, finite differences,
  and minibatch sampling introduce their respective numerical/statistical errors.
- Source cleanup does not erase files from existing Git history. No history
  rewriting, commits, remote changes, or publication were performed.
