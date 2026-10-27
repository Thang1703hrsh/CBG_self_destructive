# Collateral Barrier Geometry (CBG)

Research code for **No Cheap Escape: Collateral Barrier Geometry for Self-Destructive Language Models**.
[Paper](830_No_Cheap_Escape_Collateral.pdf) · [Method and implementation](docs/method.md) · [Review and limitations](docs/review.md)

CBG couples safety-sensitive parameter changes to changes on benign examples. Its training objective is

```text
L = L_aln + alpha * KL(p_reference || p_current) + beta * log(eta_k)
eta_k = sum_top_k_eigenvalues(S_mu) / sum_top_k_eigenvalues(K_hh)
S_mu = K_hh - K_hb @ solve(K_bb + mu * I, K_hb.T)
mu = kappa_mu * trace(K_bb) / n
```

The method searches the linear-layer weights inside transformer blocks. The optimizer can train a different set of parameters. The small Gram matrix avoids a parameter-sized Hessian or a full per-example Jacobian. The default paper recipe uses a shared eigen-sign Hessian-vector product and differentiates through damping and the relative metric.

## Setup

Use Python **3.11 or 3.12** in a virtual environment, from the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`requirements-core.txt` contains the numerical core and tiny-model dependencies. `requirements-dev.txt` adds the offline test dependencies. `requirements.txt` also installs the full training/evaluation and legacy integration dependencies. Install a CUDA-enabled PyTorch build appropriate for your server when running GPU experiments.

The tests construct tiny models locally; they do not require model downloads, credentials, or paid API calls. CI runs CPU tests on Python 3.11 and 3.12. Large-model training needs substantially more memory; the paper reports two 80 GB H100 GPUs. Layer-split execution is supported, and actual fit and runtime must be checked on the target model.

## Paper training path

Use **[config/cbg_paper.yaml](config/cbg_paper.yaml)** with `src.cbg.train_v2`. Other configs are historical experiments; several named `final` include objectives or gradient transformations outside the paper recipe. See [config/README.md](config/README.md).

Preparation downloads the public refusal data and datasets, checks source hashes, creates prompt-disjoint splits, and records the resolved model revision. Set `HF_TOKEN` if the chosen model requires access. Before preparing a comparison with additional baselines, put their training prompts in `data.eval_excluded_prompts` so the common test set excludes them too.

```bash
python -m src.experiments.prepare_v2 --config config/cbg_paper.yaml --data data/paper
python -m src.experiments.prepare_v2 --data data/paper --model-path
```

Use the local snapshot path printed by the second command as `--model`:

```bash
python -m src.cbg.train_v2 --config config/cbg_paper.yaml --data data/paper --model /path/to/snapshot --output results/smoke --stage smoke
python -m src.cbg.train_v2 --config config/cbg_paper.yaml --data data/paper --model /path/to/snapshot --output results/paper --stage train
# Resume an interrupted run with the same arguments plus --resume.
```

The paper format rejects examples whose entire response was truncated, instead of treating a prompt token as a response. Inspect the preparation truncation report before training; a record that fails this check needs an explicit, recorded data-filtering decision. This release does not silently filter or replace records to make a run proceed.

For numerical diagnosis, `--stage hvp-check` compares HVP and finite-difference curvature estimates. `--stage geometry` evaluates fixed held-out batches without training. These checks do not establish attack robustness. Training checkpoints and run records stay local unless an explicit upload command is invoked.

## Method API

```python
from src.cbg import StepConfig, cbg_step

config = StepConfig(
    xi=1e-3, kappa_mu=1.0, r=1.1e-7,
    alpha=1.0, beta=0.1, k=4, curvature="hvp",
)
optimizer.zero_grad(set_to_none=True)
metrics = cbg_step(model, batch, config, reference=reference_logits)
optimizer.step()
```

`batch` contains `harmful`, `safe`, and `benign` mappings with `input_ids`, `attention_mask`, and `response_mask`. Each sequence needs at least one predicted response token. `reference_logits` contains frozen-reference logits for benign response positions, ordered by sequence then token; see `reference_logits()` in `src/cbg/step_v2.py` for chunked extraction.

The model must use FP32 master weights (FP64 is supported for validation). Disable dropout and use attention operations that support double backward; the runner selects eager attention. The API adds to `.grad`; it does not clear gradients or step an optimizer. After a failed call, clear partial gradients before retrying. Plain Linear hooks reject LoRA wrappers, shared admissible weights, and repeated Linear invocations. LoRA attacks remain a separate evaluation scenario.

## Repository layout

- `src/cbg/`: normalized spectral solve, chunked Linear contractions, gradient estimators, and training runner.
- `src/data_processing/cbg_datasets.py`: paired safety and benign tokenization and collation.
- `src/experiments/`: preparation, fixed-batch geometry probes, attacks, evaluation, and explicit upload helpers.
- `tests/`: numerical oracles, gradient and masking regressions, and tiny-model integration tests.
- `src/core/cbg_trainer.py`, `src/train.py`: retained experimental Trainer integration; older V2/V5 objectives differ from the paper.

The unrelated vendored baseline workspace, local results, generated configs, paid refusal-generation script, and account-specific launchers have been excluded from the release. The cleanup preserves these local files under ignored `tmp/local-archive/`; that archive is not part of a fresh clone.

## Scope and attribution

CBG optimizes a local normalized surrogate. It does not guarantee robustness to arbitrary fine-tuning or weight edits. The supplied source does not establish reproduction of the paper's tables; see [review notes](docs/review.md) for the remaining experimental provenance and evaluation gaps.

The project retains code derived from the Self-Destructive Language Models (SEAM) and representation-noising projects.
