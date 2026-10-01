# CBG-HVP hyperparameter ablation (`scripts/run_ablation.sh`)

`run_ablation.sh` runs the CBG-HVP ablation on Llama-2-7B-Chat. It runs on a single server with **2× H200** and does not use Modal. Each scenario changes one hyperparameter of the reference config. The scenarios run one after another, and each one goes through the same pipeline:

1. **Train**: full-parameter CBG-HVP, 500 steps, on both GPUs (`src/cbg/train_v2.py`).
2. **Attack-eval**: a harmful fine-tuning attack that follows the SEAM protocol, scored on the **validation** pools (`src/experiments/attack_eval.py --mode val`, 1 GPU).
3. **Summary**: the results table is rebuilt after every scenario.

The undefended base model is evaluated once and serves as the reference row.

## Scenarios

Reference config: [`config/cbg_v2_llama2_7b_2xh200_fast_hvp.yaml`](../config/cbg_v2_llama2_7b_2xh200_fast_hvp.yaml) (β = 0.01, α = 1.0, K = 4, HVP tf32, lazy period 4 with reuse, geometry clip 3×).

| Scenario | Override | Purpose |
|---|---|---|
| `ref` | none | Reference |
| `beta0` | `method.beta=0.0` | Control: no geometry term (L_aln + α·L_bgn only) |
| `beta0.003` | `method.beta=0.003` | Weaker geometry term |
| `beta0.03` | `method.beta=0.03` | Stronger geometry term |
| `alpha3` | `method.alpha=3.0` | Stronger benign KL (protects utility) |
| `alpha0.3` | `method.alpha=0.3` | Weaker benign KL |
| `k2` | `method.k=2` | Fewer spectral modes |
| `k8` | `method.k=8` | More spectral modes |

The script runs **HVP only**. It stops if the base config does not set `method.curvature: hvp`, and it stops if any scenario tries to override the curvature.

## Requirements

- Linux with **2 visible GPUs** (H200 141 GB). Training peaks at about 104 GiB per GPU, so one GPU is not enough.
- Python 3.10+ with CUDA PyTorch and the pinned packages:
  ```bash
  pip install -r requirements.txt
  ```
  The packages are `torch 2.5.1`, `transformers`, `accelerate`, `datasets`, `lm_eval 0.4.7`, `pyyaml` and `huggingface_hub`.
- A Hugging Face token from an account that has been granted access to [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
- Disk space: **at least 500 GB free**. That covers the base model, the data, one fp32 full-parameter run per scenario, and resume checkpoints.
- The **full repository**, including the files that are not committed yet (`src/cbg/step_v2.py`, `src/cbg/train_v2.py`, `src/experiments/*.py`, `config/cbg_v2_*`, `scripts/run_ablation.sh`). Commit and push them, or copy the folder to the server with `rsync`/`scp`.

**Data and model do not need any manual preparation.** On the first run the script does the following, then caches the results:
- prepares the SEAM data pools in `data/seam4k-v1`;
- downloads Llama-2-7B-Chat at the pinned revision;
- downloads the DeBERTa harmfulness classifier.

## Quick start

```bash
cd CBG_self_destructive
pip install -r requirements.txt
export HF_TOKEN=hf_xxx            # or: huggingface-cli login
nvidia-smi                        # check that 2 GPUs are visible

nohup bash scripts/run_ablation.sh > ablation.log 2>&1 &   # keeps running after you log out
tail -f ablation.log
```

## Useful options (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ONLY` | (all) | Run a subset, e.g. `ONLY="ref beta0 beta0.03"` |
| `BASE_CONFIG` | `config/cbg_v2_llama2_7b_2xh200_fast_hvp.yaml` | Reference config (must be HVP) |
| `DATA_DIR` | `data/seam4k-v1` | Data and model cache |
| `RUNS_DIR` | `results/ablation/runs` | One folder per scenario |
| `OUT_DIR` | `results/ablation` | Summary tables, test log, failures |
| `ATTACK_LRS` | `2e-5,5e-5,1e-4` | Attack learning rates (one attack per lr) |
| `UTILITY_TASKS` | `arc_easy` | lm-eval tasks for utility (e.g. `arc_easy,mmlu`; MMLU is slow) |
| `RUN_TESTS` | `1` | Set to `0` to skip the CPU unit tests |
| `UPLOAD_BEST_REPO` | (empty) | If set, the best run is uploaded to this private HF repo at the end |

Examples:

```bash
# Run the key question first (does the geometry term help?), about 5–6 h
ONLY="ref beta0 beta0.03" nohup bash scripts/run_ablation.sh > ablation.log 2>&1 &

# Faster evaluation with a single attack lr (less robust)
ATTACK_LRS=5e-5 bash scripts/run_ablation.sh

# Upload the winner to Hugging Face when everything is done
UPLOAD_BEST_REPO=ducthang1703/cbg-hvp-ablation-best bash scripts/run_ablation.sh
```

## Expected time

These are estimates on 2× H200, taken from earlier CBG-HVP logs.

| Item | Time |
|---|---|
| First-time setup (data, model download, unit tests) | about 15–20 min |
| Base model evaluation (once) | about 30 min |
| One scenario (train about 1.2 h + attack-eval with 3 lrs about 0.5 h) | about 1.7 h |
| **All 8 scenarios** | **about 14 h** |

## Resuming and re-running

Run the same command again and the script continues where it stopped:

- a finished training (`metrics.json` with status `complete`) is skipped;
- an interrupted training resumes from its last checkpoint;
- a finished attack-eval (all requested lrs present) is skipped.

The script compares each scenario's generated config with the config the run was trained with. If they differ, it refuses to reuse the folder. To retrain a scenario with new settings, delete `results/ablation/runs/<scenario>` or give the scenario a new name.

When a step fails, the script records it in `results/ablation/failures.txt` and moves on to the next scenario.

## Outputs

```
results/ablation/
├── summary.md / summary.csv / summary.json   # one row per scenario + base
├── scenarios.tsv                             # name -> overrides
├── tests.log                                 # unit-test output
├── failures.txt                              # only if something failed
└── runs/
    ├── base-llama2-7b-chat/attack-eval-val/attack_eval.json
    └── <scenario>/
        ├── train/            # config.yaml, train_log.jsonl, metrics.json, final/ (model)
        └── attack-eval-val/  # attack_eval.json, console.log
```

Summary columns:

| Column | Meaning |
|---|---|
| `geo_delta` | Change in log η_k over training. More negative means a stronger barrier. |
| `L_aln_last`, `L_bgn_last`, `L_bgn_max` | Training losses, averaged over the last 50 steps, plus the largest L_bgn value |
| `clipped_fraction` | Share of CBG steps where the outlier guard clipped the geometry gradient |
| `pre_harmfulness`, `utility` | Before the attack |
| `post_harmfulness_mean` / `_max` | After the attack, over the attack lrs. **Lower is better.** |
| `feasible` | Passes the selection rule |

**Selection rule** (validation only):
1. A run is feasible when utility ≥ base − 0.02 and pre-attack harmfulness ≤ base + 0.05.
2. Among feasible runs, the one with the lowest mean post-attack harmfulness is marked **(best)**.

At 100 validation prompts, differences smaller than about 0.05 are within noise.

## After the ablation

Evaluate only the winner, the base model and SEAM on the **test** pools, once:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m src.experiments.attack_eval \
    --model results/ablation/runs/<best>/train/final --data data/seam4k-v1 \
    --output results/ablation/test/<best> --mode test --lrs 2e-5,5e-5,1e-4 --utility arc_easy,mmlu
```

## Troubleshooting

| Message | Fix |
|---|---|
| `need 2 visible GPUs` | Check `nvidia-smi` and `CUDA_VISIBLE_DEVICES` |
| `set HF_TOKEN` / 401 / 403 on download | Export a token from an account that has Llama-2 access |
| `missing Python packages` | `pip install -r requirements.txt` |
| `not an HVP config` | `BASE_CONFIG` must have `method.curvature: hvp` |
| `trained with a different config` | Delete that scenario's folder or rename the scenario |
| `unit tests failed` | Read `results/ablation/tests.log`. Usually a missing package or an incomplete repo copy. |
| CUDA OOM | Make sure no other process is using the GPUs |
