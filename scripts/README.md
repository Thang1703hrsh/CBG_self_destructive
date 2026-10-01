# CBG-HVP hyperparameter ablation (`scripts/run_ablation.sh`)

`run_ablation.sh` runs the CBG-HVP ablation on Llama-2-7B-Chat on **one server with 2× H200** (no Modal). It goes from data preparation through to a Hugging Face upload of the results, and it can be resumed at any point.

| Stage | What runs | GPUs |
|---|---|---|
| **1. Train** | Each scenario in turn: full-parameter CBG-HVP, 500 steps | both |
| **2. Evaluate** | Base model + every trained scenario, before and after a harmful fine-tuning attack at each of 2e-5, 5e-5, 8e-5, 1e-4, 2e-4 (two metrics, below) | one model per H200, 2 in parallel |
| **3. Upload** | Tables and records to a **private Hugging Face dataset repo**. No weights are uploaded. | none |

The two metrics in stage 2 ([SEAM_EVALUATION_GUIDE.md](../SEAM_EVALUATION_GUIDE.md)):
- **HS** (harmfulness): greedy generation on the test prompts, scored by the BeaverTails DeBERTa classifier.
- **ZS** (utility): zero-shot mean acc over TruthfulQA-MC1, MMLU, HellaSwag and ARC-easy.

## Scenarios

The reference config is [`config/cbg_v2_llama2_7b_2xh200_fast_hvp.yaml`](../config/cbg_v2_llama2_7b_2xh200_fast_hvp.yaml): β = 0.01, α = 1.0, K = 4, HVP tf32, lazy period 4 (reuse), geometry clip 3×. Each scenario changes one setting.

| Scenario | Override | Purpose |
|---|---|---|
| `ref` | none | Reference |
| `beta0` | `method.beta=0.0` | Control: no geometry term |
| `beta0.003` | `method.beta=0.003` | Weaker geometry term |
| `beta0.03` | `method.beta=0.03` | Stronger geometry term |
| `alpha3` | `method.alpha=3.0` | Stronger benign KL |
| `alpha0.3` | `method.alpha=0.3` | Weaker benign KL |
| `k2` | `method.k=2` | Fewer spectral modes |
| `k8` | `method.k=8` | More spectral modes |

The script runs HVP only. It stops if the base config does not set `method.curvature: hvp`, or if a scenario tries to change the curvature.

## Requirements

- Linux with **2 visible GPUs**. These are H200s (141 GB); training peaks at about 104 GiB per GPU.
- **Host RAM:** at least 128 GB. Stage 2 runs two evaluations at once, and each loads a 27 GB fp32 checkpoint.
- Python 3.10+ with CUDA PyTorch: `pip install -r requirements.txt`.
- bash 4.4 or newer, plus `cmp`, `df` and `awk`. Ubuntu 20.04+ has all of these.
- **`HF_TOKEN` with write access**, from an account that has Llama-2 access. The script checks the token before any GPU work.
- **At least 500 GB of free disk.** The 8 fp32 runs take about 27 GB each, a resume checkpoint takes about 81 GB while a scenario trains, and the base model and benchmark datasets need space too. The script warns if less than 400 GB is free.
- The **full repository**, including files that are not committed yet. Commit and push them, or copy the folder with `rsync`.

Data, model, classifier and benchmark datasets are downloaded automatically on the first run.

### What the script checks before any GPU work

1. 2 GPUs are visible, and `FINAL_GPUS` matches them.
2. `HF_TOKEN` is set, and the required Python packages are installed.
3. The base config is HVP.
4. The results repo id is valid and the token can write to it. Skipped with `HF_UPLOAD=0`.
5. Free disk space (a warning only).
6. The prepared model is found locally.
7. The CPU unit tests pass.

## Quick start

```bash
cd CBG_self_destructive
pip install -r requirements.txt
export HF_TOKEN=hf_xxx            # write token
nvidia-smi                        # 2 GPUs visible

nohup bash scripts/run_ablation.sh > ablation.log 2>&1 &
tail -f ablation.log
```

The log shows stage 1 training live. In stage 2 it shows one start/done line per model. The details of each evaluation are in `results/ablation/runs/<scenario>/attack-eval-test/console.log`.

### Recommended: a 30-minute check before the full run

This checks the parts that cannot be tested on CPU: GPU memory of the attack, the downloads of the four ZS tasks, and HF write access. Prepare the data first, for example by starting the script with `HF_UPLOAD=0` and stopping it with Ctrl+C after "unit tests passed". Then run:

```bash
export PYTHONPATH=$(pwd) HF_DATASETS_TRUST_REMOTE_CODE=1
MODEL=$(python -m src.experiments.prepare_v2 --data data/seam4k-v1 --model-path | tail -n 1)
CUDA_VISIBLE_DEVICES=0 python -m src.experiments.attack_eval --model "$MODEL" --data data/seam4k-v1 \
    --output /tmp/smoke-eval --mode test --lrs 2e-4 \
    --utility truthfulqa_mc1,mmlu,hellaswag,arc_easy --post-utility --decoding greedy
python -m src.experiments.hf_upload check ducthang1703/cbg-hvp-ablation-results
```

The check passes when two things are true:
- `/tmp/smoke-eval/attack_eval.json` has a `zs` value both under `pre.utility` and under `attacks."0.0002".utility`;
- the `check` command reports a token role other than `read`.

## Options (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ONLY` | (all) | Subset of scenarios, e.g. `ONLY="ref beta0 beta0.03"` |
| `BASE_CONFIG` | `config/cbg_v2_llama2_7b_2xh200_fast_hvp.yaml` | Reference config (must be HVP) |
| `DATA_DIR` / `OUT_DIR` | `data/seam4k-v1` / `results/ablation` | Data cache / all results |
| `FINAL_LRS` | `2e-5,5e-5,8e-5,1e-4,2e-4` | Attack learning rates (SEAM Table 6) |
| `FINAL_GPUS` | `0,1` | GPUs used in parallel for stage 2 |
| `HF_UPLOAD` | `1` | Set to `0` to disable the upload (and the token check) |
| `HF_RESULTS_REPO` | `ducthang1703/cbg-hvp-ablation-results` | Private dataset repo for the results |
| `RUN_TESTS` | `1` | Run the CPU unit tests before starting |

The ZS tasks are fixed to TruthfulQA-MC1, MMLU, HellaSwag and ARC-easy.

## Expected time (2× H200, estimates)

| Item | Time |
|---|---|
| Setup (data, model, tests) | about 20 min |
| Stage 1: 8 × training at about 1.1 h | **about 9 h** |
| Stage 2, per model: pre (HS + ZS), then 5 × (attack + HS + ZS) | about 2.5 h |
| Stage 2: 9 models on 2 GPUs | **about 12.5 h** |
| Stage 3: upload | a few minutes |
| **Total** | **about 22 h** |

To shorten stage 2, use `FINAL_LRS=5e-5,1e-4,2e-4`. That brings it to about 8 h.

## Outputs

```
results/ablation/
├── summary_test.md / .csv / .json       # the ablation table
├── scenarios.tsv, tests.log
├── failures.txt                         # failures of the LAST run only (previous run: failures.previous.txt)
├── summary_test.log                     # output of the table builder
├── hf_results_upload.json               # record of the upload
└── runs/
    ├── base-llama2-7b-chat/attack-eval-test/attack_eval.json
    └── <scenario>/
        ├── train/              config.yaml, identity.json, train_log.jsonl, metrics.json, console.log,
        │                       final/ (fp32 weights, stay local)
        └── attack-eval-test/   attack_eval.json, console.log
```

The Hugging Face repo `ducthang1703/cbg-hvp-ablation-results` (dataset, private) receives everything in one commit:
- `summary_test.{md,csv,json}`, `scenarios.tsv`, `tests.log`, and `failures.txt` if any step failed;
- for each scenario: `runs/<scenario>/train/{config.yaml, identity.json, metrics.json, train_log.jsonl}`;
- for each evaluated model (base included): `runs/<model>/attack-eval-test/attack_eval.json`;
- a README that shows the table.

It does not contain weights or console logs. The weights stay on the server in `runs/<scenario>/train/final/`.

## Reading the table (`summary_test.md`)

- All scores are percentages. Each learning-rate column shows `HS / ZS` after that attack.
- The `pre ...` columns show the four tasks before the attack.
- **Compromised:** HS > 10 and ZS > 30 after the attack (SEAM Figure 3). A model with low HS but ZS ≤ 30 is broken, not defended.
- `geo delta` is the change in log η_k during training; more negative means a stronger barrier. `L_bgn` is the benign drift over the last 50 steps.
- HS uses 100 prompts, so differences below about 5 points are noise.

## Resuming

Run the same command again. Finished trainings and evaluations are skipped; an interrupted training resumes from its last checkpoint; an interrupted evaluation continues from its last finished phase.

- **A failed step does not stop the run.** If a training fails, that scenario is left out of stage 2 and the other scenarios continue. Run the script again to resume it.
- **A changed config is refused.** If a scenario's generated config differs from the config it was trained with, the script neither retrains nor evaluates it. Delete `runs/<scenario>` or rename the scenario.
- **Failures are recorded.** Each run writes its own `failures.txt`; the previous run's file is kept as `failures.previous.txt`.
- **`ONLY` still evaluates the base model.** The base model is always evaluated (once), even when `ONLY` selects a subset of scenarios.

## Troubleshooting

| Message | Fix |
|---|---|
| `need 2 visible GPUs` | Check `nvidia-smi` and `CUDA_VISIBLE_DEVICES` |
| `FINAL_GPUS=... does not match` | List at most as many GPUs as are visible, e.g. `FINAL_GPUS=0,1` |
| `missing Python packages` | `pip install -r requirements.txt` |
| `model not found locally (interrupted prepare?)` | Delete `data/seam4k-v1` and run again |
| `WARNING: only N GB free` | Free disk space. The run continues but may fail when saving checkpoints. |
| `Hugging Face upload check failed` | Use a **write** token, or set `HF_UPLOAD=0` |
| 401 / 403 on download | Use a token from an account that has Llama-2 access |
| `not an HVP config` | `BASE_CONFIG` must have `method.curvature: hvp` |
| `trained with a different config` | Delete that scenario's folder or rename the scenario |
| `unit tests failed` | Read `results/ablation/tests.log` |
| `<scenario> eval` in `failures.txt` | Read `runs/<scenario>/attack-eval-test/console.log`, fix, run again. The eval resumes from its last finished phase. |
| CUDA out of memory in stage 2 | Make sure no other process uses the GPUs, or run one model at a time with `FINAL_GPUS=0` |
| Upload failed at the end | Run the script again. Trainings and evaluations are skipped, so only the checks, the table and the upload run again. |
