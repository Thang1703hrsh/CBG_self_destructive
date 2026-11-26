# Scripts

Run commands from the repository root. The Python modules are the portable entry points.

- `train_cbg.sh`: wrapper for `python -m src.cbg.train_v2 --config config/cbg_paper.yaml`. Pass `--data`, `--model`, `--output`, and optionally `--stage` or `--resume`.
- `run_utility_eval.sh`: compare a supplied `CBG_MODEL` with `BASE_MODEL` on two visible GPUs. The default tasks are TruthfulQA-MC1, MMLU, HellaSwag, and ARC-Easy. Set `TASKS` to restrict the task list.
- `cbg_fixed_batch_descent.py` and `cbg_five_step_probe.py`: development probes, not paper-training entry points. Use their `--help` output for arguments.

For data preparation, use `python -m src.experiments.prepare_v2 --help`. For evaluation, use `python -m src.experiments.attack_eval --help` or `python -m src.experiments.utility_eval --help`.

Account-specific PowerShell/Modal launchers and old orchestration scripts are preserved in the ignored local cleanup archive. They are excluded from the source release. Apart from `run_baselines.sh` (below), the scripts do not publish models or results.

## Baselines: Base, SEAM, SDD, CTRAP on Qwen2.5-7B and Mistral-7B

Requirements: Linux, at least 2 GPUs with 80 GB each (one SEAM run uses 3 GPUs if available, otherwise 2), about
300 GB of free disk for checkpoints and the HF cache (set `HF_HOME` to move the cache), and a Hugging Face token
with write access.

```bash
bash scripts/setup_baselines.sh          # once: creates .venv and .venv-ctrap
export HF_TOKEN=hf_...
bash scripts/run_baselines.sh --smoke    # optional: ~30 min pipeline check on Qwen2.5-0.5B, no upload
nohup bash scripts/run_baselines.sh --seeds 42,43,44 > baselines.log 2>&1 &
```

The run covers Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3, seeds 42, 43 and 44, all four methods, and every
visible GPU. For each model and seed it trains SEAM, SDD and CTRAP (500 steps each), then evaluates Base and the
three defenses before and after harmful fine-tuning at learning rates 2e-5, 5e-5, 8e-5, 1e-4 and 2e-4.
On 2×H100 the full run takes about 30 hours.

Where things go:

- Results: `outputs/baselines/RESULTS.md` (HS and ZS, mean ± std over seeds), `outputs/baselines/results.csv`
  (every seed), and `outputs/baselines/results/<model>/seed<S>/<method>/cases/` (per-case metrics and responses).
- Hugging Face (public): every checkpoint as `<hf-user>/<method>-<model>-seed<S>-step500`, and at the end the
  results table and per-case metrics (no generated responses) as the dataset
  `<hf-user>/self-destruct-baselines-results`.
- Progress: `tail -f baselines.log`, `outputs/baselines/status.json`, `outputs/baselines/logs/<job>.log`.

If something fails, fix the cause shown in the job's log and rerun the same command: finished jobs are skipped and
failed jobs are retried. Stop a run with `kill <PID of run_all.py>`; its running jobs stop too.
Options (`bash scripts/run_baselines.sh --help`): `--models qwen`, `--seeds 42,43`, `--gpus 0,1,2`, `--no-push`,
`--hf-namespace ORG`, `--seam-gpus 2|3`, `--dry-run` (list the jobs without running them).

The baseline code lives in `baselines/`: the SEAM trainer (MIT license, `baselines/seam/LICENSE`), the
reconstructed SDD trainer, the CTRAP adapter with the upstream CTRAP trainer files, and the shared evaluation
harness. Evaluation uses greedy decoding and the same attack data for all methods of a seed.
