# Scripts

Run commands from the repository root. The Python modules are the portable entry points.

- `train_cbg.sh`: wrapper for `python -m src.cbg.train_v2 --config config/cbg_paper.yaml`. Pass `--data`, `--model`, `--output`, and optionally `--stage` or `--resume`.
- `run_utility_eval.sh`: compare a supplied `CBG_MODEL` with `BASE_MODEL` on two visible GPUs. The default tasks are TruthfulQA-MC1, MMLU, HellaSwag, and ARC-Easy. Set `TASKS` to restrict the task list.
- `cbg_fixed_batch_descent.py` and `cbg_five_step_probe.py`: development probes, not paper-training entry points. Use their `--help` output for arguments.

For data preparation, use `python -m src.experiments.prepare_v2 --help`. For evaluation, use `python -m src.experiments.attack_eval --help` or `python -m src.experiments.utility_eval --help`.

Account-specific PowerShell/Modal launchers and old orchestration scripts are preserved in the ignored local cleanup archive. They are excluded from the source release. The retained scripts do not automatically publish models or results.
