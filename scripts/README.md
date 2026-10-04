# Running on the 2× H200 server

## Setup (once)

```bash
cd ~/CBG_self_destructive
pip install -r requirements.txt
export HF_TOKEN=hf_xxx        # Llama-2 + Mistral access, WRITE permission
```

## 1. CBG on Qwen2.5-7B and Mistral-7B

```bash
nohup bash scripts/run_new_models.sh > new_models.log 2>&1 &
tail -f new_models.log
```

Results: `results/new-models/<model>/summary_test.md`

## 2. Mechanism experiments (β = 0 ablation + attack trajectory)

```bash
nohup bash scripts/run_mechanism.sh > mechanism.log 2>&1 &
tail -f mechanism.log
```

Results: `results/mechanism/summary_mechanism.md`

## Both, one after the other

```bash
nohup bash -c "bash scripts/run_new_models.sh > new_models.log 2>&1; bash scripts/run_mechanism.sh > mechanism.log 2>&1" &
```

Do not run the two scripts at the same time (each needs both GPUs).

## Notes

- Interrupted: run the same command again; finished steps are skipped.
- Errors: `cat results/new-models/*/failures.txt` or `cat results/mechanism/failures.txt`.
- Everything (weights, results, losses, logs) is pushed to Hugging Face as **public**. Add `HF_PUBLIC=0` for private, `HF_UPLOAD=0` for no upload.
- One model only: `ONLY_MODELS=qwen25-7b` (script 1) or `ONLY_MODELS="base cbg"` (script 2).
