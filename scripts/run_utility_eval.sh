#!/usr/bin/env bash
# HellaSwag + MMLU utility (SEAM protocol: bf16, pad = eos, lm-eval zero-shot, batch 24) on a server with 2x H100.
# The base model runs on GPU 0 and CBG-HVP on GPU 1, in parallel; then a comparison table is printed.
#
#   export HF_TOKEN=hf_xxx
#   bash scripts/run_utility_eval.sh
#   CBG_MODEL=results/ablation/runs/ref/train/final bash scripts/run_utility_eval.sh     # a local checkpoint
#   LIMIT=50 bash scripts/run_utility_eval.sh                                            # smoke test (~5 min)
#   nohup bash scripts/run_utility_eval.sh > utility.log 2>&1 &                          # survives logout
#
# Finished models are skipped (delete <OUT_DIR>/<name>/utility.json to redo one).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd) USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled OMP_NUM_THREADS=4

BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}
CBG_MODEL=${CBG_MODEL:-ducthang1703/cbg-hvp-llama2-7b}   # HF repo or local folder
TASKS=hellaswag,mmlu                                     # fixed: only these two benchmarks
BATCH_SIZE=${BATCH_SIZE:-24}
OUT_DIR=${OUT_DIR:-results/utility}
LIMIT=${LIMIT:-}
BASE_GPU=${BASE_GPU:-0}
CBG_GPU=${CBG_GPU:-1}

[[ -n "$LIMIT" ]] && OUT_DIR="$OUT_DIR/smoke-$LIMIT"
log() { echo "[$(date '+%F %T')] $*"; }

gpus=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
python -c "import lm_eval, transformers" 2>/dev/null || { log "missing packages: pip install -r requirements.txt"; exit 1; }
if [[ "$gpus" -lt 2 ]]; then log "need 2 visible GPUs, found $gpus"; exit 1; fi
if [[ -z "${HF_TOKEN:-}" ]] && [[ ! -f "$HOME/.cache/huggingface/token" ]]; then
  log "set HF_TOKEN (Llama-2 access, and access to $CBG_MODEL if it is a private repo)"; exit 1
fi

run() {   # run <name> <model> <gpu>
  local name=$1 model=$2 gpu=$3 out="$OUT_DIR/$1"
  if [[ -f "$out/utility.json" ]]; then log "[$name] already done"; return 0; fi
  mkdir -p "$out"
  log "[$name] $model -> $TASKS on GPU $gpu"
  # The tokenizer always comes from the base model (CBG does not change it).
  CUDA_VISIBLE_DEVICES=$gpu python -m src.experiments.utility_eval --model "$model" --tokenizer "$BASE_MODEL" \
      --output "$out" --tasks "$TASKS" --batch-size "$BATCH_SIZE" ${LIMIT:+--limit "$LIMIT"} > "$out/console.log" 2>&1
  local status=$?
  if [[ $status -eq 0 ]]; then log "[$name] done"; else log "[$name] FAILED, see $out/console.log"; fi
  return $status
}

# Download the shared files once, so the two parallel jobs do not race on the HF cache.
python - "$BASE_MODEL" "$CBG_MODEL" <<'EOF' || { log "download failed"; exit 1; }
import os, sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    if not os.path.isdir(repo):
        snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
EOF

run base "$BASE_MODEL" "$BASE_GPU" & pid_base=$!
run cbg-hvp "$CBG_MODEL" "$CBG_GPU" & pid_cbg=$!
log "running in parallel (logs: $OUT_DIR/{base,cbg-hvp}/console.log)"
wait $pid_base; s1=$?
wait $pid_cbg; s2=$?

python - "$OUT_DIR" "$TASKS" <<'EOF'
import json, sys
from pathlib import Path
from src.experiments.utility_eval import table
out, tasks = Path(sys.argv[1]), sys.argv[2].split(",")
scores = {n: json.loads((out / n / "utility.json").read_text())["scores"]
          for n in ("base", "cbg-hvp") if (out / n / "utility.json").exists()}
if scores:
    text = table(scores, tasks, base="base")
    (out / "summary.md").write_text(text + "\n")
    print(text)
EOF
[[ $s1 -eq 0 && $s2 -eq 0 ]] || exit 1
