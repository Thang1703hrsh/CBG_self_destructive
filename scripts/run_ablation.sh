#!/usr/bin/env bash
# CBG-HVP hyperparameter ablation on ONE server with 2x H200 (no Modal).
# HVP only: the base config must use method.curvature: hvp, and no scenario may change the curvature.
#
# Stage 1: train every scenario, one after another (2 GPUs, src/cbg/train_v2.py).
# Stage 2: evaluate the base model and every trained scenario, one model per H200, two in parallel
#   (SEAM_EVALUATION_GUIDE.md), before and after a harmful fine-tuning attack at each of FINAL_LRS:
#     HS  greedy generation on the test prompts, BeaverTails DeBERTa classifier
#     ZS  zero-shot mean acc of TruthfulQA-MC1, MMLU, HellaSwag, ARC-easy
#   -> summary_test.{md,csv,json} (with the training geometry/losses of every scenario).
# Stage 3: results (tables + records, no weights) -> private Hugging Face dataset repo HF_RESULTS_REPO.
#
# Finished trainings / evaluations are skipped and interrupted ones resume: just run the script again.
#
# Usage:
#   export HF_TOKEN=hf_xxx                       # Llama-2 access + WRITE access (results upload)
#   nohup bash scripts/run_ablation.sh > ablation.log 2>&1 &     # keeps running after logout
#   ONLY="ref beta0" bash scripts/run_ablation.sh
#   HF_UPLOAD=0 bash scripts/run_ablation.sh     # no upload
#
# Outputs (OUT_DIR = results/ablation):
#   summary_test.{md,csv,json}           the ablation table
#   runs/<scenario>/train/               config.yaml, train_log.jsonl, metrics.json, final/ (fp32 weights)
#   runs/<scenario>/attack-eval-test/    attack_eval.json, console.log
#   runs/base-llama2-7b-chat/attack-eval-test/   the undefended reference
#   scenarios.tsv, tests.log, failures.txt, hf_results_upload.json
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
BASE_CONFIG=${BASE_CONFIG:-config/cbg_v2_llama2_7b_2xh200_fast_hvp.yaml}
DATA_DIR=${DATA_DIR:-$ROOT/data/seam4k-v1}
OUT_DIR=${OUT_DIR:-$ROOT/results/ablation}
RUNS_DIR=${RUNS_DIR:-$OUT_DIR/runs}
ONLY=${ONLY:-}
RUN_TESTS=${RUN_TESTS:-1}
FINAL_LRS=${FINAL_LRS:-2e-5,5e-5,8e-5,1e-4,2e-4}            # SEAM Table 6
FINAL_TASKS=truthfulqa_mc1,mmlu,hellaswag,arc_easy        # ZS (SEAM §5.1); fixed
FINAL_GPUS=${FINAL_GPUS:-0,1}
HF_UPLOAD=${HF_UPLOAD:-1}
HF_RESULTS_REPO=${HF_RESULTS_REPO:-ducthang1703/cbg-hvp-ablation-results}   # private dataset repo
BASE_RUN=base-llama2-7b-chat

export PYTHONPATH=$ROOT USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
       WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8 \
       HF_DATASETS_TRUST_REMOTE_CODE=1

# name|overrides (dotted config paths; empty = the reference config itself)
SCENARIOS=(
  "ref|"                                      # reference: beta 0.01, alpha 1, k 4, HVP tf32, lazy 4 (reuse)
  "beta0|method.beta=0.0"                     # control: no geometry term (L_aln + alpha L_bgn only)
  "beta0.003|method.beta=0.003"
  "beta0.03|method.beta=0.03"
  "alpha3|method.alpha=3.0"                   # stronger benign KL (utility drift seen at beta 0.01)
  "alpha0.3|method.alpha=0.3"                 # weaker benign KL
  "k2|method.k=2"                             # spectral modes
  "k8|method.k=8"
)

log() { echo "[$(date '+%F %T')] $*"; }
fail() { log "FAILED: $*"; echo "$(date '+%F %T')  $*" >> "$OUT_DIR/failures.txt"; }

# ---------------------------------------------------------------------------
# Pre-flight (everything that can fail is checked before hours of GPU work)
# ---------------------------------------------------------------------------
mkdir -p "$RUNS_DIR" "$OUT_DIR"
# failures.txt lists this run's failures only; the previous run's list is kept next to it.
[[ -s "$OUT_DIR/failures.txt" ]] && mv "$OUT_DIR/failures.txt" "$OUT_DIR/failures.previous.txt"
gpus=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [[ "$gpus" -lt 2 ]]; then log "need 2 visible GPUs, found $gpus"; exit 1; fi
if [[ -z "${HF_TOKEN:-}" ]] && [[ ! -f "$HOME/.cache/huggingface/token" ]]; then
  log "set HF_TOKEN (Llama-2 access) or run 'huggingface-cli login'"; exit 1
fi
python - <<'EOF' || { log "missing Python packages: pip install -r requirements.txt"; exit 1; }
import accelerate, datasets, huggingface_hub, lm_eval, transformers, yaml   # noqa: F401
EOF
# HVP only: the base config computes the curvature term exactly (double backward), and no scenario overrides it.
python - "$BASE_CONFIG" <<'EOF' || { log "$BASE_CONFIG is not an HVP config (method.curvature must be hvp)"; exit 1; }
import sys, yaml
sys.exit(0 if yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["method"].get("curvature") == "hvp" else 1)
EOF
for entry in "${SCENARIOS[@]}"; do
  if [[ "$entry" == *"method.curvature="* ]]; then log "scenario '${entry%%|*}' changes the curvature; HVP only"; exit 1; fi
done
if [[ "$HF_UPLOAD" == 1 ]]; then
  # The repo id is validated and the token must be able to write, otherwise the upload at the end would fail.
  python -m src.experiments.hf_upload check "$HF_RESULTS_REPO" \
      || { log "Hugging Face upload check failed (repo id / write token); or set HF_UPLOAD=0"; exit 1; }
fi
IFS=, read -r -a final_gpus <<< "$FINAL_GPUS"
[[ ${#final_gpus[@]} -ge 1 && ${#final_gpus[@]} -le $gpus ]] || { log "FINAL_GPUS=$FINAL_GPUS does not match $gpus GPUs"; exit 1; }
free_gb=$(df -Pk "$OUT_DIR" | awk 'NR == 2 {print int($4 / 1048576)}')
if [[ "${free_gb:-0}" -lt 400 ]]; then
  log "WARNING: only ${free_gb} GB free under $OUT_DIR; 8 fp32 runs (27 GB each) + one 81 GB checkpoint need ~350 GB"
fi
log "GPUs: $gpus | config: $BASE_CONFIG | data: $DATA_DIR | out: $OUT_DIR"
log "eval: greedy HS + ZS ($FINAL_TASKS), before and after attacks at $FINAL_LRS"
log "HF upload: $([[ "$HF_UPLOAD" == 1 ]] && echo "results -> $HF_RESULTS_REPO (dataset, private)" || echo off)"

# ---------------------------------------------------------------------------
# Data + model (once), tests
# ---------------------------------------------------------------------------
python -m src.experiments.prepare_v2 --config "$BASE_CONFIG" --data "$DATA_DIR" || { log "prepare failed"; exit 1; }
MODEL_PATH=$(python -m src.experiments.prepare_v2 --data "$DATA_DIR" --model-path 2>/dev/null | tail -n 1)
if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH/config.json" ]]; then
  log "model not found locally (interrupted prepare?): delete $DATA_DIR and run again"; exit 1
fi
log "model: $MODEL_PATH"
if [[ "$RUN_TESTS" == 1 ]]; then
  (cd tests && CUDA_VISIBLE_DEVICES= python -m unittest test_cbg_spec test_cbg_method_v2 test_cbg_step_v2 test_cbg_train_v2 \
      test_attack_eval test_ablation test_hf_upload > "$OUT_DIR/tests.log" 2>&1) \
      || { log "unit tests failed, see $OUT_DIR/tests.log"; exit 1; }
  log "unit tests passed"
fi

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
trained() { [[ -f "$RUNS_DIR/$1/train/metrics.json" ]] && grep -q '"complete"' "$RUNS_DIR/$1/train/metrics.json"; }

train() {   # train <name> <overrides...>
  local name=$1; shift
  local out="$RUNS_DIR/$name/train" candidate="$RUNS_DIR/$name/config.candidate.yaml"
  mkdir -p "$out"
  python -m src.experiments.ablation override --base "$BASE_CONFIG" --out "$candidate" "$@" \
      || { fail "$name config ($*)"; return 1; }
  if [[ -f "$out/config.yaml" ]] && ! cmp -s "$candidate" "$out/config.yaml"; then
    fail "$name: $out was trained with a different config; delete it or rename the scenario"; return 2
  fi
  mv "$candidate" "$out/config.yaml"
  if trained "$name"; then log "[$name] already trained"; return 0; fi
  local resume=()
  [[ -f "$out/identity.json" ]] && resume=(--resume)
  log "[$name] training on 2 GPUs (${*:-reference config}) ${resume[*]}"
  python -m src.cbg.train_v2 --config "$out/config.yaml" --data "$DATA_DIR" --output "$out" --model "$MODEL_PATH" \
      --stage train "${resume[@]}" 2>&1 | tee -a "$out/console.log"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { fail "$name training (see $out/console.log)"; return 1; }
}

eval_done() {   # eval_done <attack_eval.json>: every attack lr present, with HS and ZS
  python - "$1" "$FINAL_LRS" <<'EOF'
import json, sys
from pathlib import Path
path, lrs = Path(sys.argv[1]), [f"{float(v):g}" for v in sys.argv[2].split(",") if v]
result = json.loads(path.read_text()) if path.exists() else {}
attacks = result.get("attacks", {})
sys.exit(0 if "post_harmfulness_mean" in result and all("zs" in attacks.get(lr, {}).get("utility", {}) for lr in lrs)
         else 1)
EOF
}

evaluate() {   # evaluate <name> <model dir> <gpu>
  local name=$1 model=$2 gpu=$3 out="$RUNS_DIR/$1/attack-eval-test"
  if eval_done "$out/attack_eval.json"; then log "[$name] eval already done"; return 0; fi
  mkdir -p "$out"
  log "[$name] eval on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python -m src.experiments.attack_eval --model "$model" --data "$DATA_DIR" --output "$out" \
      --mode test --lrs "$FINAL_LRS" --utility "$FINAL_TASKS" --post-utility --decoding greedy \
      >> "$out/console.log" 2>&1
  if [[ $? -ne 0 ]]; then fail "$name eval (see $out/console.log)"; return 1; fi
  log "[$name] eval done"
}

summary() {
  python -m src.experiments.ablation final --root "$RUNS_DIR" --base "$BASE_RUN" --out "$OUT_DIR" \
      --scenarios "$OUT_DIR/scenarios.tsv" > "$OUT_DIR/summary_test.log" 2>&1 \
      || log "summary failed (see $OUT_DIR/summary_test.log)"
}

# ---------------------------------------------------------------------------
# Stage 1: train, one scenario at a time
# ---------------------------------------------------------------------------
: > "$OUT_DIR/scenarios.tsv"
for entry in "${SCENARIOS[@]}"; do
  name=${entry%%|*}; overrides=${entry#*|}
  printf '%s\t%s\n' "$name" "$overrides" >> "$OUT_DIR/scenarios.tsv"
done
selected=()
stale=()
for entry in "${SCENARIOS[@]}"; do
  name=${entry%%|*}
  if [[ -z "$ONLY" ]] || [[ " $ONLY " == *" $name "* ]]; then selected+=("$entry"); fi
done
log "scenarios: $(printf '%s ' "${selected[@]%%|*}")"

for entry in "${selected[@]}"; do
  name=${entry%%|*}; read -r -a overrides <<< "${entry#*|}"
  log "===== train $name ====="
  train "$name" "${overrides[@]}"
  [[ $? -eq 2 ]] && stale+=("$name")   # its run belongs to another config: not evaluated
done

# ---------------------------------------------------------------------------
# Stage 2: HS + ZS before / after attacks, base + every trained scenario, one model per GPU in parallel
# ---------------------------------------------------------------------------
eval_jobs=("$BASE_RUN|$MODEL_PATH")
for entry in "${selected[@]}"; do
  name=${entry%%|*}
  [[ " ${stale[*]} " == *" $name "* ]] && continue
  trained "$name" && eval_jobs+=("$name|$RUNS_DIR/$name/train/final")
done
# Download the lm-eval datasets and the classifier once, so the parallel workers do not race on the caches.
python - "$FINAL_TASKS" <<'EOF' || fail "prefetch of lm-eval tasks / classifier"
import sys
from huggingface_hub import snapshot_download
from lm_eval.tasks import TaskManager, get_task_dict
from src.experiments.attack_eval import CLASSIFIER
get_task_dict(sys.argv[1].split(","), TaskManager())
snapshot_download(CLASSIFIER)
EOF
log "evaluating ${#eval_jobs[@]} models on GPUs $FINAL_GPUS"
worker() {   # worker <slot>: every job whose index % number of GPUs == slot, on that GPU
  local slot=$1 gpu=${final_gpus[$1]} i
  for ((i = slot; i < ${#eval_jobs[@]}; i += ${#final_gpus[@]})); do
    evaluate "${eval_jobs[$i]%%|*}" "${eval_jobs[$i]#*|}" "$gpu"
  done
}
pids=()
for ((slot = 0; slot < ${#final_gpus[@]}; slot++)); do worker "$slot" & pids+=($!); done
for pid in "${pids[@]}"; do wait "$pid"; done
summary
log "ablation table: $OUT_DIR/summary_test.md"
cat "$OUT_DIR/summary_test.md" 2>/dev/null

# ---------------------------------------------------------------------------
# Stage 3: results -> Hugging Face (private dataset repo; no weights)
# ---------------------------------------------------------------------------
if [[ "$HF_UPLOAD" == 1 ]]; then
  log "uploading results to $HF_RESULTS_REPO (dataset, private)"
  python -m src.experiments.hf_upload results --out "$OUT_DIR" --repo "$HF_RESULTS_REPO" \
      || fail "upload of results to $HF_RESULTS_REPO"
fi

[[ -s "$OUT_DIR/failures.txt" ]] && { log "some steps failed:"; cat "$OUT_DIR/failures.txt"; }
log "done"
exit 0
