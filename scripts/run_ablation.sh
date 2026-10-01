#!/usr/bin/env bash
# CBG-HVP hyperparameter ablation on ONE server with 2x H200 (no Modal), scenarios run sequentially.
# HVP only: the base config must use method.curvature: hvp, and no scenario may change the curvature.
#
# For every scenario: train (2 GPUs, src/cbg/train_v2.py) -> SEAM-protocol attack evaluation on the
# VALIDATION pools (1 GPU, src/experiments/attack_eval.py --mode val) -> refreshed summary table.
# The undefended base model is evaluated once as the reference. Finished trainings / evaluations are
# skipped, interrupted trainings resume from their last checkpoint: just run the script again.
#
# Usage:
#   export HF_TOKEN=hf_xxx                       # access to meta-llama/Llama-2-7b-chat-hf
#   bash scripts/run_ablation.sh                 # all 8 scenarios
#   ONLY="ref beta0" bash scripts/run_ablation.sh
#   nohup bash scripts/run_ablation.sh > ablation.log 2>&1 &     # keeps running after logout
#
# Outputs: results/ablation/runs/<scenario>/{train,attack-eval-val}/ and results/ablation/summary.{md,csv,json}
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
BASE_CONFIG=${BASE_CONFIG:-config/cbg_v2_llama2_7b_2xh200_fast_hvp.yaml}
DATA_DIR=${DATA_DIR:-$ROOT/data/seam4k-v1}
RUNS_DIR=${RUNS_DIR:-$ROOT/results/ablation/runs}
OUT_DIR=${OUT_DIR:-$ROOT/results/ablation}
ONLY=${ONLY:-}
ATTACK_LRS=${ATTACK_LRS:-2e-5,5e-5,1e-4}
UTILITY_TASKS=${UTILITY_TASKS:-arc_easy}
RUN_TESTS=${RUN_TESTS:-1}
UPLOAD_BEST_REPO=${UPLOAD_BEST_REPO:-}        # e.g. ducthang1703/cbg-hvp-ablation-best (private)
BASE_RUN=base-llama2-7b-chat

export PYTHONPATH=$ROOT USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
       WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8

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
# Pre-flight
# ---------------------------------------------------------------------------
mkdir -p "$RUNS_DIR" "$OUT_DIR"
gpus=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [[ "$gpus" -lt 2 ]]; then log "need 2 visible GPUs, found $gpus"; exit 1; fi
if [[ -z "${HF_TOKEN:-}" ]] && [[ ! -f "$HOME/.cache/huggingface/token" ]]; then
  log "set HF_TOKEN (Llama-2 access) or run 'huggingface-cli login'"; exit 1
fi
python - <<'EOF' || { log "missing Python packages: pip install -r requirements.txt"; exit 1; }
import accelerate, datasets, lm_eval, transformers, yaml   # noqa: F401
EOF
# HVP only: the base config computes the curvature term exactly (double backward), and no scenario overrides it.
python - "$BASE_CONFIG" <<'EOF' || { log "$BASE_CONFIG is not an HVP config (method.curvature must be hvp)"; exit 1; }
import sys, yaml
sys.exit(0 if yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["method"].get("curvature") == "hvp" else 1)
EOF
for entry in "${SCENARIOS[@]}"; do
  if [[ "$entry" == *"method.curvature="* ]]; then log "scenario '${entry%%|*}' changes the curvature; HVP only"; exit 1; fi
done
log "GPUs: $gpus | config: $BASE_CONFIG | data: $DATA_DIR"

# ---------------------------------------------------------------------------
# Data + model (once), tests
# ---------------------------------------------------------------------------
python -m src.experiments.prepare_v2 --config "$BASE_CONFIG" --data "$DATA_DIR" || { log "prepare failed"; exit 1; }
MODEL_PATH=$(python -m src.experiments.prepare_v2 --data "$DATA_DIR" --model-path | tail -n 1)
log "model: $MODEL_PATH"
if [[ "$RUN_TESTS" == 1 ]]; then
  (cd tests && python -m unittest test_cbg_spec test_cbg_method_v2 test_cbg_step_v2 test_cbg_train_v2 \
      test_attack_eval > "$OUT_DIR/tests.log" 2>&1) || { log "unit tests failed, see $OUT_DIR/tests.log"; exit 1; }
  log "unit tests passed"
fi

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
evaluate() {   # evaluate <name> <model dir>
  local name=$1 model=$2 out="$RUNS_DIR/$1/attack-eval-val"
  if python - "$out/attack_eval.json" "$ATTACK_LRS" <<'EOF'
import json, sys
from pathlib import Path
path, lrs = Path(sys.argv[1]), [f"{float(v):g}" for v in sys.argv[2].split(",") if v]
done = path.exists() and json.loads(path.read_text()).get("attacks", {})
sys.exit(0 if done and all(lr in done for lr in lrs) and "post_harmfulness_mean" in json.loads(path.read_text()) else 1)
EOF
  then log "[$name] attack-eval already done"; return 0; fi
  mkdir -p "$out"
  log "[$name] attack-eval (lrs $ATTACK_LRS, 1 GPU)"
  CUDA_VISIBLE_DEVICES=0 python -m src.experiments.attack_eval --model "$model" --data "$DATA_DIR" --output "$out" \
      --mode val --lrs "$ATTACK_LRS" --utility "$UTILITY_TASKS" 2>&1 | tee -a "$out/console.log"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { fail "$name attack-eval (see $out/console.log)"; return 1; }
}

train() {   # train <name> <overrides...>
  local name=$1; shift
  local out="$RUNS_DIR/$name/train" candidate="$RUNS_DIR/$name/config.candidate.yaml"
  mkdir -p "$out"
  python -m src.experiments.ablation override --base "$BASE_CONFIG" --out "$candidate" "$@" \
      || { fail "$name config ($*)"; return 1; }
  if [[ -f "$out/config.yaml" ]] && ! cmp -s "$candidate" "$out/config.yaml"; then
    fail "$name: $out was trained with a different config; delete it or rename the scenario"; return 1
  fi
  mv "$candidate" "$out/config.yaml"
  if [[ -f "$out/metrics.json" ]] && grep -q '"complete"' "$out/metrics.json"; then
    log "[$name] already trained"; return 0
  fi
  local resume=()
  [[ -f "$out/identity.json" ]] && resume=(--resume)
  log "[$name] training on 2 GPUs (${*:-reference config}) ${resume[*]}"
  python -m src.cbg.train_v2 --config "$out/config.yaml" --data "$DATA_DIR" --output "$out" --model "$MODEL_PATH" \
      --stage train "${resume[@]}" 2>&1 | tee -a "$out/console.log"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { fail "$name training (see $out/console.log)"; return 1; }
}

summary() {
  python -m src.experiments.ablation summary --root "$RUNS_DIR" --base "$BASE_RUN" --out "$OUT_DIR" \
      --scenarios "$OUT_DIR/scenarios.tsv" --utility-task "${UTILITY_TASKS%%,*}" > "$OUT_DIR/summary.log" 2>&1 \
      || log "summary failed (see $OUT_DIR/summary.log)"
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
: > "$OUT_DIR/scenarios.tsv"
for entry in "${SCENARIOS[@]}"; do
  name=${entry%%|*}; overrides=${entry#*|}
  printf '%s\t%s\n' "$name" "$overrides" >> "$OUT_DIR/scenarios.tsv"
done
selected=()
for entry in "${SCENARIOS[@]}"; do
  name=${entry%%|*}
  if [[ -z "$ONLY" ]] || [[ " $ONLY " == *" $name "* ]]; then selected+=("$entry"); fi
done
log "scenarios: $(printf '%s ' "${selected[@]%%|*}")"

evaluate "$BASE_RUN" "$MODEL_PATH"
for entry in "${selected[@]}"; do
  name=${entry%%|*}; read -r -a overrides <<< "${entry#*|}"
  log "===== $name ====="
  if train "$name" "${overrides[@]}"; then
    evaluate "$name" "$RUNS_DIR/$name/train/final"
  fi
  summary
done

summary
log "summary: $OUT_DIR/summary.md"
cat "$OUT_DIR/summary.md" 2>/dev/null
best=$(python -c "import json; print(json.load(open('$OUT_DIR/summary.json')).get('best') or '')" 2>/dev/null)
log "best on the validation rule: ${best:-n/a}"

if [[ -n "$UPLOAD_BEST_REPO" && -n "$best" ]]; then
  log "uploading $best to $UPLOAD_BEST_REPO (private)"
  python - "$RUNS_DIR/$best/train" "$UPLOAD_BEST_REPO" "$DATA_DIR" <<'EOF' || fail "upload of $best"
import json, sys
from pathlib import Path
from src.experiments.hf_upload import upload_run
pinned = json.loads((Path(sys.argv[3]) / "model_revision.json").read_text())
print(upload_run(sys.argv[1], sys.argv[2], base_model=pinned["model"], base_revision=pinned["revision"]))
EOF
fi
[[ -s "$OUT_DIR/failures.txt" ]] && { log "some steps failed:"; cat "$OUT_DIR/failures.txt"; }
exit 0
