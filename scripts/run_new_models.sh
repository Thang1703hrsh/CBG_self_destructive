#!/usr/bin/env bash
# CBG final (the C2 recipe of config/cbg_v2_llama2_7b_2xh200_cbg_final.yaml) on new base models, on ONE server
# with 2x H200 (no Modal). Each model goes through scripts/run_ablation.sh with only the reference scenario:
#   train  CBG, 500 steps, 2 GPUs (utility gates at steps 100/250 may stop it early: then it is not evaluated)
#   eval   the base model and the CBG model, one per H200 in parallel, SEAM protocol (src/experiments/attack_eval.py):
#          clean (pre-attack) + harmful fine-tuning attacks at FINAL_LRS (default 2e-5, 5e-5, 1e-4),
#          HS = greedy generation on 100 BeaverTails test prompts, DeBERTa classifier;
#          ZS = zero-shot mean acc of TruthfulQA-MC1, MMLU, HellaSwag, ARC-easy (before and after each attack)
# Models run one after another (training needs both GPUs). Finished steps are skipped and interrupted ones
# resume: just run the script again.
#
# Usage:
#   export HF_TOKEN=hf_xxx            # must have accepted mistralai/Mistral-7B-Instruct-v0.3 (gated)
#   nohup bash scripts/run_new_models.sh > new_models.log 2>&1 &      # keeps running after logout
#   ONLY_MODELS=qwen25-7b bash scripts/run_new_models.sh               # one model
#   HF_UPLOAD=0 bash scripts/run_new_models.sh   # no Hugging Face upload
# Hugging Face (after each model's eval, PUBLIC repos): CBG weights (bf16) -> ducthang1703/cbg-<model>-ref,
#   tables + records of the base and CBG evals -> ducthang1703/cbg-<model>-results (dataset repo).
#   The token needs WRITE access (checked before any GPU work). HF_PUBLIC=0: private repos instead (the
#   account's private storage is nearly full: ~5 GB of 100 GB free, a model needs ~15 GB).
#
# Outputs: results/new-models/<model>/summary_test.{md,csv,json}
#          results/new-models/<model>/runs/ref/train/            CBG training (final/ = fp32 weights)
#          results/new-models/<model>/runs/<base run>/attack-eval-test/   the undefended reference
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
ONLY_MODELS=${ONLY_MODELS:-}
FINAL_LRS=${FINAL_LRS:-2e-5,5e-5,1e-4}
HF_UPLOAD=${HF_UPLOAD:-1}
HF_PUBLIC=${HF_PUBLIC:-1}
RESULTS_ROOT=${RESULTS_ROOT:-$ROOT/results/new-models}

# slug|config|base run name|HF model id
MODELS=(
  "qwen25-7b|config/cbg_v2_qwen25_7b_2xh200_cbg_final.yaml|base-qwen2.5-7b-instruct|Qwen/Qwen2.5-7B-Instruct"
  "mistral-7b|config/cbg_v2_mistral_7b_2xh200_cbg_final.yaml|base-mistral-7b-instruct-v0.3|mistralai/Mistral-7B-Instruct-v0.3"
)

log() { echo "[$(date '+%F %T')] $*"; }

selected=()
for entry in "${MODELS[@]}"; do
  slug=${entry%%|*}
  if [[ -z "$ONLY_MODELS" ]] || [[ " $ONLY_MODELS " == *" $slug "* ]]; then selected+=("$entry"); fi
done
[[ ${#selected[@]} -gt 0 ]] || { log "ONLY_MODELS=$ONLY_MODELS matches none of: ${MODELS[*]%%|*}"; exit 1; }

# Access to every (gated) model is checked before hours of GPU work on the first one.
python - "${selected[@]##*|}" <<'EOF' || { log "no access to a model above: accept its terms on Hugging Face"; exit 1; }
import sys
from huggingface_hub import hf_hub_download
failed = False
for model in sys.argv[1:]:
    try:
        hf_hub_download(model, "config.json")
        print(f"access ok: {model}")
    except Exception as error:   # GatedRepoError / 401 / 404
        print(f"NO ACCESS: {model}: {type(error).__name__}: {error}")
        failed = True
sys.exit(1 if failed else 0)
EOF

first=1
for entry in "${selected[@]}"; do
  IFS='|' read -r slug config base_run model <<< "$entry"
  out="$RESULTS_ROOT/$slug"
  log "========== $slug ($model) =========="
  # run_ablation.sh: data + pinned model in DATA_DIR (one per model: tokenized lengths and the model differ).
  BASE_CONFIG=$config DATA_DIR=$ROOT/data/seam4k-$slug OUT_DIR=$out ONLY=ref FINAL_LRS=$FINAL_LRS \
  BASE_RUN=$base_run RUN_TESTS=$first HF_UPLOAD=$HF_UPLOAD HF_PUBLIC=$HF_PUBLIC \
  HF_RESULTS_REPO=ducthang1703/cbg-$slug-results HF_MODEL_PREFIX=ducthang1703/cbg-$slug \
      bash scripts/run_ablation.sh
  status=$?
  first=0
  metrics="$out/runs/ref/train/metrics.json"
  if [[ $status -ne 0 ]]; then
    log "[$slug] run_ablation.sh failed before training/eval (exit $status), see above"
  elif [[ -f "$metrics" ]] && grep -q '"early_stopped"' "$metrics"; then
    log "[$slug] CBG training EARLY-STOPPED by a utility gate (only the base model was evaluated): $(cat "$metrics")"
  fi
done

log "summaries:"
for entry in "${selected[@]}"; do
  slug=${entry%%|*}
  echo "## $slug"
  cat "$RESULTS_ROOT/$slug/summary_test.md" 2>/dev/null || echo "(no summary)"
done
log "done"
