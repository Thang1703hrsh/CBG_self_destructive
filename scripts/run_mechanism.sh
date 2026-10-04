#!/usr/bin/env bash
# Mechanism experiments for CBG on Llama-2-7B-Chat, on ONE server with 2x H200 (no Modal):
#
#   1. Ablation beta = 0: train config/cbg_v2_llama2_7b_2xh200_beta0_final.yaml (L_aln + alpha L_bgn only, otherwise
#      identical to CBG final), then attack Base, CBG and CBG-beta0 with LoRA (q,v) at 1e-4 and with 10% poisoned data
#      (mixed10) at 5e-5, and measure clean + post-attack HS / ZS. Is CBG's vulnerability caused by the geometry term?
#   2. Attack trajectory: the attacks of step 1 save snapshots at steps 5, 10, 25, 50 (and 250). For every snapshot,
#      src/experiments/attack_trajectory.py measures the held-out log eta_k and the cosine between the attack
#      displacement and the top safety direction of the starting model.
#
# The three models are attacked under one protocol on this server, so the ablation table compares like with like.
# Finished steps are skipped and interrupted ones resume: just run the script again.
#
# Usage:
#   export HF_TOKEN=hf_xxx                 # Llama-2 access (+ write access if HF_UPLOAD=1)
#   nohup bash scripts/run_mechanism.sh > mechanism.log 2>&1 &      # keeps running after logout
#   ONLY_MODELS="base cbg" bash scripts/run_mechanism.sh            # subset (beta0 is then not trained)
#   HF_UPLOAD=0 bash scripts/run_mechanism.sh                       # no upload at all
#   HF_PUBLIC=0 bash scripts/run_mechanism.sh                       # private repos instead of public
#
# Hugging Face (public by default), at the end:
#   ducthang1703/cbg-llama2-7b-beta0 (model): the beta = 0 weights (bf16) + training records
#   ducthang1703/cbg-mechanism-results (dataset): tables, per-step training and attack losses, every log
#     (run.log = the whole console, tests, training and evaluation consoles, trajectory logs)
#   KEEP_SNAPSHOTS=1 bash scripts/run_mechanism.sh                  # keep the attack snapshots (~70 GB per model)
#
# Outputs (OUT_DIR = results/mechanism):
#   summary_mechanism.{md,json}                       the two tables
#   run.log                                           the whole console of the run
#   runs/beta0/train/                                 ablation training (final/ = fp32 weights)
#   runs/<model>/attack-eval-{lora-lr1e-4,mixed10-lr5e-5}/attack_eval.json, console.log
#   runs/<model>/trajectory.json, trajectory.log      geometry + cosine at every snapshot
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
CBG_CONFIG=${CBG_CONFIG:-config/cbg_v2_llama2_7b_2xh200_cbg_final.yaml}     # data recipe + geometry settings
BETA0_CONFIG=${BETA0_CONFIG:-config/cbg_v2_llama2_7b_2xh200_beta0_final.yaml}
CBG_REPO=${CBG_REPO:-ducthang1703/cbg-llama2-7b-step500}                     # the evaluated CBG model
DATA_DIR=${DATA_DIR:-$ROOT/data/seam4k-cbgfinal}
OUT_DIR=${OUT_DIR:-$ROOT/results/mechanism}
RUNS_DIR=$OUT_DIR/runs
ONLY_MODELS=${ONLY_MODELS:-base cbg beta0}
SAVE_STEPS=${SAVE_STEPS:-5,10,25,50}
TASKS=truthfulqa_mc1,mmlu,hellaswag,arc_easy
KEEP_SNAPSHOTS=${KEEP_SNAPSHOTS:-0}
RUN_TESTS=${RUN_TESTS:-1}
HF_UPLOAD=${HF_UPLOAD:-1}                                                    # results, losses and logs
HF_PUBLIC=${HF_PUBLIC:-1}                                                    # 1 = public repos; 0 = private
HF_RESULTS_REPO=${HF_RESULTS_REPO:-ducthang1703/cbg-mechanism-results}      # dataset repo
UPLOAD_MODELS=${UPLOAD_MODELS:-1}                                            # 1 = also the beta = 0 weights (bf16)
HF_BETA0_REPO=${HF_BETA0_REPO:-ducthang1703/cbg-llama2-7b-beta0}            # model repo

export PYTHONPATH=$ROOT USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
       WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8 \
       HF_DATASETS_TRUST_REMOTE_CODE=1

log() { echo "[$(date '+%F %T')] $*"; }
fail() { log "FAILED: $*"; echo "$(date '+%F %T')  $*" >> "$OUT_DIR/failures.txt"; }
selected() { [[ " $ONLY_MODELS " == *" $1 "* ]]; }

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
mkdir -p "$RUNS_DIR"
# The whole console of this run also goes to OUT_DIR/run.log (uploaded with the results).
exec > >(tee -a "$OUT_DIR/run.log") 2>&1
[[ -s "$OUT_DIR/failures.txt" ]] && mv "$OUT_DIR/failures.txt" "$OUT_DIR/failures.previous.txt"
gpus=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [[ "$gpus" -lt 2 ]]; then log "need 2 visible GPUs, found $gpus"; exit 1; fi
if [[ -z "${HF_TOKEN:-}" ]] && [[ ! -f "$HOME/.cache/huggingface/token" ]]; then
  log "set HF_TOKEN (Llama-2 access) or run 'huggingface-cli login'"; exit 1
fi
python - <<'EOF' || { log "missing Python packages: pip install -r requirements.txt"; exit 1; }
import accelerate, datasets, huggingface_hub, lm_eval, peft, safetensors, transformers, yaml   # noqa: F401
EOF
hf_public=(); visibility=private
if [[ "$HF_PUBLIC" == 1 ]]; then hf_public=(--public); visibility=public; fi
if [[ "$HF_UPLOAD" == 1 ]]; then
  python -m src.experiments.hf_upload check "$HF_RESULTS_REPO" "$HF_BETA0_REPO" \
      || { log "Hugging Face upload check failed (repo id / write token); or set HF_UPLOAD=0"; exit 1; }
  log "HF upload ($visibility): results, losses and logs -> $HF_RESULTS_REPO (dataset)$(
      [[ "$UPLOAD_MODELS" == 1 ]] && echo ", beta = 0 weights (bf16) -> $HF_BETA0_REPO")"
fi
free_gb=$(df -Pk "$OUT_DIR" | awk 'NR == 2 {print int($4 / 1048576)}')
if [[ "${free_gb:-0}" -lt 200 ]]; then
  log "WARNING: only ${free_gb} GB free; the mixed10 snapshots need ~70 GB per model (deleted after analysis)"
fi
log "models: $ONLY_MODELS | data: $DATA_DIR | out: $OUT_DIR | snapshots at steps $SAVE_STEPS"

python -m src.experiments.prepare_v2 --config "$CBG_CONFIG" --data "$DATA_DIR" || { log "prepare failed"; exit 1; }
BASE_PATH=$(python -m src.experiments.prepare_v2 --data "$DATA_DIR" --model-path 2>/dev/null | tail -n 1)
[[ -n "$BASE_PATH" && -f "$BASE_PATH/config.json" ]] || { log "base model not found locally: delete $DATA_DIR and rerun"; exit 1; }
log "base model: $BASE_PATH"
if [[ "$RUN_TESTS" == 1 ]]; then
  (cd tests && CUDA_VISIBLE_DEVICES= python -m unittest test_attack_eval test_attack_trajectory > "$OUT_DIR/tests.log" 2>&1) \
      || { log "unit tests failed, see $OUT_DIR/tests.log"; exit 1; }
  log "unit tests passed"
fi

# ---------------------------------------------------------------------------
# Step 1a: train the beta = 0 ablation (both GPUs)
# ---------------------------------------------------------------------------
declare -A MODEL_PATH=([base]="$BASE_PATH")
if selected beta0; then
  out="$RUNS_DIR/beta0/train"
  mkdir -p "$out"
  if [[ -f "$out/config.yaml" ]] && ! cmp -s "$BETA0_CONFIG" "$out/config.yaml"; then
    fail "beta0: $out was trained with a different config; delete it to retrain"
  else
    cp "$BETA0_CONFIG" "$out/config.yaml"
    if [[ -f "$out/metrics.json" ]] && grep -q '"complete"' "$out/metrics.json"; then
      log "[beta0] already trained"
    elif [[ -f "$out/metrics.json" ]] && grep -q '"early_stopped"' "$out/metrics.json"; then
      log "[beta0] was stopped by a utility gate: $(cat "$out/metrics.json")"
    else
      resume=(); [[ -f "$out/identity.json" ]] && resume=(--resume)
      log "[beta0] training on 2 GPUs ${resume[*]}"
      python -m src.cbg.train_v2 --config "$out/config.yaml" --data "$DATA_DIR" --output "$out" --model "$BASE_PATH" \
          --stage train "${resume[@]}" 2>&1 | tee -a "$out/console.log"
      [[ ${PIPESTATUS[0]} -eq 0 ]] || fail "beta0 training (see $out/console.log)"
    fi
    if [[ -f "$out/metrics.json" ]] && grep -q '"complete"' "$out/metrics.json"; then
      MODEL_PATH[beta0]="$out/final"
    else
      log "[beta0] no final model: it is left out of the attacks"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Step 1b: the CBG model under test (public HF repo, bf16)
# ---------------------------------------------------------------------------
if selected cbg; then
  cbg_path=$(python - "$CBG_REPO" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "tokenizer.model"]))
EOF
)
  if [[ -f "$cbg_path/config.json" ]]; then MODEL_PATH[cbg]="$cbg_path"; log "CBG model: $cbg_path"
  else fail "download of $CBG_REPO"; fi
fi

# ---------------------------------------------------------------------------
# Step 1c: attacks with snapshots, LoRA on GPU 0 and mixed10 on GPU 1 in parallel
# ---------------------------------------------------------------------------
attack() {   # attack <model> <path> <gpu> <folder> <extra args...>
  local model=$1 path=$2 gpu=$3 folder=$4; shift 4
  local out="$RUNS_DIR/$model/$folder"
  mkdir -p "$out"
  if [[ -f "$out/attack_eval.json" ]] && python - "$out/attack_eval.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
sys.exit(0 if r.get("attacks") and all("utility" in a for a in r["attacks"].values()) else 1)
EOF
  then log "[$model] $folder already done"; return 0; fi
  log "[$model] $folder on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python -m src.experiments.attack_eval --model "$path" --data "$DATA_DIR" --output "$out" \
      --mode test --utility "$TASKS" --post-utility --decoding checkpoint --save-at-steps "$SAVE_STEPS" "$@" \
      >> "$out/console.log" 2>&1 || { fail "$model $folder (see $out/console.log)"; return 1; }
  log "[$model] $folder done"
}

# Download the lm-eval datasets and the classifier once, so the parallel workers do not race on the caches.
python - "$TASKS" <<'EOF' || fail "prefetch of lm-eval tasks / classifier"
import sys
from huggingface_hub import snapshot_download
from lm_eval.tasks import TaskManager, get_task_dict
from src.experiments.attack_eval import CLASSIFIER
get_task_dict(sys.argv[1].split(","), TaskManager())
snapshot_download(CLASSIFIER)
EOF

for model in base cbg beta0; do
  selected "$model" || continue
  path=${MODEL_PATH[$model]:-}
  [[ -n "$path" ]] || continue
  traj="$RUNS_DIR/$model/trajectory.json"
  if [[ -f "$traj" ]]; then log "[$model] trajectory already analyzed"; continue; fi
  # The LoRA run also measures the clean model (pre); the mixed10 run skips it.
  attack "$model" "$path" 0 attack-eval-lora-lr1e-4 --lrs 1e-4 --lora &
  p0=$!
  attack "$model" "$path" 1 attack-eval-mixed10-lr5e-5 --lrs 5e-5 --attack mixed10 --skip-pre &
  p1=$!
  wait $p0; s0=$?; wait $p1; s1=$?
  [[ $s0 -eq 0 && $s1 -eq 0 ]] || { log "[$model] an attack failed; trajectory skipped"; continue; }

  # -------------------------------------------------------------------------
  # Step 2: trajectory diagnostics (both GPUs)
  # -------------------------------------------------------------------------
  log "[$model] trajectory analysis (geometry + cosine at every snapshot)"
  python -m src.experiments.attack_trajectory --config "$CBG_CONFIG" --data "$DATA_DIR" --model "$path" \
      --attack "lora=$RUNS_DIR/$model/attack-eval-lora-lr1e-4/attacked-0.0001" \
      --attack "mixed10=$RUNS_DIR/$model/attack-eval-mixed10-lr5e-5/attacked-5e-05" \
      --output "$traj" > "$RUNS_DIR/$model/trajectory.log" 2>&1
  if [[ $? -ne 0 || ! -f "$traj" ]]; then
    fail "$model trajectory (see $RUNS_DIR/$model/trajectory.log)"
  elif [[ "$KEEP_SNAPSHOTS" != 1 ]]; then
    rm -rf "$RUNS_DIR/$model"/attack-eval-*/attacked-* "$RUNS_DIR/$model"/attack-eval-*/trainer-*
    log "[$model] trajectory done; snapshots removed"
  fi
done

# ---------------------------------------------------------------------------
# Tables + upload
# ---------------------------------------------------------------------------
python -m src.experiments.mechanism_summary --root "$RUNS_DIR" --out "$OUT_DIR" > "$OUT_DIR/summary_mechanism.log" 2>&1 \
    || log "summary failed (see $OUT_DIR/summary_mechanism.log)"
cat "$OUT_DIR/summary_mechanism.md" 2>/dev/null
if [[ "$HF_UPLOAD" == 1 ]]; then
  record="$RUNS_DIR/beta0/train/hf_upload.json"
  if [[ "$UPLOAD_MODELS" == 1 && -n "${MODEL_PATH[beta0]:-}" ]]; then
    if [[ -f "$record" ]] && grep -q "\"repo_id\": \"$HF_BETA0_REPO\"" "$record"; then
      log "[beta0] weights already on $HF_BETA0_REPO"
    else
      log "[beta0] uploading weights (bf16) to $HF_BETA0_REPO ($visibility)"
      python -m src.experiments.hf_upload model --run "$RUNS_DIR/beta0/train" --repo "$HF_BETA0_REPO" \
          --dtype bfloat16 --data "$DATA_DIR" --staging "$OUT_DIR/.hf-staging" "${hf_public[@]}" \
          || fail "upload of the beta = 0 weights to $HF_BETA0_REPO"
      rm -rf "$OUT_DIR/.hf-staging"
    fi
  fi
  # Results last, so that they include the model upload record and the log up to this point.
  log "uploading results, losses and logs to $HF_RESULTS_REPO (dataset, $visibility)"
  python -m src.experiments.hf_upload results --out "$OUT_DIR" --repo "$HF_RESULTS_REPO" "${hf_public[@]}" \
      || fail "upload of results to $HF_RESULTS_REPO"
fi
[[ -s "$OUT_DIR/failures.txt" ]] && { log "some steps failed:"; cat "$OUT_DIR/failures.txt"; }
log "done"
exit 0
