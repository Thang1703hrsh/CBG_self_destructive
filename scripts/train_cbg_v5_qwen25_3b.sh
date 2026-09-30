#!/usr/bin/env bash
set -euo pipefail

# Cloud launcher for the single-GPU CBG V5 Qwen2.5-3B run.
#
# Examples:
#   MAX_STEPS=1 SAVE_STEPS=1 bash scripts/train_cbg_v5_qwen25_3b.sh
#   MAX_STEPS=50 bash scripts/train_cbg_v5_qwen25_3b.sh
#   WANDB_MODE=online RUN_NAME=qwen3b-seed42 bash scripts/train_cbg_v5_qwen25_3b.sh
#   RESUME_FROM_CHECKPOINT=/workspace/checkpoints/checkpoint-250 \
#     bash scripts/train_cbg_v5_qwen25_3b.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export USE_TF="${USE_TF:-0}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# CBG is GPU-bound.  Large CPU thread pools add contention and cloud cost.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-8}"

# Put caches on the instance's local NVMe when possible.  Override HF_HOME on
# providers whose persistent volume is mounted elsewhere.
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export WANDB_MODE="${WANDB_MODE:-offline}"

RUN_NAME="${RUN_NAME:-cbg-v5-qwen25-3b-main-seed42}"
OUTPUT_ROOT="${CBG_OUTPUT_ROOT:-$REPO_ROOT/ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SEED="${SEED:-42}"
GEOMETRY_SOLVER="${GEOMETRY_SOLVER:-dual}"
DUAL_COLUMN_BATCH_SIZE="${DUAL_COLUMN_BATCH_SIZE:-4}"
DUAL_REFINEMENT_STEPS="${DUAL_REFINEMENT_STEPS:-3}"

mkdir -p "$HF_HOME" "$OUTPUT_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  GPU_MEMORY_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  echo "GPU: $GPU_NAME (${GPU_MEMORY_MB} MiB)"
  if [[ "$GPU_MEMORY_MB" -lt 45000 ]]; then
    echo "WARNING: CBG V5 FP32 Qwen2.5-3B has not been qualified below 48 GB VRAM." >&2
  fi
else
  echo "ERROR: nvidia-smi was not found; this launcher requires one CUDA GPU." >&2
  exit 1
fi

if [[ ! -f data/beavertails_with_refusals_train.json ]]; then
  echo "ERROR: missing data/beavertails_with_refusals_train.json" >&2
  exit 1
fi

ARGS=(
  --method cbg
  --config config/cbg_v5_qwen25_3b_main.yaml
  --output_dir "$OUTPUT_DIR"
  --max_steps "$MAX_STEPS"
  --save_steps "$SAVE_STEPS"
  --seed "$SEED"
  --data_seed "$SEED"
  --wandb_run_name "$RUN_NAME"
  --geometry_solver "$GEOMETRY_SOLVER"
  --dual_column_batch_size "$DUAL_COLUMN_BATCH_SIZE"
  --dual_refinement_steps "$DUAL_REFINEMENT_STEPS"
)

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--cbg_resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

echo "Output: $OUTPUT_DIR"
echo "Steps: $MAX_STEPS; seed: $SEED; solver: $GEOMETRY_SOLVER; dual columns: $DUAL_COLUMN_BATCH_SIZE; refinements: $DUAL_REFINEMENT_STEPS; W&B mode: $WANDB_MODE"
python -m src.train "${ARGS[@]}"
