#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

GATE="${RSMOL_5_10XPOISSON_PARCAE_STAGE4_GATE:-D}"
GATE="${GATE^^}"
MODEL_PATH="${RSMOL_5_10XPOISSON_PARCAE_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10xpoisson-parcae}"
TOKENIZER_PATH="${RSMOL_5_10XPOISSON_PARCAE_TOKENIZER_PATH:-}"
DATA_DIR="${RSMOL_5_10XPOISSON_PARCAE_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset}"
OUTPUT_DIR="${RSMOL_5_10XPOISSON_PARCAE_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10xpoisson_parcae/$(date +%Y%m%d_%H%M%S)}"
WORLD_SIZE="${RSMOL_5_10XPOISSON_PARCAE_WORLD_SIZE:-8}"
if [[ "$GATE" == "D" || "$GATE" == "FORMAL" ]]; then
  if [[ "$WORLD_SIZE" != "8" ]]; then
    echo "5_10xpoisson_parcae $GATE requires exactly 8 ranks; got WORLD_SIZE=$WORLD_SIZE" >&2
    exit 2
  fi
fi

if [[ "$GATE" == "FORMAL" ]]; then
  MICRO_BATCH_SIZE="${RSMOL_5_10XPOISSON_PARCAE_MICRO_BATCH_SIZE:-2}"
  GRADIENT_ACCUMULATION_STEPS="${RSMOL_5_10XPOISSON_PARCAE_GRADIENT_ACCUMULATION_STEPS:-64}"
  MAX_LR="${RSMOL_5_10XPOISSON_PARCAE_MAX_LR:-8e-4}"
  MIN_LR="${RSMOL_5_10XPOISSON_PARCAE_MIN_LR:-8e-5}"
else
  MICRO_BATCH_SIZE="${RSMOL_5_10XPOISSON_PARCAE_MICRO_BATCH_SIZE:-8}"
  GRADIENT_ACCUMULATION_STEPS="${RSMOL_5_10XPOISSON_PARCAE_GRADIENT_ACCUMULATION_STEPS:-16}"
  MAX_LR="${RSMOL_5_10XPOISSON_PARCAE_MAX_LR:-2e-4}"
  MIN_LR="${RSMOL_5_10XPOISSON_PARCAE_MIN_LR:-2e-5}"
fi

if [[ "$GATE" == "FORMAL" ]]; then
  MAX_STEPS=9244
  WARMUP_STEPS=463
  SCHEDULER_STEPS=9244
else
  MAX_STEPS="${RSMOL_5_10XPOISSON_PARCAE_MAX_OPTIMIZER_STEPS:-10}"
  WARMUP_STEPS="${RSMOL_5_10XPOISSON_PARCAE_WARMUP_STEPS:-1}"
  SCHEDULER_STEPS="${RSMOL_5_10XPOISSON_PARCAE_SCHEDULER_TOTAL_STEPS:-$MAX_STEPS}"
fi

ARGS=(--gate "$GATE" --model-path "$MODEL_PATH" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR"
  --world-size "$WORLD_SIZE" --backend "${RSMOL_5_10XPOISSON_PARCAE_BACKEND:-nccl}"
  --micro-batch-size "$MICRO_BATCH_SIZE" --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" --context-length 1024
  --learning-rate "$MAX_LR" --max-lr "$MAX_LR" --min-lr "$MIN_LR"
  --max-optimizer-steps "$MAX_STEPS" --scheduler-total-steps "$SCHEDULER_STEPS"
  --warmup-steps "$WARMUP_STEPS" --save-every "${RSMOL_5_10XPOISSON_PARCAE_SAVE_EVERY:-500}"
  --checkpoint-retention 3 --seed "${RSMOL_5_10XPOISSON_PARCAE_SEED:-0}")
[[ -n "$TOKENIZER_PATH" ]] && ARGS+=(--tokenizer-path "$TOKENIZER_PATH")
[[ -n "${RSMOL_5_10XPOISSON_PARCAE_RESUME_FROM:-}" ]] && ARGS+=(--resume-from "$RSMOL_5_10XPOISSON_PARCAE_RESUME_FROM")
[[ "${RSMOL_5_10XPOISSON_PARCAE_DRY_RUN:-0}" == "1" ]] && ARGS+=(--dry-run)

echo "========== STAGE 4 5-10xpoisson-parcae $GATE =========="
echo "MAX_LR=$MAX_LR MIN_LR=$MIN_LR"
torchrun --standalone --nproc_per_node="$WORLD_SIZE" code/RSmol/scripts/train_stage4_5_10xpoisson_parcae_ddp.py "${ARGS[@]}"
echo "[result] status=PASS"
