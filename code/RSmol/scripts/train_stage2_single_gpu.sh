#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

: "${RSMOL_RECURSIVE_OUTPUT_DIR:?Set RSMOL_RECURSIVE_OUTPUT_DIR to the converted recursive checkpoint}"

DATA_DIR="${RSMOL_STAGE2_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset}"
OUTPUT_DIR="${RSMOL_STAGE2_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage2_single_gpu/$(date +%Y%m%d_%H%M%S)}"
REPORT_PATH="${RSMOL_STAGE2_REPORT:-$OUTPUT_DIR/stage2_training_report.json}"

echo "========== STAGE 2 SINGLE-GPU TRAINING VALIDATION =========="
echo "ACTIVE_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "PYTHON=$(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH=$RSMOL_RECURSIVE_OUTPUT_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "REPORT_PATH=$REPORT_PATH"
echo "MICRO_BATCH_SIZE=${RSMOL_STAGE2_MICRO_BATCH_SIZE:-8}"
echo "GRADIENT_ACCUMULATION_STEPS=${RSMOL_STAGE2_GRADIENT_ACCUMULATION_STEPS:-16}"
echo "LEARNING_RATE=${RSMOL_STAGE2_LEARNING_RATE:-2e-4}"
echo "CONTEXT_LENGTH=${RSMOL_STAGE2_CONTEXT_LENGTH:-1024}"
echo "WARMUP_STEPS=${RSMOL_STAGE2_WARMUP_STEPS:-2}"
echo "MAX_OPTIMIZER_STEPS=${RSMOL_STAGE2_MAX_OPTIMIZER_STEPS:-10}"

ARGS=(
  --model-path "$RSMOL_RECURSIVE_OUTPUT_DIR"
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --report-path "$REPORT_PATH"
  --micro-batch-size "${RSMOL_STAGE2_MICRO_BATCH_SIZE:-8}"
  --gradient-accumulation-steps "${RSMOL_STAGE2_GRADIENT_ACCUMULATION_STEPS:-16}"
  --learning-rate "${RSMOL_STAGE2_LEARNING_RATE:-2e-4}"
  --context-length "${RSMOL_STAGE2_CONTEXT_LENGTH:-1024}"
  --warmup-steps "${RSMOL_STAGE2_WARMUP_STEPS:-2}"
  --max-optimizer-steps "${RSMOL_STAGE2_MAX_OPTIMIZER_STEPS:-10}"
  --seed "${RSMOL_STAGE2_SEED:-0}"
  --weight-decay "${RSMOL_STAGE2_WEIGHT_DECAY:-0.1}"
  --max-grad-norm "${RSMOL_STAGE2_MAX_GRAD_NORM:-1.0}"
  --record-buffer-size "${RSMOL_STAGE2_RECORD_BUFFER_SIZE:-4096}"
  --save-every "${RSMOL_STAGE2_SAVE_EVERY:-10}"
  --device "${RSMOL_STAGE2_DEVICE:-cuda:0}"
)
if [[ -n "${RSMOL_STAGE2_TOKENIZER_PATH:-}" ]]; then
  ARGS+=(--tokenizer-path "$RSMOL_STAGE2_TOKENIZER_PATH")
fi
if [[ -n "${RSMOL_STAGE2_RESUME_FROM:-}" ]]; then
  ARGS+=(--resume-from "$RSMOL_STAGE2_RESUME_FROM")
fi

python -u code/RSmol/scripts/train_stage2_single_gpu.py "${ARGS[@]}"
echo "[result] status=PASS"

