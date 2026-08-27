#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  RSMOL_RECURSIVE_OUTPUT_DIR \
  RSMOL_STAGE2_TOKENIZER_PATH \
  RSMOL_STAGE2_DATA_DIR \
  RSMOL_STAGE2_OUTPUT_DIR \
  RSMOL_STAGE2_REPORT \
  RSMOL_STAGE2_RESUME_FROM \
  RSMOL_STAGE2_MICRO_BATCH_SIZE \
  RSMOL_STAGE2_GRADIENT_ACCUMULATION_STEPS \
  RSMOL_STAGE2_LEARNING_RATE \
  RSMOL_STAGE2_CONTEXT_LENGTH \
  RSMOL_STAGE2_WARMUP_STEPS \
  RSMOL_STAGE2_MAX_OPTIMIZER_STEPS \
  RSMOL_STAGE2_SEED \
  RSMOL_STAGE2_WEIGHT_DECAY \
  RSMOL_STAGE2_MAX_GRAD_NORM \
  RSMOL_STAGE2_RECORD_BUFFER_SIZE \
  RSMOL_STAGE2_SAVE_EVERY \
  RSMOL_STAGE2_DEVICE; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX+="$name=$quoted_value "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j stage2-single-gpu-training-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/stage2_single_gpu_training_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_stage2_single_gpu.sh"

