#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

# Gate B is a single-process Parquet/tokenizer pre-audit and does not launch
# torchrun.  Request only one GPU for that gate; the DDP gates retain the full
# eight-card 5090 allocation.
GATE="${RSMOL_STAGE4_GATE:-D}"
GPU_COUNT=8
if [[ "$GATE" == "B" ]]; then
  GPU_COUNT=1
fi

CMD_PREFIX=""
for name in \
  RSMOL_RECURSIVE_OUTPUT_DIR RSMOL_STAGE4_TOKENIZER_PATH RSMOL_STAGE4_DATA_DIR \
  RSMOL_STAGE4_OUTPUT_DIR RSMOL_STAGE4_REPORT RSMOL_STAGE4_RESUME_FROM \
  RSMOL_STAGE4_AUDIT_REPORT \
  RSMOL_STAGE4_TASK_TMP RSMOL_STAGE4_GATE RSMOL_STAGE4_WORLD_SIZE \
  RSMOL_STAGE4_MICRO_BATCH_SIZE RSMOL_STAGE4_GRADIENT_ACCUMULATION_STEPS \
  RSMOL_STAGE4_LEARNING_RATE RSMOL_STAGE4_CONTEXT_LENGTH RSMOL_STAGE4_WARMUP_STEPS \
  RSMOL_STAGE4_MAX_OPTIMIZER_STEPS RSMOL_STAGE4_SEED RSMOL_STAGE4_WEIGHT_DECAY \
  RSMOL_STAGE4_MAX_GRAD_NORM RSMOL_STAGE4_RECORD_BUFFER_SIZE RSMOL_STAGE4_SAVE_EVERY \
  RSMOL_STAGE4_MONITOR_INTERVAL RSMOL_STAGE4_BACKEND RSMOL_STAGE4_ALLOW_NON8 \
  RSMOL_STAGE4_DRY_RUN; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX+="$name=$quoted_value "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g "$GPU_COUNT" -n 1 \
  -j stage4-ddp-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/stage4_ddp_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/train_stage4_ddp.sh"
