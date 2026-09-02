#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
GATE="${RSMOL_5_10XR_5_STAGE4_GATE:-D}"
GATE="${GATE^^}"
if [[ "$GATE" == "B" || "$GATE" == "C" ]]; then
  echo "Stage 4 5-10xr-5 Gate $GATE is unsupported; reuse the existing Gate B JSON." >&2
  exit 2
fi
CMD_PREFIX=""
for name in \
  RSMOL_5_10XR_5_MODEL_DIR RSMOL_5_10XR_5_TOKENIZER_PATH RSMOL_5_10XR_5_DATA_DIR \
  RSMOL_5_10XR_5_OUTPUT_DIR RSMOL_5_10XR_5_REPORT RSMOL_5_10XR_5_RESUME_FROM \
  RSMOL_5_10XR_5_AUDIT_REPORT RSMOL_5_10XR_5_TASK_TMP RSMOL_5_10XR_5_STAGE4_GATE \
  RSMOL_5_10XR_5_WORLD_SIZE RSMOL_5_10XR_5_SEED RSMOL_5_10XR_5_MAX_GRAD_NORM \
  RSMOL_5_10XR_5_RECORD_BUFFER_SIZE RSMOL_5_10XR_5_WARMUP_STEPS \
  RSMOL_5_10XR_5_MAX_OPTIMIZER_STEPS RSMOL_5_10XR_5_SCHEDULER_TOTAL_STEPS \
  RSMOL_5_10XR_5_SAVE_EVERY RSMOL_5_10XR_5_ALLOW_NON8 RSMOL_5_10XR_5_DRY_RUN; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX+="$name=$quoted_value "
  fi
done

QUEUE="${RSMOL_5_10XR_5_QUEUE:-pdgpu-3090}"
JOB_NAME="${RSMOL_5_10XR_5_JOB_NAME:-stage4-5-10xr-5-${QUEUE#pdgpu-}-$(date +%m%d%H%M)}"
SUBMIT_LOG="${RSMOL_5_10XR_5_SUBMIT_LOG:-$SCRIPT_DIR/log/stage4_5_10xr_5_${QUEUE#pdgpu-}.JOB.log}"
vc submit -p "$QUEUE" -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 -j "$JOB_NAME" -d "$SCRIPT_DIR" JOB=1:1 "$SUBMIT_LOG" \
  --cmd "${CMD_PREFIX}bash scripts/train_stage4_5_10xr_5_ddp.sh"
