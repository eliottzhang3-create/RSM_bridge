#!/bin/bash
set -euo pipefail

# GPU-backed Stage 4 entry point for the isolated 5-10-5 linear comparison.
# Gate B remains the existing CPU-only external dataset audit.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

GATE="${RSMOL_5_10_5_LINEAR_STAGE4_GATE:-D}"
GATE="${GATE^^}"
if [[ "$GATE" == "B" || "$GATE" == "C" ]]; then
  echo "Stage 4 5-10-5 linear Gate $GATE is unsupported; reuse the external Gate B JSON and run Gate A, D, E, or FORMAL." >&2
  exit 2
fi

CMD_PREFIX=""
for name in \
  RSMOL_5_10_5_LINEAR_MODEL_DIR RSMOL_5_10_5_LINEAR_TOKENIZER_PATH RSMOL_5_10_5_LINEAR_DATA_DIR \
  RSMOL_5_10_5_LINEAR_OUTPUT_DIR RSMOL_5_10_5_LINEAR_REPORT RSMOL_5_10_5_LINEAR_RESUME_FROM \
  RSMOL_5_10_5_LINEAR_AUDIT_REPORT RSMOL_5_10_5_LINEAR_TASK_TMP RSMOL_5_10_5_LINEAR_STAGE4_GATE \
  RSMOL_5_10_5_LINEAR_WORLD_SIZE RSMOL_5_10_5_LINEAR_MICRO_BATCH_SIZE \
  RSMOL_5_10_5_LINEAR_GRADIENT_ACCUMULATION_STEPS RSMOL_5_10_5_LINEAR_LEARNING_RATE \
  RSMOL_5_10_5_LINEAR_MAX_LR RSMOL_5_10_5_LINEAR_MIN_LR RSMOL_5_10_5_LINEAR_CONTEXT_LENGTH \
  RSMOL_5_10_5_LINEAR_WARMUP_STEPS RSMOL_5_10_5_LINEAR_MAX_OPTIMIZER_STEPS \
  RSMOL_5_10_5_LINEAR_FORMAL_OPTIMIZER_STEPS RSMOL_5_10_5_LINEAR_LOG_INTERVAL_STEPS \
  RSMOL_5_10_5_LINEAR_SCHEDULER_TOTAL_STEPS RSMOL_5_10_5_LINEAR_SEED \
  RSMOL_5_10_5_LINEAR_WEIGHT_DECAY RSMOL_5_10_5_LINEAR_MAX_GRAD_NORM \
  RSMOL_5_10_5_LINEAR_RECORD_BUFFER_SIZE RSMOL_5_10_5_LINEAR_SAVE_EVERY \
  RSMOL_5_10_5_LINEAR_CHECKPOINT_RETENTION RSMOL_5_10_5_LINEAR_MONITOR_INTERVAL \
  RSMOL_5_10_5_LINEAR_BACKEND RSMOL_5_10_5_LINEAR_ALLOW_NON8 RSMOL_5_10_5_LINEAR_DRY_RUN; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX+="$name=$quoted_value "
  fi
done

QUEUE="${RSMOL_5_10_5_LINEAR_QUEUE:-pdgpu-3090}"
JOB_NAME="${RSMOL_5_10_5_LINEAR_JOB_NAME:-stage4-5-10-5-linear-${GATE,,}-${QUEUE#pdgpu-}-$(date +%m%d%H%M)}"
SUBMIT_LOG="${RSMOL_5_10_5_LINEAR_SUBMIT_LOG:-$SCRIPT_DIR/log/stage4_5_10_5_linear_${GATE,,}_${QUEUE#pdgpu-}.JOB.log}"

vc submit \
  -p "$QUEUE" \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "$JOB_NAME" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SUBMIT_LOG" \
  --cmd "${CMD_PREFIX}bash scripts/train_stage4_5_10_5_linear_ddp.sh"
