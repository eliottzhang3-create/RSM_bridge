#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in RSMOL_5_10_5_LINEAR_MODEL_DIR RSMOL_5_10_5_LINEAR_SMOKE_REPORT RSMOL_5_10_5_LINEAR_DEVICE RSMOL_5_10_5_LINEAR_MAX_NEW_TOKENS RSMOL_5_10_5_LINEAR_SMOKE_PROMPT RSMOL_5_10_5_LINEAR_SAMPLE_PROMPTS; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX+="$name=$quoted_value "
  fi
done

QUEUE="${RSMOL_5_10_5_LINEAR_QUEUE:-pdgpu-3090}"
JOB_NAME="${RSMOL_5_10_5_LINEAR_JOB_NAME:-stage1-5-10-5-linear-${QUEUE#pdgpu-}-$(date +%m%d%H%M)}"
SUBMIT_LOG="${RSMOL_5_10_5_LINEAR_SUBMIT_LOG:-$SCRIPT_DIR/log/stage1_5_10_5_linear_${QUEUE#pdgpu-}.JOB.log}"

vc submit \
  -p "$QUEUE" \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "$JOB_NAME" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SUBMIT_LOG" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_5_10_5_linear.sh"
