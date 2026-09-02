#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in RSMOL_5_10XR_5_MODEL_DIR RSMOL_5_10XR_5_SMOKE_REPORT RSMOL_5_10XR_5_DEVICE RSMOL_5_10XR_5_MAX_NEW_TOKENS; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX+="$name=$quoted_value "
  fi
done

QUEUE="${RSMOL_RECURSIVE_QUEUE:-pdgpu-3090}"
JOB_NAME="${RSMOL_5_10XR_5_JOB_NAME:-recursive-5-10xr-5-${QUEUE#pdgpu-}-$(date +%m%d%H%M)}"
SUBMIT_LOG="${RSMOL_5_10XR_5_SUBMIT_LOG:-$SCRIPT_DIR/log/recursive_5_10xr_5_${QUEUE#pdgpu-}.JOB.log}"

vc submit \
  -p "$QUEUE" \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "$JOB_NAME" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SUBMIT_LOG" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_recursive_5_10xr_5.sh"
