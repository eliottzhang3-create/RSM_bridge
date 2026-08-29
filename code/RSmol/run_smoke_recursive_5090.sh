#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in RSMOL_RECURSIVE_OUTPUT_DIR RSMOL_RECURSIVE_SMOKE_REPORT RSMOL_RECURSIVE_DEVICE RSMOL_RECURSIVE_MAX_NEW_TOKENS RSMOL_RECURSIVE_SMOKE_PROMPT RSMOL_RECURSIVE_SAMPLE_PROMPTS; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX+="$name=$quoted_value "
  fi
done

QUEUE="${RSMOL_RECURSIVE_QUEUE:-pdgpu-5090}"
JOB_NAME="${RSMOL_RECURSIVE_JOB_NAME:-recursive-generation-${QUEUE#pdgpu-}-$(date +%m%d%H%M)}"
SUBMIT_LOG="${RSMOL_RECURSIVE_SUBMIT_LOG:-$SCRIPT_DIR/log/recursive_generation_${QUEUE#pdgpu-}.JOB.log}"

vc submit \
  -p "$QUEUE" \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "$JOB_NAME" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SUBMIT_LOG" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_recursive.sh"
