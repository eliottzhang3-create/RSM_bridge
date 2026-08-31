#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in RSMOL_5_10_5_LINEAR_SOURCE_CHECKPOINT RSMOL_5_10_5_LINEAR_OUTPUT_DIR RSMOL_5_10_5_LINEAR_ALLOW_OVERWRITE; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX+="$name=$quoted_value "
  fi
done

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j stepwise-convert-5-10-5-linear-3090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/stepwise_convert_5_10_5_linear_3090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/convert_stepwise_5_10_5_linear.sh"
