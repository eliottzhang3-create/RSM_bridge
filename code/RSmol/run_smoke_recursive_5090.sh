#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in RSMOL_RECURSIVE_OUTPUT_DIR RSMOL_RECURSIVE_SMOKE_REPORT RSMOL_RECURSIVE_DEVICE RSMOL_RECURSIVE_MAX_NEW_TOKENS RSMOL_RECURSIVE_SMOKE_PROMPT; do
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
  -j recursive-smoke-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/recursive_smoke_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_recursive.sh"
