#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
CMD_PREFIX=""
for name in RSMOL_5_10XPOISSON_PARCAE_MODEL_DIR RSMOL_5_10XPOISSON_PARCAE_STAGE1_REPORT RSMOL_5_10XPOISSON_PARCAE_DEVICE; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then printf -v quoted_value '%q' "$value"; CMD_PREFIX+="$name=$quoted_value "; fi
done
vc submit -p pdgpu-3090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 -c 8 -m 32G -g 1 -n 1 \
  -j "audit-stage1-5-10xpoisson-parcae-$(date +%m%d%H%M)" -d "$SCRIPT_DIR" JOB=1:1 "$SCRIPT_DIR/log/audit_stage1_5_10xpoisson_parcae.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/audit_stage1_5_10xpoisson_parcae.sh"
