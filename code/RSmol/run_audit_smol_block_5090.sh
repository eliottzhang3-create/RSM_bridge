#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_ROOT="${RSMOL_SMOL_BLOCK_LOG_ROOT:-$SCRIPT_DIR/log}"
mkdir -p "$LOG_ROOT"
CMD_PREFIX=""
for name in \
  RSMOL_SMOL_BLOCK_MODEL_PATH \
  RSMOL_SMOL_BLOCK_DEVICE \
  RSMOL_SMOL_BLOCK_DTYPE \
  RSMOL_SMOL_BLOCK_LAYER_INDEX \
  RSMOL_SMOL_BLOCK_SEED \
  RSMOL_SMOL_BLOCK_SEQ_LEN \
  RSMOL_SMOL_BLOCK_OUTPUT_DIR \
  RSMOL_SMOL_BLOCK_LOG_ROOT; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p "${RSMOL_SMOL_BLOCK_QUEUE:-pdgpu-5090}" \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c "${RSMOL_SMOL_BLOCK_CPUS:-8}" \
  -m "${RSMOL_SMOL_BLOCK_MEMORY:-32G}" \
  -g "${RSMOL_SMOL_BLOCK_GPUS:-1}" \
  -n 1 \
  -j smol-block-audit-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$LOG_ROOT/smol_block_audit_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/audit_smol_block_5090.sh"

