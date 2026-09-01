#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_LOG_ROOT="${RSMOL_STAGE3_5_10_5_RECURSIVE_LOG_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol/log}"
case "$SUBMIT_LOG_ROOT" in
  /*) ;;
  *) echo "RSMOL_STAGE3_5_10_5_RECURSIVE_LOG_ROOT must be absolute: $SUBMIT_LOG_ROOT" >&2; exit 2 ;;
esac
mkdir -p "$SUBMIT_LOG_ROOT"

CMD_PREFIX=""
for name in \
  RSMOL_STAGE3_5_10_5_RECURSIVE_MODEL RSMOL_STAGE3_5_10_5_RECURSIVE_BENCHMARK_ROOT \
  RSMOL_STAGE3_5_10_5_RECURSIVE_OUTPUT_DIR RSMOL_STAGE3_5_10_5_RECURSIVE_DEVICE \
  RSMOL_STAGE3_5_10_5_RECURSIVE_DTYPE RSMOL_STAGE3_5_10_5_RECURSIVE_BATCH_SIZE \
  RSMOL_STAGE3_5_10_5_RECURSIVE_SEED RSMOL_STAGE3_5_10_5_RECURSIVE_TASKS \
  RSMOL_STAGE3_5_10_5_RECURSIVE_CACHE_ROOT RSMOL_STAGE3_5_10_5_RECURSIVE_LOG_ROOT \
  RSMOL_STAGE3_5_10_5_RECURSIVE_VALIDATION_ONLY RSMOL_STAGE3_5_10_5_RECURSIVE_SMOKE \
  RSMOL_STAGE3_5_10_5_RECURSIVE_NO_LOG_SAMPLES RSMOL_STAGE3_5_10_5_RECURSIVE_LIMIT; do
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
  -j stage3-5-10-5-recursive-5090-$(date +%m%d%H%M%S) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SUBMIT_LOG_ROOT/stage3_5_10_5_recursive_eval_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/evaluate_stage3_5_10_5_recursive.sh"
