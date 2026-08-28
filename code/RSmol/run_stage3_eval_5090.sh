#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
# vc's per-job log path must stay outside the Git checkout.  The submitted
# runtime also writes all task logs under RSMOL_STAGE3_OUTPUT_DIR.
SUBMIT_LOG_ROOT="${RSMOL_STAGE3_SUBMIT_LOG_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage3-submit-logs}"
case "$SUBMIT_LOG_ROOT" in
  /*) ;;
  *)
    echo "RSMOL_STAGE3_SUBMIT_LOG_ROOT must be an absolute external path: $SUBMIT_LOG_ROOT" >&2
    exit 2
    ;;
esac
case "$SUBMIT_LOG_ROOT" in
  "$SCRIPT_DIR"|"$SCRIPT_DIR"/*|"$SCRIPT_DIR/.."|"$SCRIPT_DIR/../"*|"$SCRIPT_DIR/../.."|"$SCRIPT_DIR/../../"*|/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM|/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/*)
    echo "RSMOL_STAGE3_SUBMIT_LOG_ROOT must be outside the Git checkout: $SUBMIT_LOG_ROOT" >&2
    exit 2
    ;;
esac
mkdir -p "$SUBMIT_LOG_ROOT"

CMD_PREFIX=""
for name in \
  RSMOL_STAGE3_MODEL RSMOL_STAGE3_ORIGINAL_MODEL RSMOL_STAGE3_RECURSIVE_MODEL \
  RSMOL_STAGE3_BENCHMARK_ROOT RSMOL_STAGE3_OUTPUT_DIR RSMOL_STAGE3_OUTPUT_ROOT \
  RSMOL_STAGE3_DEVICE RSMOL_STAGE3_BATCH_SIZE RSMOL_STAGE3_SEED \
  RSMOL_STAGE3_TASKS RSMOL_STAGE3_CACHE_ROOT RSMOL_STAGE3_VALIDATION_ONLY \
  RSMOL_STAGE3_SMOKE RSMOL_STAGE3_NO_LOG_SAMPLES RSMOL_STAGE3_LIMIT; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX+="$name=$quoted_value "
  fi
done

# Stage 3 is one-GPU evaluation, not Stage 4 DDP.  All model/data work is
# executed inside the submitted job and output paths are passed explicitly.
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j stage3-eval-5090-$(date +%m%d%H%M%S) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SUBMIT_LOG_ROOT/stage3_eval_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/evaluate_stage3.sh"
