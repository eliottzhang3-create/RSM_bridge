#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_LOG_ROOT="${RSMOL_BASELINE_SUBMIT_LOG_ROOT:-${RSMOL_BASELINE_LOG_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol/log}}"
mkdir -p "$SUBMIT_LOG_ROOT"

CMD_PREFIX=""
for name in \
  USER_CONDA_BASE RSMOL_BASELINE_FALCON_MODEL RSMOL_BASELINE_BENCHMARK_ROOT \
  RSMOL_BASELINE_FALCON_OUTPUT_DIR RSMOL_BASELINE_DEVICE RSMOL_BASELINE_DTYPE \
  RSMOL_BASELINE_BATCH_SIZE RSMOL_BASELINE_SEED RSMOL_BASELINE_TASKS \
  RSMOL_BASELINE_CACHE_DIR RSMOL_BASELINE_LOG_ROOT RSMOL_BASELINE_SUBMIT_LOG_ROOT \
  RSMOL_BASELINE_CONDA_ENV RSMOL_BASELINE_VALIDATION_ONLY RSMOL_BASELINE_SMOKE \
  RSMOL_BASELINE_NO_LOG_SAMPLES RSMOL_BASELINE_LIMIT; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX+="$name=$quoted_value "
  fi
done

vc submit \
  -p pdgpu-4090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j stage3-baseline-falcon90m-4090-$(date +%m%d%H%M%S) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SUBMIT_LOG_ROOT/stage3_baseline_falcon90m.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/evaluate_stage3_baseline_falcon90m.sh"
