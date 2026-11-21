#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
MODEL_DIR="${RSMOL_HRM_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-HRM-H2L3-P5-L5-H5-C5}"
DATA_DIR="${RSMOL_HRM_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset}"
OUTPUT_DIR="${RSMOL_HRM_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_h2l3/formal_$STAMP}"
RESUME_FROM="${RSMOL_HRM_RESUME_FROM:-}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir|--data-dir|--output-dir|--resume-from)
      [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --model-dir) MODEL_DIR="$2";; --data-dir) DATA_DIR="$2";;
        --output-dir) OUTPUT_DIR="$2";; --resume-from) RESUME_FROM="$2";;
      esac
      shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --help|-h)
      echo "Usage: bash $0 [--model-dir PATH] [--data-dir PATH] [--output-dir NEW_PATH] [--resume-from CHECKPOINT] [--dry-run]"
      echo "20 physical / 50 executions; H2L3; 8x2x64; 9244 updates; Adam-atan2/EMA; warmup3000 to3e-4 then constant."
      exit 0;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
export RSMOL_HRM_MODEL_DIR="$MODEL_DIR" RSMOL_HRM_DATA_DIR="$DATA_DIR"
export RSMOL_HRM_OUTPUT_DIR="$OUTPUT_DIR" RSMOL_HRM_RESUME_FROM="$RESUME_FROM" RSMOL_HRM_DRY_RUN="$DRY_RUN"
RUNTIME="$SCRIPT_DIR/scripts/train_text_hrm_h2l3_ddp.sh"
if [[ "$DRY_RUN" == 1 ]]; then
  exec bash "$RUNTIME"
fi
PREFIX=""
for name in RSMOL_HRM_MODEL_DIR RSMOL_HRM_DATA_DIR RSMOL_HRM_OUTPUT_DIR RSMOL_HRM_RESUME_FROM RSMOL_HRM_DRY_RUN RSMOL_HRM_CONDA_BASE; do
  value="${!name:-}"
  printf -v quoted '%q' "$value"
  PREFIX+="$name=$quoted "
done
printf -v QUOTED_RUNTIME '%q' "$RUNTIME"
LOG_DIR="$SCRIPT_DIR/log/text_hrm_h2l3"
mkdir -p "$LOG_DIR"
echo "[formal] HRM P5/L5/H5/C5, H2L3, 20 physical / 50 executions"
echo "[formal] 8xmicro2xGA64=1024; 9244 updates; Adam-atan2/EMA .9999; no clipping"
echo "[formal] warmup3000 -> 3e-4 constant; BP3/4/5; output=$OUTPUT_DIR"
vc submit -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 64 -m 256G -g 8 -n 1 -j "hrm-text-h2l3-$STAMP" -d "$SCRIPT_DIR" \
  JOB=1:1 "$LOG_DIR/formal_$STAMP.JOB.log" \
  --cmd "${PREFIX}bash $QUOTED_RUNTIME"
