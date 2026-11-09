#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
MODEL_DIR="${RSMOL_MIZAR_TEXT_5_10X2_5_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-text-5-10x2-5}"
DATA_DIR="${RSMOL_MIZAR_TEXT_5_10X2_5_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset}"
OUTPUT_DIR="${RSMOL_MIZAR_TEXT_5_10X2_5_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_gbs1024/formal_$STAMP/formal}"
MAX_LR="${RSMOL_MIZAR_TEXT_5_10X2_5_MAX_LR:-8e-4}"
MIN_LR="${RSMOL_MIZAR_TEXT_5_10X2_5_MIN_LR:-}"
RESUME_FROM="${RSMOL_MIZAR_TEXT_5_10X2_5_RESUME_FROM:-}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir|--data-dir|--output-dir|--max-lr|--min-lr|--resume-from)
      [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --model-dir) MODEL_DIR="$2";; --data-dir) DATA_DIR="$2";; --output-dir) OUTPUT_DIR="$2";;
        --max-lr) MAX_LR="$2";; --min-lr) MIN_LR="$2";; --resume-from) RESUME_FROM="$2";;
      esac
      shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --help|-h)
      echo "Usage: bash $0 [--model-dir PATH] [--data-dir PATH] [--output-dir NEW_PATH] [--max-lr LR] [--min-lr LR] [--resume-from CHECKPOINT] [--dry-run]"
      echo "One epoch, GBS1024, 9244 steps. min_lr defaults to 0.1*max_lr."
      echo "Epoch 2: --model-dir EPOCH1_FINAL --max-lr 2e-4 --min-lr 2e-5 (fresh optimizer/schedule)."
      echo "--resume-from restores same-epoch state and requires the original learning rates."
      exit 0;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
export RSMOL_MIZAR_TEXT_5_10X2_5_MODEL_DIR="$MODEL_DIR" RSMOL_MIZAR_TEXT_5_10X2_5_DATA_DIR="$DATA_DIR"
export RSMOL_MIZAR_TEXT_5_10X2_5_OUTPUT_DIR="$OUTPUT_DIR" RSMOL_MIZAR_TEXT_5_10X2_5_MAX_LR="$MAX_LR"
export RSMOL_MIZAR_TEXT_5_10X2_5_MIN_LR="$MIN_LR" RSMOL_MIZAR_TEXT_5_10X2_5_RESUME_FROM="$RESUME_FROM"
export RSMOL_MIZAR_TEXT_5_10X2_5_DRY_RUN="$DRY_RUN"
if [[ "$DRY_RUN" == 1 ]]; then
  exec bash "$SCRIPT_DIR/scripts/train_stage4_mizar_text_5_10x2_5_ddp.sh"
fi
CMD_PREFIX=""
for name in RSMOL_MIZAR_TEXT_5_10X2_5_MODEL_DIR RSMOL_MIZAR_TEXT_5_10X2_5_DATA_DIR \
            RSMOL_MIZAR_TEXT_5_10X2_5_OUTPUT_DIR RSMOL_MIZAR_TEXT_5_10X2_5_MAX_LR \
            RSMOL_MIZAR_TEXT_5_10X2_5_MIN_LR RSMOL_MIZAR_TEXT_5_10X2_5_RESUME_FROM RSMOL_MIZAR_TEXT_5_10X2_5_DRY_RUN \
            RSMOL_MIZAR_TEXT_5_10X2_5_TASK_TMP RSMOL_MIZAR_TEXT_5_10X2_5_CONDA_BASE; do
  value="${!name:-}"
  # Forward empty route variables too, so a job environment cannot revive an old resume path.
  printf -v quoted_value '%q' "$value"
  CMD_PREFIX+="$name=$quoted_value "
done
printf -v RUNTIME '%q' "$SCRIPT_DIR/scripts/train_stage4_mizar_text_5_10x2_5_ddp.sh"
LOG_DIR="$SCRIPT_DIR/log/mizar_text_5_10x2_5_gbs1024"
mkdir -p "$LOG_DIR"
echo "[formal] Mizar final LM -> 20 physical / 30 executions; 8 x micro8 x GA16 = GBS1024"
echo "[formal] max_lr=$MAX_LR min_lr=${MIN_LR:-0.1*max_lr}; 9244 updates / warmup463; OUTPUT_DIR=$OUTPUT_DIR"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "mizar-text-51025-$STAMP" -d "$SCRIPT_DIR" \
  JOB=1:1 "$LOG_DIR/formal_$STAMP.JOB.log" \
  --cmd "${CMD_PREFIX}bash $RUNTIME"
