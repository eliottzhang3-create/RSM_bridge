#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
MODEL_DIR="${RSMOL_5_20X2_5_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-20x2-5}"
DATA_DIR="${RSMOL_5_20X2_5_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset}"
OUTPUT_DIR="${RSMOL_5_20X2_5_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_5_20x2_5_gbs512/formal_$STAMP/formal}"
AUDIT_REPORT="${RSMOL_5_20X2_5_AUDIT_REPORT:-}"
RESUME_FROM="${RSMOL_5_20X2_5_RESUME_FROM:-}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir|--data-dir|--output-dir|--audit-report|--resume-from)
      [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --model-dir) MODEL_DIR="$2";; --data-dir) DATA_DIR="$2";; --output-dir) OUTPUT_DIR="$2";;
        --audit-report) AUDIT_REPORT="$2";; --resume-from) RESUME_FROM="$2";;
      esac
      shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --help|-h)
      echo "Usage: bash $0 [--model-dir PATH] [--data-dir PATH] [--output-dir NEW_PATH] [--audit-report CPU_JSON] [--resume-from CHECKPOINT] [--dry-run]"
      echo "Fresh: run CPU conversion first. Reuses compatible Gate B JSON, or runs a CPU audit before submitting one FORMAL job."
      exit 0;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
[[ -n "$AUDIT_REPORT" ]] || AUDIT_REPORT="$(dirname "$OUTPUT_DIR")/cpu_audit/stage4_gate_B_audit.json"
export RSMOL_5_20X2_5_MODEL_DIR="$MODEL_DIR" RSMOL_5_20X2_5_DATA_DIR="$DATA_DIR"
export RSMOL_5_20X2_5_OUTPUT_DIR="$OUTPUT_DIR" RSMOL_5_20X2_5_AUDIT_REPORT="$AUDIT_REPORT"
export RSMOL_5_20X2_5_RESUME_FROM="$RESUME_FROM" RSMOL_5_20X2_5_DRY_RUN="$DRY_RUN"
if [[ "$DRY_RUN" == 1 ]]; then
  exec bash "$SCRIPT_DIR/scripts/train_stage4_5_20x2_5_ddp.sh"
fi
CONDA_BASE="${RSMOL_5_20X2_5_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_BASE/envs/rsmol"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PREPARE=(--model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" --audit-report "$AUDIT_REPORT")
[[ -z "$RESUME_FROM" ]] || PREPARE+=(--resume-from "$RESUME_FROM")
python -u "$SCRIPT_DIR/scripts/prepare_text_5_20x2_5.py" "${PREPARE[@]}"
CMD_PREFIX=""
for name in RSMOL_5_20X2_5_MODEL_DIR RSMOL_5_20X2_5_DATA_DIR RSMOL_5_20X2_5_OUTPUT_DIR \
            RSMOL_5_20X2_5_AUDIT_REPORT RSMOL_5_20X2_5_RESUME_FROM RSMOL_5_20X2_5_TASK_TMP \
            RSMOL_5_20X2_5_CONDA_BASE; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX+="$name=$quoted_value "
  fi
done
printf -v RUNTIME '%q' "$SCRIPT_DIR/scripts/train_stage4_5_20x2_5_ddp.sh"
LOG_DIR="$SCRIPT_DIR/log/text_5_20x2_5_gbs512"
mkdir -p "$LOG_DIR"
echo "[formal] 30 physical / 50 executions; 8 x micro4 x GA16 = GBS512; 18488 updates / warmup925"
echo "[formal] OUTPUT_DIR=$OUTPUT_DIR"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "text-5-20x2-5-gbs512-$STAMP" -d "$SCRIPT_DIR" \
  JOB=1:1 "$LOG_DIR/formal_$STAMP.JOB.log" \
  --cmd "${CMD_PREFIX}bash $RUNTIME"
