#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SOURCE="${RSMOL_HRM3600_SOURCE_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_h2l3/formal_20261122_000344_4005340/checkpoint-step-003600}"
INIT="${RSMOL_HRM3600_INIT_SNAPSHOT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-HRM-H2L3-reinit3600-raw}"
DATA="${RSMOL_HRM3600_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset}"
OUTPUT="${RSMOL_HRM3600_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_reinit3600_bp5/formal_$STAMP}"
RESUME="${RSMOL_HRM3600_RESUME_FROM:-}"
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-checkpoint|--init-snapshot|--data-dir|--output-dir|--resume-from)
      [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --source-checkpoint) SOURCE="$2";; --init-snapshot) INIT="$2";;
        --data-dir) DATA="$2";; --output-dir) OUTPUT="$2";; --resume-from) RESUME="$2";;
      esac
      shift 2;;
    --dry-run) DRY=1; shift;;
    --help|-h)
      echo "Usage: bash $0 [--source-checkpoint STEP3600] [--init-snapshot PATH] [--data-dir PATH] [--output-dir NEW_PATH] [--resume-from NEW_ROUTE_CHECKPOINT] [--dry-run]"
      echo "Raw step3600 -> fresh9244; data seed1; BP5; LR3e-4->6e-4 at462 -> cosine6e-5; 8x2x64."
      exit 0;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
export RSMOL_HRM3600_INIT_SNAPSHOT="$INIT" RSMOL_HRM3600_DATA_DIR="$DATA"
export RSMOL_HRM3600_OUTPUT_DIR="$OUTPUT" RSMOL_HRM3600_RESUME_FROM="$RESUME" RSMOL_HRM3600_DRY_RUN="$DRY"
RUNTIME="$SCRIPT_DIR/scripts/train_text_hrm_reinit3600_bp5_ddp.sh"
if [[ "$DRY" == 1 ]]; then
  exec bash "$RUNTIME"
fi
if [[ -z "$RESUME" ]]; then
  # Pin before vc submit; failure queues nothing. Existing verified same-source snapshot is reusable.
  bash "$SCRIPT_DIR/scripts/prepare_text_hrm_reinit3600_bp5.sh" --source-checkpoint "$SOURCE" --output-dir "$INIT"
fi
PREFIX=""
for name in RSMOL_HRM3600_INIT_SNAPSHOT RSMOL_HRM3600_DATA_DIR RSMOL_HRM3600_OUTPUT_DIR RSMOL_HRM3600_RESUME_FROM RSMOL_HRM3600_DRY_RUN RSMOL_HRM3600_CONDA_BASE; do
  value="${!name:-}"
  printf -v quoted '%q' "$value"
  PREFIX+="$name=$quoted "
done
printf -v QUOTED_RUNTIME '%q' "$RUNTIME"
LOG_DIR="$SCRIPT_DIR/log/text_hrm_reinit3600_bp5"
mkdir -p "$LOG_DIR"
echo "[formal] HRM raw3600 fresh9244; BP5; seed1; Prelude gradients audited per layer"
echo "[formal] LR3e-4 ->6e-4 at462 ->cosine6e-5 at9244; output=$OUTPUT"
vc submit -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 64 -m 256G -g 8 -n 1 -j "hrm-r3600-bp5-$STAMP" -d "$SCRIPT_DIR" \
  JOB=1:1 "$LOG_DIR/formal_$STAMP.JOB.log" \
  --cmd "${PREFIX}bash $QUOTED_RUNTIME"
