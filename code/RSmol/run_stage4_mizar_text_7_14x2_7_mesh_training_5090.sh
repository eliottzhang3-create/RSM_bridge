#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
PREFIX=RSMOL_MIZAR_TEXT_7_14X2_7_MESH
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir|--data-dir|--output-dir|--resume-from|--micro-batch-size|--gradient-accumulation-steps|--gate)
      [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --model-dir) key=MODEL_DIR;; --data-dir) key=DATA_DIR;; --output-dir) key=OUTPUT_DIR;;
        --resume-from) key=RESUME_FROM;; --micro-batch-size) key=MICRO_BATCH_SIZE;;
        --gradient-accumulation-steps) key=GRADIENT_ACCUMULATION_STEPS;; --gate) key=STAGE4_GATE;;
      esac
      export "${PREFIX}_${key}=$2"; shift 2;;
    --dry-run) export "${PREFIX}_DRY_RUN=1"; shift;;
    --help|-h)
      echo 'Usage: bash run_stage4_mizar_text_7_14x2_7_mesh_training_5090.sh [--dry-run] [--output-dir NEW_PATH] [--resume-from CHECKPOINT] [--micro-batch-size 4 --gradient-accumulation-steps 32] [--gate FORMAL|D|E]'
      echo 'Default: checkpoint-001000 weights only; fresh 9244 updates; LR6e-4/6e-5; vc submit pdgpu-5090, 8 GPUs, BF16 forward, GBS1024.'; exit 0;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
export RSMOL_MIZAR_TEXT_7_14X2_7_MESH_STAGE4_GATE="${RSMOL_MIZAR_TEXT_7_14X2_7_MESH_STAGE4_GATE:-FORMAL}"
export RSMOL_MIZAR_TEXT_7_14X2_7_MESH_WORLD_SIZE=8
# MODEL_DIR is a weights/tokenizer source. Only --resume-from restores training state.
export RSMOL_MIZAR_TEXT_7_14X2_7_MESH_MODEL_DIR="${RSMOL_MIZAR_TEXT_7_14X2_7_MESH_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/formal_20261117_202454_1706253/checkpoint-001000}"
export RSMOL_MIZAR_TEXT_7_14X2_7_MESH_DATA_DIR="${RSMOL_MIZAR_TEXT_7_14X2_7_MESH_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset}"
export RSMOL_MIZAR_TEXT_7_14X2_7_MESH_OUTPUT_DIR="${RSMOL_MIZAR_TEXT_7_14X2_7_MESH_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/${RSMOL_MIZAR_TEXT_7_14X2_7_MESH_STAGE4_GATE,,}_reinit1000_lr6e4_$STAMP}"
export RSMOL_MIZAR_TEXT_7_14X2_7_MESH_RESUME_FROM="${RSMOL_MIZAR_TEXT_7_14X2_7_MESH_RESUME_FROM:-}"
export RSMOL_MIZAR_TEXT_7_14X2_7_MESH_DRY_RUN="${RSMOL_MIZAR_TEXT_7_14X2_7_MESH_DRY_RUN:-0}"
export RSMOL_MIZAR_TEXT_7_14X2_7_MESH_MICRO_BATCH_SIZE="${RSMOL_MIZAR_TEXT_7_14X2_7_MESH_MICRO_BATCH_SIZE:-8}"
export RSMOL_MIZAR_TEXT_7_14X2_7_MESH_GRADIENT_ACCUMULATION_STEPS="${RSMOL_MIZAR_TEXT_7_14X2_7_MESH_GRADIENT_ACCUMULATION_STEPS:-16}"
RUNTIME="$SCRIPT_DIR/scripts/train_stage4_mizar_text_7_14x2_7_mesh_ddp.sh"
# Validate and print the exact command before queue submission, without loading torch.
RSMOL_MIZAR_TEXT_7_14X2_7_MESH_DRY_RUN=1 bash "$RUNTIME"
[[ "$RSMOL_MIZAR_TEXT_7_14X2_7_MESH_DRY_RUN" == 1 ]] && exit 0
CMD_PREFIX=''
for suffix in STAGE4_GATE WORLD_SIZE MODEL_DIR DATA_DIR OUTPUT_DIR RESUME_FROM DRY_RUN MICRO_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS SEED CONDA_BASE; do
  name="${PREFIX}_${suffix}"
  printf -v value '%q' "${!name:-}"
  CMD_PREFIX+="$name=$value "
done
printf -v runtime_quoted '%q' "$RUNTIME"
LOG_DIR="${RSMOL_MIZAR_TEXT_7_14X2_7_MESH_JOB_LOG_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/job_logs}"
mkdir -p "$LOG_DIR"
vc submit -p pdgpu-5090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 -j "mizar-71427-lr6e4-$STAMP" -d "$SCRIPT_DIR" \
  JOB=1:1 "$LOG_DIR/$STAMP.JOB.log" --cmd "${CMD_PREFIX}bash $runtime_quoted"
