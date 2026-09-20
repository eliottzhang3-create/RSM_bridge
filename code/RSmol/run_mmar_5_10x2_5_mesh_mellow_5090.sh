#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
CHECKPOINT="${RSMOL_MMAR_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/partition_formal_answer_eos_v2_10epochs_20260918/checkpoint-037810}"
DATASET_DIR="${RSMOL_MMAR_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR}"
METADATA_JSON="${RSMOL_MMAR_METADATA_JSON:-$DATASET_DIR/MMAR-meta.json}"
AUDIO_ROOT="${RSMOL_MMAR_AUDIO_ROOT:-$DATASET_DIR/mmar-audio}"
EVALUATION_SCRIPT="${RSMOL_MMAR_EVALUATION_SCRIPT:-$DATASET_DIR/code/evaluation.py}"
HTSAT_CHECKPOINT="${RSMOL_HTSAT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt}"
MELLOW_ROOT="${RSMOL_MELLOW_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main}"
OUTPUT_DIR="${RSMOL_MMAR_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/mmar_fixed_order_${RUN_TAG}}"
MODE="${RSMOL_MMAR_MODE:-smoke}"
JOB_LOG="$SCRIPT_DIR/log/mmar_5090.${RUN_TAG}.JOB.log"

ARGS=(
  --mode "$MODE"
  --checkpoint "$CHECKPOINT"
  --dataset-dir "$DATASET_DIR"
  --metadata-json "$METADATA_JSON"
  --audio-root "$AUDIO_ROOT"
  --evaluation-script "$EVALUATION_SCRIPT"
  --htsat-checkpoint "$HTSAT_CHECKPOINT"
  --mellow-root "$MELLOW_ROOT"
  --output-dir "$OUTPUT_DIR"
  --max-prompt-tokens 622
  --max-new-tokens 16
  --dtype bf16
  --run-official-evaluation
)
if (($#)); then
  ARGS+=("$@")
fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 1 -n 1 \
  -j "mmar-5090-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/evaluate_mmar_5_10x2_5_mesh_mellow.sh $CMD_ARGS"

echo "MMAR ${MODE} output directory: ${OUTPUT_DIR}"
