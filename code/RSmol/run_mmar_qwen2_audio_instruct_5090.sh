#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
MODE="${RSMOL_QWEN2_AUDIO_MMAR_MODE:-smoke}"
MODEL_PATH="${RSMOL_QWEN2_AUDIO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen2Audio-Instruct}"
DATASET_DIR="${RSMOL_MMAR_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR}"
METADATA_JSON="${RSMOL_MMAR_METADATA_JSON:-$DATASET_DIR/MMAR-meta.json}"
AUDIO_ROOT="${RSMOL_MMAR_AUDIO_ROOT:-$DATASET_DIR/mmar-audio}"
EVALUATION_SCRIPT="${RSMOL_MMAR_EVALUATION_SCRIPT:-$DATASET_DIR/code/evaluation.py}"
OUTPUT_DIR="${RSMOL_QWEN2_AUDIO_MMAR_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/qwen2_audio_instruct_eval/mmar_dual_scoring_v1}"
JOB_LOG="$SCRIPT_DIR/log/qwen2_audio_mmar_${MODE}.${RUN_TAG}.JOB.log"

ARGS=(
  --mode "$MODE"
  --checkpoint "$MODEL_PATH"
  --dataset-dir "$DATASET_DIR"
  --metadata-json "$METADATA_JSON"
  --audio-root "$AUDIO_ROOT"
  --evaluation-script "$EVALUATION_SCRIPT"
  --output-dir "$OUTPUT_DIR"
  --max-prompt-tokens 476
  --max-new-tokens 32
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
  -c 8 -m 64G -g 1 -n 1 \
  -j "qwen2audio-mmar-${MODE}-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/evaluate_mmar_qwen2_audio_instruct.sh $CMD_ARGS"

echo "Qwen2-Audio MMAR ${MODE} output directory: $OUTPUT_DIR"
