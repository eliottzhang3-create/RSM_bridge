#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
MODE="${RSMOL_QWEN2_AUDIO_MMAU_MODE:-smoke}"
MODEL_PATH="${RSMOL_QWEN2_AUDIO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen2Audio-Instruct}"
DATASET_DIR="${RSMOL_MMAU_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini}"
PARQUET="${RSMOL_MMAU_PARQUET:-$DATASET_DIR/test_mini.parquet}"
METADATA_JSON="${RSMOL_MMAU_METADATA_JSON:-$DATASET_DIR/mmau-test-mini.json}"
EVALUATION_SCRIPT="${RSMOL_MMAU_EVALUATION_SCRIPT:-$DATASET_DIR/evaluation.py}"
AUDIO_ROOT="${RSMOL_MMAU_AUDIO_ROOT:-$DATASET_DIR/test-mini-audios}"
OUTPUT_DIR="${RSMOL_QWEN2_AUDIO_MMAU_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/qwen2_audio_instruct_eval/mmau_test_mini_dual_scoring_v2_paren_casefold}"
JOB_LOG="$SCRIPT_DIR/log/qwen2_audio_mmau_${MODE}.${RUN_TAG}.JOB.log"

ARGS=(
  --mode "$MODE"
  --checkpoint "$MODEL_PATH"
  --dataset-dir "$DATASET_DIR"
  --parquet "$PARQUET"
  --metadata-json "$METADATA_JSON"
  --evaluation-script "$EVALUATION_SCRIPT"
  --audio-root "$AUDIO_ROOT"
  --output-dir "$OUTPUT_DIR"
  --parquet-batch-size 8
  --max-prompt-tokens 129
  --max-new-tokens 300
  --dtype bf16
  --run-official-evaluation
)
if (($#)); then
  ARGS+=("$@")
fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 64G -g 1 -n 1 \
  -j "qwen2audio-mmau-${MODE}-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/evaluate_mmau_test_mini_qwen2_audio_instruct.sh $CMD_ARGS"

echo "Qwen2-Audio MMAU ${MODE} output directory: $OUTPUT_DIR"
