#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
DATASET_DIR="${RSMOL_MMAR_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR}"
METADATA_JSON="${RSMOL_MMAR_METADATA_JSON:-$DATASET_DIR/MMAR-meta.json}"
AUDIO_ROOT="${RSMOL_MMAR_AUDIO_ROOT:-$DATASET_DIR/mmar-audio}"
EVALUATION_SCRIPT="${RSMOL_MMAR_EVALUATION_SCRIPT:-$DATASET_DIR/code/evaluation.py}"
OUTPUT_DIR="${RSMOL_MELLOW_V0_MMAR_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/mmar_mellow_v0_dual_scoring_matched_smollm2_113430_v1}"
PREFLIGHT_REPORT="${RSMOL_MELLOW_V0_PREFLIGHT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/preflight/mellow_v0_artifact_preflight.json}"
MODE="${RSMOL_MELLOW_V0_MMAR_MODE:-full}"
JOB_LOG="$SCRIPT_DIR/log/mmar_mellow_v0_4090.${RUN_TAG}.JOB.log"

ARGS=(
  --mode "$MODE"
  --preflight-report "$PREFLIGHT_REPORT"
  --dataset-dir "$DATASET_DIR"
  --metadata-json "$METADATA_JSON"
  --audio-root "$AUDIO_ROOT"
  --evaluation-script "$EVALUATION_SCRIPT"
  --output-dir "$OUTPUT_DIR"
  --max-prompt-tokens 129
  --max-new-tokens 32
  --dtype fp32
  --run-official-evaluation
)
if (($#)); then
  ARGS+=("$@")
fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"

vc submit \
  -p pdgpu-4090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "mmar-mellow-v0-$MODE-$RUN_TAG" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/evaluate_mmar_mellow_v0.sh $CMD_ARGS"

echo "Mellow-v0 MMAR $MODE output directory: $OUTPUT_DIR"
