#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG=$(date +%Y%m%d_%H%M%S)
ROUTE="${RSMOL_MELLOW_TRAINING_ROUTE:-c8204d8}"
MODE="${RSMOL_MELLOW_EVAL_MODE:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "RSMOL_MELLOW_EVAL_MODE must be smoke or full: $MODE" >&2
  exit 2
fi
case "$ROUTE" in
  c8204d8)
    DEFAULT_OUTPUT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/formal_30epochs_20260930_132438/eval/mmar_official_training_v1"
    ;;
  adamw_cosine)
    DEFAULT_OUTPUT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5090/formal_30epochs_20261001_003838/eval/mmar_official_training_v1"
    ;;
  *) echo "unsupported route: $ROUTE" >&2; exit 2 ;;
esac
OUTPUT_DIR="${RSMOL_MELLOW_EVAL_OUTPUT_DIR:-$DEFAULT_OUTPUT}"
DATASET_DIR="${RSMOL_MMAR_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR}"
METADATA_JSON="${RSMOL_MMAR_METADATA_JSON:-$DATASET_DIR/MMAR-meta.json}"
AUDIO_ROOT="${RSMOL_MMAR_AUDIO_ROOT:-$DATASET_DIR/mmar-audio}"
EVALUATION_SCRIPT="${RSMOL_MMAR_EVALUATION_SCRIPT:-$DATASET_DIR/code/evaluation.py}"
ARGS=(
  --route "$ROUTE"
  --mode "$MODE"
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
if [[ -n "${RSMOL_MELLOW_CHECKPOINT_FILE:-}" ]]; then ARGS+=(--checkpoint-file "$RSMOL_MELLOW_CHECKPOINT_FILE"); fi
if [[ -n "${RSMOL_MELLOW_RUNTIME_CONFIG:-}" ]]; then ARGS+=(--runtime-config "$RSMOL_MELLOW_RUNTIME_CONFIG"); fi
if [[ -n "${RSMOL_MELLOW_ROUTE_ROOT:-}" ]]; then ARGS+=(--route-root "$RSMOL_MELLOW_ROUTE_ROOT"); fi
if (($#)); then ARGS+=("$@"); fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"
JOB_NAME="mellow-mmar-$ROUTE-$MODE-$RUN_TAG"
if ((${#JOB_NAME} > 60)); then
  echo "generated vc job name exceeds 60 characters: $JOB_NAME (${#JOB_NAME})" >&2
  exit 2
fi
JOB_LOG="$SCRIPT_DIR/log/$JOB_NAME.JOB.log"
vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "$JOB_NAME" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/evaluate_mmar_mellow_official_training.sh $CMD_ARGS"

echo "submitted MMAR route=$ROUTE mode=$MODE output=$OUTPUT_DIR"
