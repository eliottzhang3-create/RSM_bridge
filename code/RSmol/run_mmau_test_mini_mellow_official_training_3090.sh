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
    DEFAULT_OUTPUT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/formal_30epochs_20260930_132438/eval/mmau_test_mini_official_training_v1"
    ;;
  adamw_cosine)
    DEFAULT_OUTPUT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5090/formal_30epochs_20261001_003838/eval/mmau_test_mini_official_training_v1"
    ;;
  *) echo "unsupported route: $ROUTE" >&2; exit 2 ;;
esac
OUTPUT_DIR="${RSMOL_MELLOW_EVAL_OUTPUT_DIR:-$DEFAULT_OUTPUT}"
DATASET_DIR="${RSMOL_MMAU_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini}"
PARQUET="${RSMOL_MMAU_PARQUET:-$DATASET_DIR/test_mini.parquet}"
METADATA_JSON="${RSMOL_MMAU_METADATA_JSON:-$DATASET_DIR/mmau-test-mini.json}"
AUDIO_ROOT="${RSMOL_MMAU_AUDIO_ROOT:-$DATASET_DIR/test-mini-audios}"
EVALUATION_SCRIPT="${RSMOL_MMAU_EVALUATION_SCRIPT:-$DATASET_DIR/evaluation.py}"
ARGS=(
  --route "$ROUTE"
  --mode "$MODE"
  --dataset-dir "$DATASET_DIR"
  --parquet "$PARQUET"
  --metadata-json "$METADATA_JSON"
  --audio-root "$AUDIO_ROOT"
  --evaluation-script "$EVALUATION_SCRIPT"
  --output-dir "$OUTPUT_DIR"
  --parquet-batch-size 8
  --max-prompt-tokens 129
  --max-new-tokens 300
  --dtype fp32
  --run-official-evaluation
)
if [[ -n "${RSMOL_MELLOW_CHECKPOINT_FILE:-}" ]]; then ARGS+=(--checkpoint-file "$RSMOL_MELLOW_CHECKPOINT_FILE"); fi
if [[ -n "${RSMOL_MELLOW_RUNTIME_CONFIG:-}" ]]; then ARGS+=(--runtime-config "$RSMOL_MELLOW_RUNTIME_CONFIG"); fi
if [[ -n "${RSMOL_MELLOW_ROUTE_ROOT:-}" ]]; then ARGS+=(--route-root "$RSMOL_MELLOW_ROUTE_ROOT"); fi
if (($#)); then ARGS+=("$@"); fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"
JOB_LOG="$SCRIPT_DIR/log/mellow-official-training-mmau-$ROUTE-$MODE.$RUN_TAG.JOB.log"
vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "mellow-official-training-mmau-$ROUTE-$MODE-$RUN_TAG" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/evaluate_mmau_test_mini_mellow_official_training.sh $CMD_ARGS"

echo "submitted MMAU route=$ROUTE mode=$MODE output=$OUTPUT_DIR"
