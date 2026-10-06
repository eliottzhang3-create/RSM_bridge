#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
ROUTE="${RSMOL_MMAU_ROUTE:-x5_7slot}"
MODE="${RSMOL_MMAU_MODE:-full}"
CHECKPOINT="${RSMOL_MMAU_CHECKPOINT:-}"
DATASET_DIR="${RSMOL_MMAU_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini}"
PARQUET="${RSMOL_MMAU_PARQUET:-$DATASET_DIR/test_mini.parquet}"
METADATA_JSON="${RSMOL_MMAU_METADATA_JSON:-$DATASET_DIR/mmau-test-mini.json}"
EVALUATION_SCRIPT="${RSMOL_MMAU_EVALUATION_SCRIPT:-$DATASET_DIR/evaluation.py}"
AUDIO_ROOT="${RSMOL_MMAU_AUDIO_ROOT:-$DATASET_DIR/test-mini-audios}"
HTSAT_CHECKPOINT="${RSMOL_HTSAT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt}"
MELLOW_ROOT="${RSMOL_MELLOW_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main}"
OUTPUT_DIR="${RSMOL_MMAU_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mmau_audio_mesh_zero_slot/${ROUTE}/${MODE}_${RUN_TAG}}"

if [[ -z "$CHECKPOINT" ]]; then
  case "$ROUTE" in
    x2_7slot)
      CHECKPOINT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_7slot_mellow_shared_store_configurable_epochs/formal_3epochs_20261002_metadatafix_v1/checkpoint-011343"
      ;;
    x4)
      CHECKPOINT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/formal_3epochs_20261002_configfix_v3/checkpoint-011343"
      ;;
    x4_8slot)
      CHECKPOINT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x4_5_mesh_8slot_mellow_shared_store_configurable_epochs/formal_3epochs_20261004_x4_8slot_v2/checkpoint-011343"
      ;;
    x3_7slot)
      CHECKPOINT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x3_5_mesh_7slot_mellow_shared_store_configurable_epochs/formal_3epochs_20261003_125649710077883-20/checkpoint-011343"
      ;;
    x5_7slot)
      CHECKPOINT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x5_5_mesh_7slot_mellow_shared_store_configurable_epochs/formal_3epochs_20261002_151018142328161-20/checkpoint-011343"
      ;;
    x5_8slot)
      CHECKPOINT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x5_5_mesh_8slot_mellow_shared_store_configurable_epochs/formal_3epochs_20261004_x5_8slot_v1/checkpoint-011343"
      ;;
    x5_9slot)
      CHECKPOINT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x5_5_mesh_9slot_mellow_shared_store_configurable_epochs/formal_3epochs_20261006_054353274259875-20/checkpoint-011343"
      ;;
    *)
      echo "unsupported RSMOL_MMAU_ROUTE=$ROUTE" >&2
      exit 2
      ;;
  esac
fi

JOB_LOG="$SCRIPT_DIR/log/mmau-${ROUTE}-${MODE}.${RUN_TAG}.JOB.log"
JOB_NAME="mmau-${ROUTE}-${MODE}-${RUN_TAG}"

ARGS=(
  --route "$ROUTE"
  --mode "$MODE"
  --checkpoint "$CHECKPOINT"
  --dataset-dir "$DATASET_DIR"
  --parquet "$PARQUET"
  --metadata-json "$METADATA_JSON"
  --evaluation-script "$EVALUATION_SCRIPT"
  --audio-root "$AUDIO_ROOT"
  --htsat-checkpoint "$HTSAT_CHECKPOINT"
  --mellow-root "$MELLOW_ROOT"
  --output-dir "$OUTPUT_DIR"
  --parquet-batch-size 8
  --max-prompt-tokens 129
  --max-new-tokens 300
  --dtype fp32
  --run-official-evaluation
)
if (($#)); then
  ARGS+=("$@")
fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "$JOB_NAME" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/evaluate_mmau_test_mini_audio_mesh_shared_store.sh $CMD_ARGS"

echo "MMAU ${MODE} ${ROUTE} output directory: ${OUTPUT_DIR}"
