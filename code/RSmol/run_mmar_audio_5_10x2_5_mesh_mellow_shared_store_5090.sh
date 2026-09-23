#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
CHECKPOINT="${RSMOL_MMAR_SHARED_STORE_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/formal_fixed260_3ep_20260922_v1/checkpoint-011343}"
DATASET_DIR="${RSMOL_MMAR_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR}"
METADATA_JSON="${RSMOL_MMAR_METADATA_JSON:-$DATASET_DIR/MMAR-meta.json}"
AUDIO_ROOT="${RSMOL_MMAR_AUDIO_ROOT:-$DATASET_DIR/mmar-audio}"
EVALUATION_SCRIPT="${RSMOL_MMAR_EVALUATION_SCRIPT:-$DATASET_DIR/code/evaluation.py}"
HTSAT_CHECKPOINT="${RSMOL_HTSAT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt}"
MELLOW_ROOT="${RSMOL_MELLOW_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main}"
OUTPUT_DIR="${RSMOL_MMAR_SHARED_STORE_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/formal_fixed260_3ep_20260922_v1/mmar_checkpoint_011343_verbatim_v1}"
JOB_LOG="$SCRIPT_DIR/log/mmar_shared_store_011343_5090.${RUN_TAG}.JOB.log"

ARGS=(
  --mode full
  --checkpoint "$CHECKPOINT"
  --dataset-dir "$DATASET_DIR"
  --metadata-json "$METADATA_JSON"
  --audio-root "$AUDIO_ROOT"
  --evaluation-script "$EVALUATION_SCRIPT"
  --htsat-checkpoint "$HTSAT_CHECKPOINT"
  --mellow-root "$MELLOW_ROOT"
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
  -c 32 -m 256G -g 1 -n 1 \
  -j "mmar-shared-011343-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/evaluate_mmar_audio_5_10x2_5_mesh_mellow_shared_store.sh $CMD_ARGS"

echo "Shared-store checkpoint-011343 MMAR full output directory: ${OUTPUT_DIR}"
