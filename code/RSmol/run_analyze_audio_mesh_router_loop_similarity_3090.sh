#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test.json}"
AUDIO_ROOT="${AUDIO_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_mesh_router_loop_similarity/${RUN_TAG}}"
MELLOW_ROOT="${MELLOW_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/mellow-main}"
HTSAT_CHECKPOINT="${HTSAT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt}"
X2_CHECKPOINT="${X2_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_7slot_mellow_shared_store_configurable_epochs/formal_3epochs_20261002_metadatafix_v1/checkpoint-011343}"
X3_CHECKPOINT="${X3_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x3_5_mesh_7slot_mellow_shared_store_configurable_epochs/formal_3epochs_20261003_125649710077883-20/checkpoint-011343}"
X4_CHECKPOINT="${X4_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/formal_3epochs_20261002_configfix_v3/checkpoint-011343}"

ARGS=(
  --dataset "$DATASET"
  --output-dir "$OUTPUT_DIR"
  --mellow-root "$MELLOW_ROOT"
  --htsat-checkpoint "$HTSAT_CHECKPOINT"
  --x2-checkpoint "$X2_CHECKPOINT"
  --x3-checkpoint "$X3_CHECKPOINT"
  --x4-checkpoint "$X4_CHECKPOINT"
)
if [[ -n "$AUDIO_ROOT" ]]; then
  ARGS+=(--audio-root "$AUDIO_ROOT")
fi

printf -v CMD_ARGS '%q ' "${ARGS[@]}"
mkdir -p "$SCRIPT_DIR/log"

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "mesh-router-loop-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mesh-router-loop.${RUN_TAG}.JOB.log" \
  --cmd "bash scripts/analyze_audio_mesh_router_loop_similarity.sh ${CMD_ARGS}"
