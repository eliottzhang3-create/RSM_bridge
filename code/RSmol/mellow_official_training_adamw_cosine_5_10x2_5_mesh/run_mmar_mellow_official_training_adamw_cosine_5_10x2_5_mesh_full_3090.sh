#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHECKPOINT="${MELLOW_MESH_EVAL_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5_10x2_5_mesh_5090/formal_30epochs_20261005_175504/checkpoints/mellow_adamw_cosine_reasonaqa_mesh_formal_20_20261005_095511181778133/model--epo-30.ckpt}"
RUNTIME_CONFIG="${MELLOW_MESH_EVAL_RUNTIME_CONFIG:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5_10x2_5_mesh_5090/formal_30epochs_20261005_175504/runtime_30epochs.yaml}"
OUTPUT_DIR="${MELLOW_MESH_MMAR_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5_10x2_5_mesh_5090/formal_30epochs_20261005_175504/eval/mmar_mesh_official_training_v1}"
DATASET_DIR="${MELLOW_MMAR_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR}"

if [[ $# -gt 4 ]]; then
  echo "usage: $0 [checkpoint] [runtime-config] [output-dir] [dataset-dir]" >&2
  exit 2
fi
CHECKPOINT="${1:-$CHECKPOINT}"
RUNTIME_CONFIG="${2:-$RUNTIME_CONFIG}"
OUTPUT_DIR="${3:-$OUTPUT_DIR}"
DATASET_DIR="${4:-$DATASET_DIR}"
[[ -f "$CHECKPOINT" ]] || { echo "MeSH checkpoint does not exist: $CHECKPOINT" >&2; exit 2; }
[[ -f "$RUNTIME_CONFIG" ]] || { echo "MeSH runtime config does not exist: $RUNTIME_CONFIG" >&2; exit 2; }
[[ -d "$SCRIPT_DIR" ]] || { echo "MeSH route root does not exist: $SCRIPT_DIR" >&2; exit 2; }
mkdir -p "$SCRIPT_DIR/log"

RUN_TAG=$(date +%Y%m%d_%H%M%S)
JOB_NAME="mellow-mesh-mmar-full-3090-$RUN_TAG"
JOB_LOG="$SCRIPT_DIR/log/$JOB_NAME.JOB.log"
ARGS=(
  --mode full
  --checkpoint-file "$CHECKPOINT"
  --runtime-config "$RUNTIME_CONFIG"
  --route-root "$SCRIPT_DIR"
  --dataset-dir "$DATASET_DIR"
  --output-dir "$OUTPUT_DIR"
  --max-prompt-tokens 129
  --max-new-tokens 32
  --dtype fp32
  --run-official-evaluation
)
printf -v CMD_ARGS '%q ' "${ARGS[@]}"

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "$JOB_NAME" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/eval/run_mmar_mesh_official_training.sh $CMD_ARGS"

echo "submitted full MMAR MeSH evaluation; output=$OUTPUT_DIR"
