#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-200000}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,GRAPH,NET}"

GATE="${RSMOL_5_10X4_5_MESH_STAGE4_GATE:-D}"
WORLD_SIZE="${RSMOL_5_10X4_5_MESH_WORLD_SIZE:-8}"
MODEL="${RSMOL_5_10X4_5_MESH_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x4-5-mesh}"
DATA="${RSMOL_5_10X4_5_MESH_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset/data}"
OUTPUT="${RSMOL_5_10X4_5_MESH_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/${GATE,,}_2epochs_$(date +%Y%m%d_%H%M%S)}"

if [[ "$GATE" == "FORMAL" ]]; then
  if [[ "$WORLD_SIZE" != "8" ]]; then
    echo "FORMAL requires WORLD_SIZE=8" >&2
    exit 2
  fi
  MICRO=4
  GA=32
  MAX_STEPS=18488
  SCHEDULER=18488
  WARMUP=925
  STEPS_PER_EPOCH=9244
  EPOCHS=2
  MAX_LR="${RSMOL_5_10X4_5_MESH_MAX_LR:-1e-3}"
  MIN_LR="${RSMOL_5_10X4_5_MESH_MIN_LR:-5e-5}"
  export RSMOL_5_10X4_5_MESH_LOG_INTERVAL_STEPS="${RSMOL_5_10X4_5_MESH_LOG_INTERVAL_STEPS:-10}"
else
  MICRO="${RSMOL_5_10X4_5_MESH_MICRO_BATCH_SIZE:-4}"
  GA="${RSMOL_5_10X4_5_MESH_GRADIENT_ACCUMULATION_STEPS:-32}"
  MAX_STEPS="${RSMOL_5_10X4_5_MESH_MAX_OPTIMIZER_STEPS:-10}"
  SCHEDULER="${RSMOL_5_10X4_5_MESH_SCHEDULER_TOTAL_STEPS:-$MAX_STEPS}"
  WARMUP="${RSMOL_5_10X4_5_MESH_WARMUP_STEPS:-1}"
  STEPS_PER_EPOCH="${RSMOL_5_10X4_5_MESH_STEPS_PER_EPOCH:-9244}"
  EPOCHS="${RSMOL_5_10X4_5_MESH_EPOCHS:-1}"
  MAX_LR="${RSMOL_5_10X4_5_MESH_MAX_LR:-1e-3}"
  MIN_LR="${RSMOL_5_10X4_5_MESH_MIN_LR:-1e-4}"
  export RSMOL_5_10X4_5_MESH_LOG_INTERVAL_STEPS="${RSMOL_5_10X4_5_MESH_LOG_INTERVAL_STEPS:-1}"
fi

if [[ ! -d "$DATA" ]]; then
  echo "persistent parquet data directory is missing: $DATA" >&2
  exit 2
fi
if [[ "$GATE" == "E" && -z "${RSMOL_5_10X4_5_MESH_RESUME_FROM:-}" ]]; then
  echo "Gate E requires RSMOL_5_10X4_5_MESH_RESUME_FROM" >&2
  exit 2
fi
if [[ -e "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing nonempty output directory: $OUTPUT" >&2
  exit 2
fi

ARGS=(
  --gate "$GATE"
  --model-path "$MODEL"
  --data-dir "$DATA"
  --output-dir "$OUTPUT"
  --world-size "$WORLD_SIZE"
  --micro-batch-size "$MICRO"
  --gradient-accumulation-steps "$GA"
  --context-length 1024
  --max-optimizer-steps "$MAX_STEPS"
  --scheduler-total-steps "$SCHEDULER"
  --warmup-steps "$WARMUP"
  --max-lr "$MAX_LR"
  --min-lr "$MIN_LR"
  --save-every 500
  --seed "${RSMOL_5_10X4_5_MESH_SEED:-0}"
  --steps-per-epoch "$STEPS_PER_EPOCH"
  --epochs "$EPOCHS"
)
[[ -n "${RSMOL_5_10X4_5_MESH_TOKENIZER_PATH:-}" ]] && ARGS+=(--tokenizer-path "$RSMOL_5_10X4_5_MESH_TOKENIZER_PATH")
[[ -n "${RSMOL_5_10X4_5_MESH_RESUME_FROM:-}" ]] && ARGS+=(--resume-from "$RSMOL_5_10X4_5_MESH_RESUME_FROM")
ARGS+=("$@")

torchrun --standalone --nproc_per_node="$WORLD_SIZE" \
  code/RSmol/scripts/train_stage4_5_10x4_5_mesh_ddp.py \
  "${ARGS[@]}"
