#!/bin/bash
set -euo pipefail
USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# DDP/NCCL diagnostics are enabled for the MeSH formal retry.  The trainer
# also writes per-rank heartbeat/failure files under output/ddp_diagnostics.
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-200000}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,GRAPH,NET}"
GATE="${RSMOL_5_10X2_5_MESH_6SLOT_STAGE4_GATE:-FORMAL}"
WORLD_SIZE="${RSMOL_5_10X2_5_MESH_6SLOT_WORLD_SIZE:-8}"
MODEL="${RSMOL_5_10X2_5_MESH_6SLOT_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x2-5-mesh-6slot}"
DATA="${RSMOL_5_10X2_5_MESH_6SLOT_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset/data}"
OUTPUT="${RSMOL_5_10X2_5_MESH_6SLOT_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh_6slot/${GATE,,}_1epoch_9244steps_fp32_$(date +%Y%m%d_%H%M%S)}"
if [[ "$GATE" == "FORMAL" && "$WORLD_SIZE" != "8" ]]; then echo "FORMAL requires WORLD_SIZE=8" >&2; exit 2; fi
if [[ "$GATE" == "FORMAL" ]]; then
  MICRO=8; GA=16; MAX_STEPS=9244; SCHEDULER=9244; WARMUP=463
  STEPS_PER_EPOCH=9244; EPOCHS=1
  MAX_LR="${RSMOL_5_10X2_5_MESH_6SLOT_MAX_LR:-8e-4}"
  MIN_LR="${RSMOL_5_10X2_5_MESH_6SLOT_MIN_LR:-8e-5}"
else
  MICRO="${RSMOL_5_10X2_5_MESH_6SLOT_MICRO_BATCH_SIZE:-8}"; GA="${RSMOL_5_10X2_5_MESH_6SLOT_GRADIENT_ACCUMULATION_STEPS:-16}"; MAX_STEPS="${RSMOL_5_10X2_5_MESH_6SLOT_MAX_OPTIMIZER_STEPS:-10}"; SCHEDULER="${RSMOL_5_10X2_5_MESH_6SLOT_SCHEDULER_TOTAL_STEPS:-$MAX_STEPS}"; WARMUP="${RSMOL_5_10X2_5_MESH_6SLOT_WARMUP_STEPS:-1}"; MAX_LR="${RSMOL_5_10X2_5_MESH_6SLOT_MAX_LR:-8e-4}"; MIN_LR="${RSMOL_5_10X2_5_MESH_6SLOT_MIN_LR:-8e-5}"
  STEPS_PER_EPOCH="${RSMOL_5_10X2_5_MESH_6SLOT_STEPS_PER_EPOCH:-9244}"; EPOCHS="${RSMOL_5_10X2_5_MESH_6SLOT_EPOCHS:-1}"
fi
if [[ ! -d "$DATA" ]]; then echo "persistent parquet data directory is missing: $DATA" >&2; exit 2; fi
if [[ -e "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing nonempty output directory: $OUTPUT" >&2; exit 2
fi
if [[ "$GATE" == "E" && -z "${RSMOL_5_10X2_5_MESH_6SLOT_RESUME_FROM:-}" ]]; then
  echo "Gate E requires RSMOL_5_10X2_5_MESH_6SLOT_RESUME_FROM" >&2; exit 2
fi
ARGS=(--gate "$GATE" --model-path "$MODEL" --data-dir "$DATA" --output-dir "$OUTPUT" --world-size "$WORLD_SIZE" --micro-batch-size "$MICRO" --gradient-accumulation-steps "$GA" --context-length 1024 --max-optimizer-steps "$MAX_STEPS" --scheduler-total-steps "$SCHEDULER" --warmup-steps "$WARMUP" --steps-per-epoch "$STEPS_PER_EPOCH" --epochs "$EPOCHS" --max-lr "$MAX_LR" --min-lr "$MIN_LR" --save-every 500 --seed "${RSMOL_5_10X2_5_MESH_6SLOT_SEED:-0}")
[[ -n "${RSMOL_5_10X2_5_MESH_6SLOT_TOKENIZER_PATH:-}" ]] && ARGS+=(--tokenizer-path "$RSMOL_5_10X2_5_MESH_6SLOT_TOKENIZER_PATH")
[[ -n "${RSMOL_5_10X2_5_MESH_6SLOT_RESUME_FROM:-}" ]] && ARGS+=(--resume-from "$RSMOL_5_10X2_5_MESH_6SLOT_RESUME_FROM")
echo "[stage4] gate=$GATE precision=FP32 epochs=$EPOCHS steps_per_epoch=$STEPS_PER_EPOCH total_steps=$MAX_STEPS warmup=$WARMUP lr=$MAX_LR->$MIN_LR output=$OUTPUT"
torchrun --standalone --nproc_per_node="$WORLD_SIZE" code/RSmol/scripts/train_stage4_5_10x2_5_mesh_6slot_ddp.py "${ARGS[@]}"
