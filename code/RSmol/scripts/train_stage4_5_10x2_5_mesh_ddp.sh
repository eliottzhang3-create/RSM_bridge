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
GATE="${RSMOL_5_10X2_5_MESH_STAGE4_GATE:-D}"
WORLD_SIZE="${RSMOL_5_10X2_5_MESH_WORLD_SIZE:-8}"
MODEL="${RSMOL_5_10X2_5_MESH_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244}"
DATA="${RSMOL_5_10X2_5_MESH_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset/data}"
OUTPUT="${RSMOL_5_10X2_5_MESH_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/${GATE,,}_continue_3081steps_fp32_from009244_$(date +%Y%m%d_%H%M%S)}"
if [[ "$GATE" == "FORMAL" && "$WORLD_SIZE" != "8" ]]; then echo "FORMAL requires WORLD_SIZE=8" >&2; exit 2; fi
if [[ "$GATE" == "FORMAL" ]]; then
  MICRO=8; GA=16; MAX_STEPS=3081; SCHEDULER=3081; WARMUP=155
  STEPS_PER_EPOCH=9244; EPOCHS=0.3333333333333333
  MAX_LR="${RSMOL_5_10X2_5_MESH_MAX_LR:-2e-4}"
  MIN_LR="${RSMOL_5_10X2_5_MESH_MIN_LR:-1e-5}"
else
  MICRO="${RSMOL_5_10X2_5_MESH_MICRO_BATCH_SIZE:-8}"; GA="${RSMOL_5_10X2_5_MESH_GRADIENT_ACCUMULATION_STEPS:-16}"; MAX_STEPS="${RSMOL_5_10X2_5_MESH_MAX_OPTIMIZER_STEPS:-10}"; SCHEDULER="${RSMOL_5_10X2_5_MESH_SCHEDULER_TOTAL_STEPS:-$MAX_STEPS}"; WARMUP="${RSMOL_5_10X2_5_MESH_WARMUP_STEPS:-1}"; MAX_LR="${RSMOL_5_10X2_5_MESH_MAX_LR:-8e-4}"; MIN_LR="${RSMOL_5_10X2_5_MESH_MIN_LR:-2e-5}"
  STEPS_PER_EPOCH="${RSMOL_5_10X2_5_MESH_STEPS_PER_EPOCH:-9244}"; EPOCHS="${RSMOL_5_10X2_5_MESH_EPOCHS:-1}"
fi
ARGS=(--gate "$GATE" --model-path "$MODEL" --data-dir "$DATA" --output-dir "$OUTPUT" --world-size "$WORLD_SIZE" --micro-batch-size "$MICRO" --gradient-accumulation-steps "$GA" --context-length 1024 --max-optimizer-steps "$MAX_STEPS" --scheduler-total-steps "$SCHEDULER" --warmup-steps "$WARMUP" --steps-per-epoch "$STEPS_PER_EPOCH" --epochs "$EPOCHS" --max-lr "$MAX_LR" --min-lr "$MIN_LR" --save-every 500 --seed "${RSMOL_5_10X2_5_MESH_SEED:-0}")
[[ -n "${RSMOL_5_10X2_5_MESH_TOKENIZER_PATH:-}" ]] && ARGS+=(--tokenizer-path "$RSMOL_5_10X2_5_MESH_TOKENIZER_PATH")
[[ -n "${RSMOL_5_10X2_5_MESH_RESUME_FROM:-}" ]] && ARGS+=(--resume-from "$RSMOL_5_10X2_5_MESH_RESUME_FROM")
echo "[stage4] gate=$GATE precision=FP32 epochs=$EPOCHS steps_per_epoch=$STEPS_PER_EPOCH total_steps=$MAX_STEPS warmup=$WARMUP lr=$MAX_LR->$MIN_LR output=$OUTPUT"
torchrun --standalone --nproc_per_node="$WORLD_SIZE" code/RSmol/scripts/train_stage4_5_10x2_5_mesh_ddp.py "${ARGS[@]}"
