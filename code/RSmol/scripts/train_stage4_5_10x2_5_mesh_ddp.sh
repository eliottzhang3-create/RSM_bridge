#!/bin/bash
set -euo pipefail
USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
GATE="${RSMOL_5_10X2_5_MESH_STAGE4_GATE:-D}"
WORLD_SIZE="${RSMOL_5_10X2_5_MESH_WORLD_SIZE:-8}"
MODEL="${RSMOL_5_10X2_5_MESH_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x2-5-mesh}"
DATA="${RSMOL_5_10X2_5_MESH_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset/data}"
OUTPUT="${RSMOL_5_10X2_5_MESH_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/$(date +%Y%m%d_%H%M%S)}"
if [[ "$GATE" == "FORMAL" && "$WORLD_SIZE" != "8" ]]; then echo "FORMAL requires WORLD_SIZE=8" >&2; exit 2; fi
if [[ "$GATE" == "FORMAL" ]]; then
  MICRO=8; GA=16; MAX_STEPS=9244; SCHEDULER=9244; WARMUP=463; MAX_LR=8e-4; MIN_LR=8e-5
else
  MICRO="${RSMOL_5_10X2_5_MESH_MICRO_BATCH_SIZE:-8}"; GA="${RSMOL_5_10X2_5_MESH_GRADIENT_ACCUMULATION_STEPS:-16}"; MAX_STEPS="${RSMOL_5_10X2_5_MESH_MAX_OPTIMIZER_STEPS:-10}"; SCHEDULER="${RSMOL_5_10X2_5_MESH_SCHEDULER_TOTAL_STEPS:-$MAX_STEPS}"; WARMUP="${RSMOL_5_10X2_5_MESH_WARMUP_STEPS:-1}"; MAX_LR="${RSMOL_5_10X2_5_MESH_MAX_LR:-8e-4}"; MIN_LR="${RSMOL_5_10X2_5_MESH_MIN_LR:-8e-5}"
fi
ARGS=(--gate "$GATE" --model-path "$MODEL" --data-dir "$DATA" --output-dir "$OUTPUT" --world-size "$WORLD_SIZE" --micro-batch-size "$MICRO" --gradient-accumulation-steps "$GA" --context-length 1024 --max-optimizer-steps "$MAX_STEPS" --scheduler-total-steps "$SCHEDULER" --warmup-steps "$WARMUP" --max-lr "$MAX_LR" --min-lr "$MIN_LR" --save-every 500 --seed "${RSMOL_5_10X2_5_MESH_SEED:-0}")
[[ -n "${RSMOL_5_10X2_5_MESH_TOKENIZER_PATH:-}" ]] && ARGS+=(--tokenizer-path "$RSMOL_5_10X2_5_MESH_TOKENIZER_PATH")
[[ -n "${RSMOL_5_10X2_5_MESH_RESUME_FROM:-}" ]] && ARGS+=(--resume-from "$RSMOL_5_10X2_5_MESH_RESUME_FROM")
torchrun --standalone --nproc_per_node="$WORLD_SIZE" code/RSmol/scripts/train_stage4_5_10x2_5_mesh_ddp.py "${ARGS[@]}"
