#!/bin/bash
set -euo pipefail
USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Conversion is CPU-only, including when this wrapper is run on a GPU host.
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
SOURCE="${RSMOL_5_10X3_5_MESH_6SLOT_SOURCE_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5}"
ROUTER_SOURCE="${RSMOL_5_10X3_5_MESH_6SLOT_ROUTER_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh_6slot/formal_1epoch_fp32_20261031_124955519529617/checkpoint-009244}"
OUTPUT="${RSMOL_5_10X3_5_MESH_6SLOT_CONVERT_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x3-5-mesh-6slot}"
ARGS=(--source-checkpoint "$SOURCE" --router-checkpoint "$ROUTER_SOURCE" --output-dir "$OUTPUT" --seed "${RSMOL_5_10X3_5_MESH_6SLOT_SEED:-0}" --threads 8)
python code/RSmol/scripts/convert_stepwise_5_10x3_5_mesh_6slot.py "${ARGS[@]}" "$@"
