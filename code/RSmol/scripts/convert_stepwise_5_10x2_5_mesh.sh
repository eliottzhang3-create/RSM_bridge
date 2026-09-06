#!/bin/bash
set -euo pipefail
USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SOURCE="${RSMOL_5_10X2_5_MESH_SOURCE_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5}"
OUTPUT="${RSMOL_5_10X2_5_MESH_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x2-5-mesh}"
ARGS=(--source-checkpoint "$SOURCE" --output-dir "$OUTPUT" --seed "${RSMOL_5_10X2_5_MESH_SEED:-0}")
[[ "${RSMOL_5_10X2_5_MESH_ALLOW_OVERWRITE:-0}" == "1" ]] && ARGS+=(--allow-overwrite)
[[ "${RSMOL_5_10X2_5_MESH_ALLOW_TRAINING_STATE:-0}" == "1" ]] && ARGS+=(--allow-training-state)
python code/RSmol/scripts/convert_stepwise_5_10x2_5_mesh.py "${ARGS[@]}"
