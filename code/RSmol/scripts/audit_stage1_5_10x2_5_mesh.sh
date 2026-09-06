#!/bin/bash
set -euo pipefail
USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODEL="${RSMOL_5_10X2_5_MESH_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x2-5-mesh}"
TOKENIZER="${RSMOL_5_10X2_5_MESH_TOKENIZER_PATH:-}"
REPORT="${RSMOL_5_10X2_5_MESH_STAGE1_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage1_5_10x2_5_mesh/report.json}"
ARGS=(--model-path "$MODEL" --report-path "$REPORT")
[[ -n "$TOKENIZER" ]] && ARGS+=(--tokenizer-path "$TOKENIZER")
python code/RSmol/scripts/audit_stage1_5_10x2_5_mesh.py "${ARGS[@]}"
