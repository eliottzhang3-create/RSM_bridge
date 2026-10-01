#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="${RSMOL_5_10X5_5_MESH_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x5-5-mesh}"
REPORT="${RSMOL_5_10X5_5_MESH_STAGE1_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x5_5_mesh/preflight/stage1_audit.json}"
python code/RSmol/scripts/audit_stage1_5_10x5_5_mesh.py \
  --model-path "$MODEL" \
  --report-path "$REPORT" \
  "$@"
