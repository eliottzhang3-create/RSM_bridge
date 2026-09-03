#!/bin/bash
set -euo pipefail
USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
MODEL_DIR="${RSMOL_5_10XPOISSON_PARCAE_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10xpoisson-parcae}"
REPORT="${RSMOL_5_10XPOISSON_PARCAE_STAGE1_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage1_5_10xpoisson_parcae_$(date +%Y%m%d_%H%M%S).json}"
python -u code/RSmol/scripts/audit_stage1_5_10xpoisson_parcae.py --model-path "$MODEL_DIR" --output-path "$REPORT" --device "${RSMOL_5_10XPOISSON_PARCAE_DEVICE:-cuda}"
