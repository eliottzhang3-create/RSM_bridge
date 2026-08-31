#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
MODEL_DIR="${RSMOL_5_10_5_LINEAR_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5-linear}"
REPORT="${RSMOL_5_10_5_LINEAR_SMOKE_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage1_5_10_5_linear_$(date +%Y%m%d_%H%M%S).json}"
echo "========== SMOLLM2 5-10-5 LINEAR STAGE 1 SMOKE =========="
echo "ACTIVE_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "PYTHON=$(which python)"
echo "MODEL_DIR=$MODEL_DIR"
echo "REPORT=$REPORT"
echo "DEVICE=${RSMOL_5_10_5_LINEAR_DEVICE:-cuda:0}"
python -u code/RSmol/scripts/smoke_5_10_5_linear.py \
  --model-path "$MODEL_DIR" \
  --device "${RSMOL_5_10_5_LINEAR_DEVICE:-cuda:0}" \
  --output-report "$REPORT" \
  --max-new-tokens "${RSMOL_5_10_5_LINEAR_MAX_NEW_TOKENS:-4}"
