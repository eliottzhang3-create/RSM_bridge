#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
: "${RSMOL_RECURSIVE_OUTPUT_DIR:?Set RSMOL_RECURSIVE_OUTPUT_DIR to converted checkpoint}"
REPORT="${RSMOL_RECURSIVE_SMOKE_REPORT:-$REPO_ROOT/outputs/RSmol/recursive_smoke_$(date +%Y%m%d_%H%M%S).json}"
echo "========== RECURSIVE SMOKE JOB =========="
echo "ACTIVE_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "PYTHON=$(which python)"
echo "RECURSIVE_OUTPUT_DIR=$RSMOL_RECURSIVE_OUTPUT_DIR"
echo "REPORT=$REPORT"
echo "DEVICE=${RSMOL_RECURSIVE_DEVICE:-cuda:0}"
python -u code/RSmol/scripts/smoke_recursive.py \
  --model-path "$RSMOL_RECURSIVE_OUTPUT_DIR" \
  --device "${RSMOL_RECURSIVE_DEVICE:-cuda:0}" \
  --output-report "$REPORT" \
  --max-new-tokens "${RSMOL_RECURSIVE_MAX_NEW_TOKENS:-4}"
