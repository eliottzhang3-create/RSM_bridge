#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RSMOL_ROOT=$(cd "$EVAL_ROOT/.." && pwd)
CONDA_BASE="${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${MELLOW_CONDA_ENV:-mellow_c8204d8}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NVIDIA_TF32_OVERRIDE=0
cd "$RSMOL_ROOT"
exec python -u "$EVAL_ROOT/evaluate.py" "$@"
