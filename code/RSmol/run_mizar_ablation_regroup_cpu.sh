#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSMOL_CONDA_BASE="${RSMOL_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$RSMOL_CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8
export CUDA_VISIBLE_DEVICES=""
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
cd "$SCRIPT_DIR"
exec python -u -B -m mizar_ablation_regroup.regroup "$@"
