#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"
case "$MODE" in submit|evaluate|prepare|plot) shift ;; *) echo 'Expected submit/evaluate/prepare/plot' >&2; exit 2 ;; esac
RSMOL_CONDA_BASE="${RSMOL_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$RSMOL_CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$(dirname "$SCRIPT_DIR")"
exec python -u -B -m "mizar_benchmark_tracing.$MODE" "$@"
