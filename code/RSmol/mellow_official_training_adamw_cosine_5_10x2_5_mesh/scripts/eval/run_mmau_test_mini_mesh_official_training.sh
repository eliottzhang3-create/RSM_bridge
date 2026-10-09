#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
CONDA_BASE="${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${MELLOW_CONDA_ENV:-mellow_c8204d8}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$ROUTE_ROOT"
exec python -u scripts/eval/evaluate_mmau_test_mini_mesh_official_training.py "$@"
