#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSMOL_CONDA_BASE="${RSMOL_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$RSMOL_CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$(dirname "$SCRIPT_DIR")"

exec python -u scripts/evaluate_mizar_s1_usad2_recursive_mesh_benchmark.py "$@"
