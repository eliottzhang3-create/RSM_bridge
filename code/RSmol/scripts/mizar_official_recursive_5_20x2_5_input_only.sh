#!/bin/bash
set -euo pipefail
CONDA_BASE="${MIZAR_520X2_INPUT_ONLY_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_BASE/envs/mellow_c8204d8"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
# No scheduler submission, GPU allocation, dataset access or training occurs here.
exec python -B -m mizar_official_recursive_5_20x2_5_input_only "$@"
