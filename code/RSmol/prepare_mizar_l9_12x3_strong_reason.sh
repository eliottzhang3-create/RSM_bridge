#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONDA_BASE=${RSMOL_CONDA_BASE:-${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u "$SCRIPT_DIR/mizar_l9_12x3_strong_reason/prepare_manifest.py" "$@"
