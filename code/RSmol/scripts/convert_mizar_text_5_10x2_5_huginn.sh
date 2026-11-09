#!/bin/bash
set -euo pipefail
# CPU/login node conversion; all model assets remain on the remote filesystem.
CONDA_BASE="${RSMOL_MIZAR_TEXT_5_10X2_5_HUGINN_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u "$SCRIPT_DIR/convert_mizar_text_5_10x2_5_huginn.py" \
  --source-root "${RSMOL_MIZAR_TEXT_5_10X2_5_HUGINN_SOURCE_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M}" \
  --output-dir "${RSMOL_MIZAR_TEXT_5_10X2_5_HUGINN_CONVERT_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-text-5-10x2-5-huginn}" \
  --threads 8 "$@"
