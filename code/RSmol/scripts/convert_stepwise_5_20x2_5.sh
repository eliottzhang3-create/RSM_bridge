#!/bin/bash
set -euo pipefail

# Run directly on the remote CPU/login node; no GPU job is submitted.
CONDA_BASE="${RSMOL_5_20X2_5_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u "$SCRIPT_DIR/convert_stepwise_5_20x2_5.py" \
  --source-checkpoint "${RSMOL_5_20X2_5_SOURCE_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2}" \
  --output-dir "${RSMOL_5_20X2_5_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-20x2-5}" \
  --threads 8 "$@"
