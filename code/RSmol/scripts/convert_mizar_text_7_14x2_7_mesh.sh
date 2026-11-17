#!/bin/bash
set -euo pipefail
# CPU/login node conversion; all model assets remain on the remote filesystem.
CONDA_BASE="${RSMOL_MIZAR_TEXT_7_14X2_7_MESH_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u "$SCRIPT_DIR/convert_mizar_text_7_14x2_7_mesh.py" \
  --source-root "${RSMOL_MIZAR_TEXT_7_14X2_7_MESH_SOURCE_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M}" \
  --output-dir "${RSMOL_MIZAR_TEXT_7_14X2_7_MESH_CONVERT_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-text-7-14x2-7-mesh-5slot}" \
  --threads 8 "$@"
