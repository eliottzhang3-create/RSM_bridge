#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate rsmol
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
torchrun --standalone --nproc_per_node=1 \
  scripts/preflight_audio_5_10x2to10_5_mesh_t10_activation.py "$@"
