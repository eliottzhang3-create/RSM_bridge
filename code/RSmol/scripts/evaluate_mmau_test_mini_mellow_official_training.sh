#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CONDA_BASE="${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8
cd "$ROOT"
exec python -u scripts/evaluate_mmau_test_mini_mellow_official_training.py "$@"
