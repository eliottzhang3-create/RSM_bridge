#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"
case "$MODE" in submit|reasonaqa|benchmark|mmsu|compare) shift ;; *) echo 'Expected submit/reasonaqa/benchmark/mmsu/compare' >&2; exit 2 ;; esac
RSMOL_CONDA_BASE="${RSMOL_CONDA_BASE:-${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}}"
source "$RSMOL_CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
cd "$(dirname "$SCRIPT_DIR")"
exec python -u -B -m "mizar_core3_eval.$MODE" "$@"
