#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"
case "$MODE" in submit|benchmark) shift ;; *) echo 'Expected submit/benchmark' >&2; exit 2 ;; esac
CONDA_BASE="${RSMOL_CONDA_BASE:-${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
cd "$(dirname "$SCRIPT_DIR")"
exec python -u -B -m "mizar_s3_full_finetune_eval.$MODE" "$@"
