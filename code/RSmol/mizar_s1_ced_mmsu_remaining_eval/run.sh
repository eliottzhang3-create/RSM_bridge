#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSMOL_CONDA_BASE="${RSMOL_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$RSMOL_CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
cd "$(dirname "$SCRIPT_DIR")"
MODE="${1:-submit}"
if (($#)); then shift; fi
case "$MODE" in
  submit|inspect) exec python -u -B -m mizar_s1_ced_mmsu_remaining_eval.submit --mode "$MODE" "$@" ;;
  compare) exec python -u -B -m mizar_s1_ced_mmsu_remaining_eval.compare "$@" ;;
  *) echo "Usage: run.sh {submit|inspect|compare} [OPTIONS]" >&2; exit 2 ;;
esac
