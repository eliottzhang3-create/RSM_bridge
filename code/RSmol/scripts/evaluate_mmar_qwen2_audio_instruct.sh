#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${RSMOL_QWEN2_AUDIO_PYTHON:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/rsmol/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Qwen2-Audio Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi
exec "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_mmar_qwen2_audio_instruct.py" "$@"
