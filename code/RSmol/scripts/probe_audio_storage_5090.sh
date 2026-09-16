#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE_RUN_ID="${RSMOL_STORAGE_PROBE_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}"
DEFAULT_OUTPUT_DIR="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_storage_probe/${PROBE_RUN_ID}"

python "$SCRIPT_DIR/probe_audio_storage.py" \
  --output-dir "$DEFAULT_OUTPUT_DIR" \
  --expected-cache-gib 115.30 \
  --minimum-local-free-gib 150 \
  "$@"
