#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_ID="${RSMOL_VARIABLE_DEPTH_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}"
DEFAULT_OUTPUT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2to10_5_mesh_mellow_shared_store/smoke20_7epochs_${RUN_ID}"
exec bash "$SCRIPT_DIR/stage_audio_shared_store_5_10x2to10_5_mesh_mellow.sh" smoke \
  --output-dir "$DEFAULT_OUTPUT" \
  --epochs 7 \
  --max-lr 1e-3 \
  --min-lr 1e-4 \
  --save-every 500 \
  --checkpoint-retention 4 \
  "$@"
