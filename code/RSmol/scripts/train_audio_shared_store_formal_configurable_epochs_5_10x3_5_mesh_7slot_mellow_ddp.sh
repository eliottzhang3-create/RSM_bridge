#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_ID="${RSMOL_SHARED_TRAIN_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}"
EPOCHS_VALUE=""
OUTPUT_DIR_SEEN=0
arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
  case "${arguments[$index]}" in
    --epochs)
      if ((index + 1 >= ${#arguments[@]})); then
        echo "--epochs requires a positive integer" >&2
        exit 2
      fi
      EPOCHS_VALUE="${arguments[$((index + 1))]}"
      ;;
    --epochs=*) EPOCHS_VALUE="${arguments[$index]#--epochs=}" ;;
    --output-dir)
      OUTPUT_DIR_SEEN=1
      ;;
    --output-dir=*) OUTPUT_DIR_SEEN=1 ;;
  esac
done
if [[ ! "$EPOCHS_VALUE" =~ ^[1-9][0-9]*$ ]]; then
    echo "x3/7-slot formal training requires --epochs <positive integer>" >&2
  exit 2
fi
DEFAULT_OUTPUT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x3_5_mesh_7slot_mellow_shared_store_configurable_epochs/formal_${EPOCHS_VALUE}epochs_${RUN_ID}"
if [[ "$OUTPUT_DIR_SEEN" -eq 1 ]]; then
  exec bash "$SCRIPT_DIR/stage_audio_shared_store_configurable_epochs_5_10x3_5_mesh_7slot_mellow.sh" formal \
    --max-lr 1e-3 \
    --min-lr 1e-4 \
    --save-every 500 \
    --checkpoint-retention 4 \
    "$@"
else
  exec bash "$SCRIPT_DIR/stage_audio_shared_store_configurable_epochs_5_10x3_5_mesh_7slot_mellow.sh" formal \
    --output-dir "$DEFAULT_OUTPUT" \
    --max-lr 1e-3 \
    --min-lr 1e-4 \
    --save-every 500 \
    --checkpoint-retention 4 \
    "$@"
fi
