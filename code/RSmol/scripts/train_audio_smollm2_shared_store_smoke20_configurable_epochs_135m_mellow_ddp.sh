#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_ID="${RSMOL_SMOLLM2_SHARED_TRAIN_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}"
EPOCHS_VALUE=""
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
  esac
done
if [[ ! "$EPOCHS_VALUE" =~ ^[1-9][0-9]*$ ]]; then
  echo "configurable SmolLM2 shared-store smoke20 requires --epochs <positive integer>" >&2
  exit 2
fi
DEFAULT_OUTPUT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow_shared_store_configurable_epochs/smoke20_${EPOCHS_VALUE}epochs_${RUN_ID}"
exec bash "$SCRIPT_DIR/stage_audio_smollm2_shared_store_configurable_epochs_135m_mellow.sh" smoke \
  --output-dir "$DEFAULT_OUTPUT" \
  --max-lr 1e-3 \
  --min-lr 1e-4 \
  --save-every 500 \
  --checkpoint-retention 4 \
  "$@"

