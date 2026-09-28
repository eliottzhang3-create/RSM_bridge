#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_ID="${RSMOL_SMOLLM2_MELLOW_OFFICIAL_FAITHFUL_V2_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}"
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
if [[ "$EPOCHS_VALUE" != "30" ]]; then
  echo "Mellow official-faithful-v2 smoke20 requires --epochs 30" >&2
  exit 2
fi
DEFAULT_OUTPUT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow_official_faithful_v2_shared_store_configurable_epochs/smoke20_${EPOCHS_VALUE}epochs_${RUN_ID}"
exec bash "$SCRIPT_DIR/stage_audio_smollm2_shared_store_configurable_epochs_135m_mellow_official_faithful_v2.sh" smoke \
  --output-dir "$DEFAULT_OUTPUT" \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --micro-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --num-workers 0 \
  --save-every-steps 5000 \
  --checkpoint-retention 4 \
  "$@"
