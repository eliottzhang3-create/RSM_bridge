#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_ID="${RSMOL_SMOLLM2_SHARED_TRAIN_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}"
EPOCHS_VALUE=""
SMOKE_SEEN=0
REFERENCE_SEEN=0
RESUME_SEEN=0
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
    --smoke20-report|--smoke20-report=*) SMOKE_SEEN=1 ;;
    --reference22-report|--reference22-report=*) REFERENCE_SEEN=1 ;;
    --smoke-resume-report|--smoke-resume-report=*) RESUME_SEEN=1 ;;
  esac
done
if [[ "$EPOCHS_VALUE" != "30" ]]; then
  echo "Mellow-faithful formal training requires --epochs 30" >&2
  exit 2
fi
if [[ "$SMOKE_SEEN" -ne 1 || "$REFERENCE_SEEN" -ne 1 || "$RESUME_SEEN" -ne 1 ]]; then
  echo "formal requires --smoke20-report, --reference22-report, and --smoke-resume-report" >&2
  exit 2
fi
DEFAULT_OUTPUT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow_shared_store_configurable_epochs/formal_${EPOCHS_VALUE}epochs_${RUN_ID}"
exec bash "$SCRIPT_DIR/stage_audio_smollm2_shared_store_configurable_epochs_135m_mellow.sh" formal \
  --output-dir "$DEFAULT_OUTPUT" \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --micro-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --num-workers 0 \
  --save-every-epochs 1 \
  --checkpoint-retention 3 \
  "$@"
