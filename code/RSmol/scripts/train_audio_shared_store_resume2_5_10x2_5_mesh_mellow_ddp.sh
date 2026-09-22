#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_ID="${RSMOL_SHARED_TRAIN_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}"
DEFAULT_OUTPUT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/resume2_${RUN_ID}"
RESUME_SEEN=0
for argument in "$@"; do
  case "$argument" in
    --resume-from|--resume-from=*) RESUME_SEEN=1 ;;
  esac
done
if [[ "$RESUME_SEEN" -ne 1 ]]; then
  echo "resume2 requires --resume-from <shared-store checkpoint-000020>" >&2
  exit 2
fi
exec bash "$SCRIPT_DIR/stage_audio_shared_store_5_10x2_5_mesh_mellow.sh" smoke \
  --output-dir "$DEFAULT_OUTPUT" \
  --max-lr 1e-3 \
  --min-lr 0 \
  --save-every 500 \
  --checkpoint-retention 4 \
  "$@"

