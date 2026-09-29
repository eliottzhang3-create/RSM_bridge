#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RUN_ID=${RSMOL_PHYSICAL_BATCH32_PROBE_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}
DEFAULT_OUTPUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow_official_faithful_v2_physical_batch32_probe/probe_$RUN_ID

exec bash "$SCRIPT_DIR/stage_audio_smollm2_mellow_official_faithful_v2_physical_batch32_probe.sh" +  --output-dir "$DEFAULT_OUTPUT" +  --epochs 30 +  --world-size 8 +  --learning-rate 1e-3 +  --weight-decay 1e-4 +  --micro-batch-size 32 +  --gradient-accumulation-steps 1 +  --num-workers 0 +  --save-every-steps 5000 +  --checkpoint-retention 4 +  "$@"
