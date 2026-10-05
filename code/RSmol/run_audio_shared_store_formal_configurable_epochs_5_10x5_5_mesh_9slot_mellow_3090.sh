#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
CMD_ARGS=""
if (($#)); then printf -v CMD_ARGS '%q ' "$@"; fi
JOB_TAG="aud-x5-9s-f-$(date +%m%d%H%M%S%N)"
vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "$JOB_TAG" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/${JOB_TAG}.JOB.log" \
  --cmd "bash scripts/train_audio_shared_store_formal_configurable_epochs_5_10x5_5_mesh_9slot_mellow_ddp.sh $CMD_ARGS"
