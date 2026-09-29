#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p log
CMD_ARGS=""
if (($#)); then
  printf -v CMD_ARGS '%q ' "$@"
fi
JOB_TAG=audio-smollm2-mellow-official-v2-b32-probe-$(date +%m%d%H%M%S%N)

submit_args=(
  -p pdgpu-3090
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1
  -c 32
  -m 256G
  -g 8
  -n 1
  -j "$JOB_TAG"
  -d "$SCRIPT_DIR"
  JOB=1:1
  "$SCRIPT_DIR/log/$JOB_TAG.JOB.log"
  --cmd
  "bash scripts/train_audio_smollm2_mellow_official_faithful_v2_physical_batch32_probe_ddp.sh $CMD_ARGS"
)
vc submit "${submit_args[@]}"
