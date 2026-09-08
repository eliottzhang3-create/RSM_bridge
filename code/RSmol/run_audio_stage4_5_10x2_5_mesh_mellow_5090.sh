#!/bin/bash
set -euo pipefail

# Submitted single-GPU Stage 4 forward/backward audit.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_ARGS=""
if (($#)); then
  printf -v CMD_ARGS '%q ' "$@"
fi

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j audio-mesh-stage4-5090-$(date +%m%d%H%M%S) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/audio_mesh_stage4_5090.JOB.log" \
  --cmd "bash scripts/audit_audio_stage4_5_10x2_5_mesh_mellow.sh $CMD_ARGS"
