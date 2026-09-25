#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
CMD_ARGS=""
if (($#)); then printf -v CMD_ARGS '%q ' "$@"; fi
JOB_TAG="audio-mesh-t2to10-memory-$(date +%m%d%H%M%S%N)"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "$JOB_TAG" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/${JOB_TAG}.JOB.log" \
  --cmd "bash scripts/preflight_audio_5_10x2to10_5_mesh_t10_activation.sh $CMD_ARGS"
