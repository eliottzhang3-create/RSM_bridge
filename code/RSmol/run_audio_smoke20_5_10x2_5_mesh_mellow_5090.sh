#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
CMD_ARGS=""
if (($#)); then
  printf -v CMD_ARGS '%q ' "$@"
fi
vc submit -p pdgpu-5090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 -c 32 -m 256G -g 8 -n 1 -j audio-mesh-smoke20-5090-$(date +%m%d%H%M%S) -d "$SCRIPT_DIR" JOB=1:1 "$SCRIPT_DIR/log/audio_mesh_smoke20_5090.JOB.log" --cmd "bash scripts/train_audio_smoke20_5_10x2_5_mesh_mellow_ddp.sh $CMD_ARGS"
