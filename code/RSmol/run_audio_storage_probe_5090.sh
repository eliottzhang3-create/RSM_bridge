#!/bin/bash
set -euo pipefail

# Match the formal 8x5090 job's node, image, CPU, memory, and GPU request so
# mount/cgroup discovery is performed in the relevant container environment.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_ARGS=""
if (($#)); then
  printf -v CMD_ARGS '%q ' "$@"
fi
JOB_TAG="audio-storage-probe-5090-$(date +%m%d%H%M%S%N)"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "$JOB_TAG" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/${JOB_TAG}.JOB.log" \
  --cmd "bash scripts/probe_audio_storage_5090.sh $CMD_ARGS"
