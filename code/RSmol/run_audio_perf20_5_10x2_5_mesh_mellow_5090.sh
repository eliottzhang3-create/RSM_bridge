#!/bin/bash
set -euo pipefail

# Independent 8x5090 PERF20 launcher. Pass --preload-data for the rank-local
# in-memory control, --waveform-cache-dir for the fixed-shard mmap control,
# and --profiler for profiling. Output directories are generated uniquely by
# the inner script unless --output-dir is supplied.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_ARGS=""
if (($#)); then
  printf -v CMD_ARGS '%q ' "$@"
fi
JOB_TAG="audio-mesh-perf20-5090-$(date +%m%d%H%M%S%N)"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "$JOB_TAG" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/${JOB_TAG}.JOB.log" \
  --cmd "bash scripts/train_audio_perf20_5_10x2_5_mesh_mellow_ddp.sh $CMD_ARGS"
