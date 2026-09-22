#!/bin/bash
set -euo pipefail

# Independent 8x5090 PERF20 launcher. Select one strict causal input control
# with --perf20-input-mode: online, warm_online, waveform_preload,
# full_preload, shared_waveform_store, shared_waveform_store_tmpfs, or
# partition_rank_ram_preload.  The tmpfs mode stages the complete v3 store in
# node-shared /dev/shm before timing.  The last mode selects a materialized component partition with
# --perf20-partition-id (default 0) and copies that complete partition into
# every rank's CPU RAM before timing.  Pass --profiler only when an operator
# trace is explicitly needed.
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
