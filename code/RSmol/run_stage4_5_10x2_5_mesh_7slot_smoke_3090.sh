#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${RSMOL_5_10X2_5_MESH_7SLOT_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh_7slot/smoke10_3090_$RUN_TAG}"
printf -v OUTPUT_Q '%q' "$OUTPUT"
vc submit -p pdgpu-3090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "text-x2-7s-smoke-$RUN_TAG" -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/text_x2_7s_smoke_3090.$RUN_TAG.JOB.log" \
  --cmd "RSMOL_5_10X2_5_MESH_7SLOT_OUTPUT_DIR=$OUTPUT_Q RSMOL_5_10X2_5_MESH_7SLOT_STAGE4_GATE=D bash scripts/train_stage4_5_10x2_5_mesh_7slot_ddp.sh"
echo "smoke output: $OUTPUT"
