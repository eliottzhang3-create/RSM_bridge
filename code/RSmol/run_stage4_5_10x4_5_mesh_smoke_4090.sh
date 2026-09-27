#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${RSMOL_5_10X4_5_MESH_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/smoke10_20260927_$RUN_TAG}"
printf -v OUTPUT_Q '%q' "$OUTPUT"
vc submit -p pdgpu-4090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "text-x4-smoke-$RUN_TAG" -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/text_x4_smoke.$RUN_TAG.JOB.log" \
  --cmd "RSMOL_5_10X4_5_MESH_OUTPUT_DIR=$OUTPUT_Q bash scripts/stage_text_shared_store_5_10x4_5_mesh.sh smoke"
echo "smoke output: $OUTPUT"
