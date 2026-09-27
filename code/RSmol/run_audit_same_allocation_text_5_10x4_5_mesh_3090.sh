#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${RSMOL_5_10X4_5_MESH_SAME_ALLOCATION_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/same_allocation_3090_20260927_$RUN_TAG}"
printf -v OUTPUT_Q '%q' "$OUTPUT"

vc submit -p pdgpu-3090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 128G -g 8 -n 1 \
  -j "text-same-allocation-3090-$RUN_TAG" -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/text_same_allocation_3090.$RUN_TAG.JOB.log" \
  --cmd "RSMOL_5_10X4_5_MESH_SAME_ALLOCATION_OUTPUT=$OUTPUT_Q bash scripts/audit_same_allocation_text_5_10x4_5_mesh.sh"

echo "Same-allocation audit queue: pdgpu-3090"
echo "Same-allocation audit output: $OUTPUT"
