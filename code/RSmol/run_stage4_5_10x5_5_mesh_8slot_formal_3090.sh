#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
# This route intentionally has no smoke/resume gate wrapper. It submits the
# one-third-epoch FORMAL contract directly from the converted x5 model.
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${RSMOL_5_10X5_5_MESH_8SLOT_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x5_5_mesh_8slot/formal_third_epoch_3081steps_${RUN_TAG}_3090_v1}"
printf -v OUTPUT_Q '%q' "$OUTPUT"
vc submit -p pdgpu-3090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "text-x5-8slot-formal-3090-$RUN_TAG" -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/text_x5_8slot_formal_3090.$RUN_TAG.JOB.log" \
  --cmd "RSMOL_5_10X5_5_MESH_8SLOT_OUTPUT_DIR=$OUTPUT_Q RSMOL_5_10X5_5_MESH_8SLOT_STAGE4_GATE=FORMAL bash scripts/train_stage4_5_10x5_5_mesh_8slot_ddp.sh"
echo "formal output: $OUTPUT"
