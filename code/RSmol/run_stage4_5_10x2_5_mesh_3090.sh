#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
vc submit -p "${RSMOL_5_10X2_5_MESH_QUEUE:-pdgpu-3090}" -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 -c 32 -m 256G -g 8 -n 1 -j "stage4-5-10x2-5-mesh-$(date +%m%d%H%M)" -d "$SCRIPT_DIR" JOB=1:1 "$SCRIPT_DIR/log/stage4_5_10x2_5_mesh.JOB.log" --cmd "bash scripts/train_stage4_5_10x2_5_mesh_ddp.sh"
