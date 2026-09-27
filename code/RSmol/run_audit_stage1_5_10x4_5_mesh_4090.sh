#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
MODEL="${RSMOL_5_10X4_5_MESH_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x4-5-mesh}"
REPORT="${RSMOL_5_10X4_5_MESH_STAGE1_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/preflight/stage1_audit.json}"
printf -v MODEL_Q '%q' "$MODEL"
printf -v REPORT_Q '%q' "$REPORT"
vc submit -p pdgpu-4090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "audit-text-x4-$RUN_TAG" -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/audit_text_x4.$RUN_TAG.JOB.log" \
  --cmd "RSMOL_5_10X4_5_MESH_MODEL_DIR=$MODEL_Q RSMOL_5_10X4_5_MESH_STAGE1_REPORT=$REPORT_Q bash scripts/audit_stage1_5_10x4_5_mesh.sh"
