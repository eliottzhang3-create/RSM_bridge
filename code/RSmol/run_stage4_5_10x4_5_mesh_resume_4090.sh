#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${RSMOL_5_10X4_5_MESH_RESUME_FROM:-}" ]]; then
  echo "set RSMOL_5_10X4_5_MESH_RESUME_FROM to a complete checkpoint" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${RSMOL_5_10X4_5_MESH_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/resume2_20260927_$RUN_TAG}"
printf -v OUTPUT_Q '%q' "$OUTPUT"
printf -v RESUME_Q '%q' "$RSMOL_5_10X4_5_MESH_RESUME_FROM"
vc submit -p pdgpu-4090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "text-x4-resume-$RUN_TAG" -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/text_x4_resume.$RUN_TAG.JOB.log" \
  --cmd "RSMOL_5_10X4_5_MESH_OUTPUT_DIR=$OUTPUT_Q RSMOL_5_10X4_5_MESH_RESUME_FROM=$RESUME_Q RSMOL_5_10X4_5_MESH_STAGE4_GATE=E bash scripts/train_stage4_5_10x4_5_mesh_ddp.sh"
echo "resume output: $OUTPUT"
