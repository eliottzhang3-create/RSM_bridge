#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${RSMOL_5_10X4_5_MESH_RESUME_FROM:-}" ]]; then
  echo "set RSMOL_5_10X4_5_MESH_RESUME_FROM to a complete checkpoint" >&2
  exit 2
fi
RESUME_FROM="$RSMOL_5_10X4_5_MESH_RESUME_FROM"
if [[ ! -d "$RESUME_FROM" ]]; then
  echo "resume checkpoint directory does not exist: $RESUME_FROM" >&2
  exit 2
fi
for REQUIRED_FILE in checkpoint_complete.json training_state.pt; do
  if [[ ! -f "$RESUME_FROM/$REQUIRED_FILE" ]]; then
    echo "resume checkpoint is incomplete; missing $RESUME_FROM/$REQUIRED_FILE" >&2
    exit 2
  fi
done
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${RSMOL_5_10X4_5_MESH_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/resume2_20260927_3090_$RUN_TAG}"
printf -v OUTPUT_Q '%q' "$OUTPUT"
printf -v RESUME_Q '%q' "$RESUME_FROM"
vc submit -p pdgpu-3090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "text-x4-resume-3090-$RUN_TAG" -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/text_x4_resume_3090.$RUN_TAG.JOB.log" \
  --cmd "RSMOL_5_10X4_5_MESH_OUTPUT_DIR=$OUTPUT_Q RSMOL_5_10X4_5_MESH_RESUME_FROM=$RESUME_Q bash scripts/stage_text_shared_store_5_10x4_5_mesh.sh resume"
echo "resume output: $OUTPUT"
