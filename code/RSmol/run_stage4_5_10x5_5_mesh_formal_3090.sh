#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
# This route intentionally has no smoke/resume gate wrapper. It submits the
# one-third-epoch FORMAL contract directly from the converted x5 model.
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RESUME_FROM="${RSMOL_5_10X5_5_MESH_RESUME_FROM:-}"
if [[ -n "$RESUME_FROM" ]]; then
  if [[ -z "${RSMOL_5_10X5_5_MESH_OUTPUT_DIR:-}" ]]; then
    echo "formal resume requires a new RSMOL_5_10X5_5_MESH_OUTPUT_DIR" >&2
    exit 2
  fi
  if [[ ! -d "$RESUME_FROM" ]]; then
    echo "formal resume checkpoint directory does not exist: $RESUME_FROM" >&2
    exit 2
  fi
  for REQUIRED_FILE in checkpoint_complete.json checkpoint_manifest.json training_state.pt; do
    if [[ ! -f "$RESUME_FROM/$REQUIRED_FILE" ]]; then
      echo "formal resume checkpoint is incomplete; missing $RESUME_FROM/$REQUIRED_FILE" >&2
      exit 2
    fi
  done
fi
OUTPUT="${RSMOL_5_10X5_5_MESH_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x5_5_mesh/formal_third_epoch_3081steps_${RUN_TAG}_3090_v1}"
if [[ -n "$RESUME_FROM" && -e "$OUTPUT" ]]; then
  echo "formal resume output path must be new and absent: $OUTPUT" >&2
  exit 2
fi
printf -v OUTPUT_Q '%q' "$OUTPUT"
RESUME_ENV=""
if [[ -n "$RESUME_FROM" ]]; then
  printf -v RESUME_Q '%q' "$RESUME_FROM"
  RESUME_ENV=" RSMOL_5_10X5_5_MESH_RESUME_FROM=$RESUME_Q"
fi
vc submit -p pdgpu-3090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "text-x5-formal-3090-$RUN_TAG" -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/text_x5_formal_3090.$RUN_TAG.JOB.log" \
  --cmd "RSMOL_5_10X5_5_MESH_OUTPUT_DIR=$OUTPUT_Q$RESUME_ENV RSMOL_5_10X5_5_MESH_STAGE4_GATE=FORMAL bash scripts/train_stage4_5_10x5_5_mesh_ddp.sh"
echo "formal output: $OUTPUT"
if [[ -n "$RESUME_FROM" ]]; then
  echo "formal resume source: $RESUME_FROM"
fi
