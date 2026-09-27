#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 <x2|x4>" >&2
  exit 2
fi
VARIANT="$1"
case "$VARIANT" in
  x2|x4) ;;
  *) echo "invalid variant: $VARIANT" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${RSMOL_5_10X4_5_MESH_DDP_PREFLIGHT_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/ddp_preflight/${VARIANT}_20260927_$RUN_TAG}"
printf -v OUTPUT_Q '%q' "$OUTPUT"
MODEL_ENV=""
if [[ -n "${RSMOL_5_10X4_5_MESH_DDP_PREFLIGHT_MODEL:-}" ]]; then
  printf -v MODEL_Q '%q' "$RSMOL_5_10X4_5_MESH_DDP_PREFLIGHT_MODEL"
  MODEL_ENV="RSMOL_5_10X4_5_MESH_DDP_PREFLIGHT_MODEL=$MODEL_Q "
fi

vc submit -p pdgpu-4090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 128G -g 8 -n 1 \
  -j "text-ddp-preflight-$VARIANT-$RUN_TAG" -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/text_ddp_preflight_${VARIANT}.$RUN_TAG.JOB.log" \
  --cmd "${MODEL_ENV}RSMOL_5_10X4_5_MESH_DDP_PREFLIGHT_OUTPUT=$OUTPUT_Q bash scripts/audit_ddp_preflight_5_10x4_5_mesh.sh $VARIANT"

echo "DDP preflight variant: $VARIANT"
echo "DDP preflight output: $OUTPUT"
