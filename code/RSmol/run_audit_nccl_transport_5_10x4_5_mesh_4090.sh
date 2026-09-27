#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 <baseline|p2p_off|p2p_cumem_off>" >&2
  exit 2
fi
PROFILE="$1"
case "$PROFILE" in
  baseline|p2p_off|p2p_cumem_off) ;;
  *) echo "invalid profile: $PROFILE" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${RSMOL_5_10X4_5_MESH_NCCL_AUDIT_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/nccl_transport/${PROFILE}_20260927_$RUN_TAG}"
printf -v OUTPUT_Q '%q' "$OUTPUT"

vc submit -p pdgpu-4090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 64G -g 8 -n 1 \
  -j "nccl-transport-$PROFILE-$RUN_TAG" -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/nccl_transport_${PROFILE}.$RUN_TAG.JOB.log" \
  --cmd "RSMOL_5_10X4_5_MESH_NCCL_AUDIT_OUTPUT=$OUTPUT_Q bash scripts/audit_nccl_transport_5_10x4_5_mesh.sh $PROFILE"

echo "NCCL transport profile: $PROFILE"
echo "NCCL transport output: $OUTPUT"
