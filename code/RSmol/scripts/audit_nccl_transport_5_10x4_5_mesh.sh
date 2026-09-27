#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 <baseline|p2p_off|p2p_cumem_off>" >&2
  exit 2
fi
PROFILE="$1"
case "$PROFILE" in
  baseline)
    unset NCCL_P2P_DISABLE NCCL_CUMEM_ENABLE
    ;;
  p2p_off)
    export NCCL_P2P_DISABLE=1
    unset NCCL_CUMEM_ENABLE
    ;;
  p2p_cumem_off)
    export NCCL_P2P_DISABLE=1
    export NCCL_CUMEM_ENABLE=0
    ;;
  *)
    echo "invalid profile: $PROFILE" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
cd "$REPO_ROOT"

OUTPUT="${RSMOL_5_10X4_5_MESH_NCCL_AUDIT_OUTPUT:?set NCCL audit output directory}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,GRAPH,COLL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-300}"

echo "PROFILE=$PROFILE NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset} NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-unset}"
torchrun --standalone --nproc_per_node=8 \
  code/RSmol/scripts/audit_nccl_transport_5_10x4_5_mesh.py \
  --profile "$PROFILE" \
  --output-dir "$OUTPUT" \
  --world-size 8 \
  --timeout-seconds 300 \
  --file-rendezvous-seconds 120 \
  --broadcast-mib 64
