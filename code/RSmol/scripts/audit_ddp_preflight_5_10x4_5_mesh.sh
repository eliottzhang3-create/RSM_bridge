#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 <x2|x4>" >&2
  exit 2
fi
VARIANT="$1"
case "$VARIANT" in
  x2) DEFAULT_MODEL=/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x2-5-mesh ;;
  x4) DEFAULT_MODEL=/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x4-5-mesh ;;
  *) echo "invalid variant: $VARIANT" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
cd "$REPO_ROOT"

MODEL="${RSMOL_5_10X4_5_MESH_DDP_PREFLIGHT_MODEL:-$DEFAULT_MODEL}"
OUTPUT="${RSMOL_5_10X4_5_MESH_DDP_PREFLIGHT_OUTPUT:?set preflight output directory}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"

torchrun --standalone --nproc_per_node=8 \
  code/RSmol/scripts/audit_ddp_preflight_5_10x4_5_mesh.py \
  --variant "$VARIANT" \
  --model-path "$MODEL" \
  --output-dir "$OUTPUT" \
  --world-size 8 \
  --timeout-seconds 1200 \
  --file-rendezvous-seconds 300 \
  --broadcast-mib 64
