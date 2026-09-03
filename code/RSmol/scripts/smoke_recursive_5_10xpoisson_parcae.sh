#!/bin/bash
set -euo pipefail
USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
MODEL_DIR="${RSMOL_5_10XPOISSON_PARCAE_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10xpoisson-parcae}"
DATA_DIR="${RSMOL_5_10XPOISSON_PARCAE_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset}"
OUTPUT_DIR="${RSMOL_5_10XPOISSON_PARCAE_SMOKE_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_10step_5_10xpoisson_parcae/$(date +%Y%m%d_%H%M%S)}"
WORLD_SIZE="${RSMOL_5_10XPOISSON_PARCAE_WORLD_SIZE:-8}"
if [[ "$WORLD_SIZE" != "8" ]]; then
  echo "5_10xpoisson_parcae Gate D requires exactly 8 ranks; got WORLD_SIZE=$WORLD_SIZE" >&2
  exit 2
fi
# Gate D is a real distributed smoke: every rank owns an independent depth
# vector/local Tmax and the last accumulation microbatch performs the DDP
# all-reduce.  Keep this wrapper on the same torchrun path as formal Stage 4.
torchrun --standalone --nproc_per_node="$WORLD_SIZE" code/RSmol/scripts/smoke_recursive_5_10xpoisson_parcae.py \
  --model-path "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" \
  --device "${RSMOL_5_10XPOISSON_PARCAE_DEVICE:-cuda}" --seed "${RSMOL_5_10XPOISSON_PARCAE_SEED:-0}"
