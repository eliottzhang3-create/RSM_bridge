#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate rsmol
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
torchrun --standalone --nproc_per_node=8 \
  "$SCRIPT_DIR/train_audio_partitioned_5_10x2_5_mesh_mellow_silence_slot_ddp.py" \
  --mode formal --epochs 10 --micro-batch-size 8 --gradient-accumulation-steps 4 \
  --save-every 500 --checkpoint-retention 4 "$@"
