#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate rsmol
# Formal online-data configuration: 8 samples/GPU with GA=4 gives effective
# global batch 8 GPUs * 8 * 4 = 256.  Epochs and max/min learning rates are
# explicit trainer hyperparameters supplied by the caller; warmup is always
# derived as ceil(5% of the actual optimizer-step budget).
torchrun --standalone --nproc_per_node=8 "$SCRIPT_DIR/train_audio_5_10x2_5_mesh_mellow_ddp.py" --gate FORMAL --micro-batch-size 8 --gradient-accumulation-steps 4 --save-every 500 --checkpoint-retention 4 "$@"
