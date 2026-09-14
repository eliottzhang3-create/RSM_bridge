#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
RECURSIVE_AUDIO_BASE_OUTPUT="${RECURSIVE_AUDIO_BASE_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_recursive_mellow}"
RUN_STAMP="${RECURSIVE_AUDIO_RUN_STAMP:-$(date +%Y%m%d_%H%M%S_%N)}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate rsmol
torchrun --standalone --nproc_per_node=8 "$SCRIPT_DIR/train_audio_5_10_5_recursive_mellow_ddp.py" \
  --gate FORMAL \
  --model-path /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244 \
  --train-manifest /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl \
  --val-manifest /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_val.jsonl \
  --micro-batch-size 8 --gradient-accumulation-steps 4 \
  --epochs 3 --max-lr 1e-3 --min-lr 0 \
  --save-every 500 --checkpoint-retention 4 \
  --output-dir "$RECURSIVE_AUDIO_BASE_OUTPUT/formal_${RUN_STAMP}" "$@"
