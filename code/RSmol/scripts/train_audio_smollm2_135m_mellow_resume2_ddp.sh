#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
SMOLLM2_BASE_OUTPUT="${SMOLLM2_BASE_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate rsmol
torchrun --standalone --nproc_per_node=8 "$SCRIPT_DIR/train_audio_smollm2_135m_mellow_ddp.py" \
  --gate STAGE7 --micro-batch-size 8 --gradient-accumulation-steps 4 \
  --max-steps 22 --schedule-total-steps 22 --expected-resume-step 20 \
  --save-every 20 --checkpoint-retention 2 \
  --output-dir "$SMOLLM2_BASE_OUTPUT/resume2_from20" "$@"
