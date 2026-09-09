#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
conda activate rsmol
torchrun --standalone --nproc_per_node=8 "$SCRIPT_DIR/train_audio_5_10x2_5_mesh_mellow_ddp.py" --gate STAGE5 "$@"
