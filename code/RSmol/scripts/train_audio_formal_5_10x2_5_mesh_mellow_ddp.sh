#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Formal configuration: 8 samples/GPU with GA=4 gives effective global batch
# 8 GPUs * 8 * 4 = 256.  User-supplied arguments after these defaults may still
# override it when an intentional experiment needs a different accumulation.
torchrun --standalone --nproc_per_node=8 "$SCRIPT_DIR/train_audio_5_10x2_5_mesh_mellow_ddp.py" --gate FORMAL --micro-batch-size 8 --gradient-accumulation-steps 4 --save-every 1000 --checkpoint-retention 4 "$@"
