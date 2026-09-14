#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate rsmol

# PERF20 is intentionally isolated from STAGE5/STAGE7/FORMAL.  The default
# output name is unique per launch; the Python gate refuses to reuse it.
# The shared trainer uses torch.autocast(device_type="cuda", dtype=torch.bfloat16).
DEFAULT_MESH="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244"
DEFAULT_HTSAT="/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt"
DEFAULT_MELLOW="/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main"
DEFAULT_MANIFEST="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl"
PERF20_RUN_ID="${PERF20_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}"
DEFAULT_OUTPUT_DIR="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/perf20_${PERF20_RUN_ID}"

torchrun --standalone --nproc_per_node=8 "$SCRIPT_DIR/train_audio_5_10x2_5_mesh_mellow_ddp.py" \
  --gate PERF20 \
  --model-path "$DEFAULT_MESH" \
  --htsat-checkpoint "$DEFAULT_HTSAT" \
  --mellow-root "$DEFAULT_MELLOW" \
  --train-manifest "$DEFAULT_MANIFEST" \
  --output-dir "$DEFAULT_OUTPUT_DIR" \
  --world-size 8 \
  --micro-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --epochs 1 \
  --max-steps 20 \
  --max-lr 1e-3 \
  --min-lr 0 \
  --seed 0 \
  --steady-state-start-step 6 \
  --no-profiler \
  "$@"
