#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${RSMOL_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2}"
PROMPT="${RSMOL_SMOKE_PROMPT:-Gravity is}"
MAX_NEW_TOKENS="${RSMOL_SMOKE_MAX_NEW_TOKENS:-32}"
DEVICE="${RSMOL_SMOKE_DEVICE:-cuda:0}"
OUTPUT_REPORT="${RSMOL_SMOKE_OUTPUT_REPORT:-$REPO_ROOT/outputs/RSmol/smollm2_inference_smoke_$(date +%Y%m%d_%H%M%S).json}"

mkdir -p "$(dirname "$OUTPUT_REPORT")"

echo "========== SMOLLM2-135M INFERENCE SMOKE JOB =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH=$MODEL_PATH"
echo "PROMPT=$PROMPT"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "DEVICE=$DEVICE"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"
echo "OFFLINE_MODE=true"

python -u code/RSmol/scripts/smoke_smollm2_inference.py \
  --model-path "$MODEL_PATH" \
  --device "$DEVICE" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output-report "$OUTPUT_REPORT"
