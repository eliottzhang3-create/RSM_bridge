#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE="/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${RSMOL_SMOL_BLOCK_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2}"
DEVICE="${RSMOL_SMOL_BLOCK_DEVICE:-cuda:0}"
DTYPE="${RSMOL_SMOL_BLOCK_DTYPE:-bfloat16}"
LAYER_INDEX="${RSMOL_SMOL_BLOCK_LAYER_INDEX:-0}"
SEED="${RSMOL_SMOL_BLOCK_SEED:-0}"
SEQ_LEN="${RSMOL_SMOL_BLOCK_SEQ_LEN:-4}"
OUTPUT_DIR="${RSMOL_SMOL_BLOCK_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/smol-block-audit-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${RSMOL_SMOL_BLOCK_LOG_ROOT:-$REPO_ROOT/code/RSmol/log}"
LOG_PATH="$LOG_ROOT/smol_block_audit_runtime.log"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "========== SMOLLM2-135M TRANSFORMER BLOCK AUDIT JOB =========="
echo "ACTIVE_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "PYTHON=$(command -v python)"
echo "MODEL_PATH=$MODEL_PATH"
echo "DEVICE=$DEVICE"
echo "DTYPE=$DTYPE"
echo "LAYER_INDEX=$LAYER_INDEX"
echo "SEED=$SEED"
echo "SEQ_LEN=$SEQ_LEN"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOG_PATH=$LOG_PATH"
echo "OFFLINE_MODE=true"

python -u code/RSmol/scripts/audit_smol_block.py \
  --model-path "$MODEL_PATH" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --layer-index "$LAYER_INDEX" \
  --seed "$SEED" \
  --seq-len "$SEQ_LEN" \
  --output-dir "$OUTPUT_DIR"

