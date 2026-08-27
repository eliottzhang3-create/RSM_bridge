#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

: "${RSMOL_SOURCE_CHECKPOINT:?Set RSMOL_SOURCE_CHECKPOINT to an external local checkpoint}"
: "${RSMOL_RECURSIVE_OUTPUT_DIR:?Set RSMOL_RECURSIVE_OUTPUT_DIR to an external output directory}"

echo "========== STEPWISE RECURSIVE CONVERSION JOB =========="
echo "ACTIVE_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "PYTHON=$(which python)"
echo "SOURCE_CHECKPOINT=$RSMOL_SOURCE_CHECKPOINT"
echo "RECURSIVE_OUTPUT_DIR=$RSMOL_RECURSIVE_OUTPUT_DIR"
echo "ALLOW_OVERWRITE=${RSMOL_ALLOW_OVERWRITE:-false}"
echo "OFFLINE_MODE=true"

ARGS=(--source-checkpoint "$RSMOL_SOURCE_CHECKPOINT" --output-dir "$RSMOL_RECURSIVE_OUTPUT_DIR")
if [[ "${RSMOL_ALLOW_OVERWRITE:-false}" == "true" ]]; then
  ARGS+=(--allow-overwrite)
fi
if [[ -n "${RSMOL_SOURCE_LAYER_INDICES:-}" ]]; then
  ARGS+=(--source-layer-indices "$RSMOL_SOURCE_LAYER_INDICES")
fi
python -u code/RSmol/scripts/convert_stepwise.py "${ARGS[@]}"
