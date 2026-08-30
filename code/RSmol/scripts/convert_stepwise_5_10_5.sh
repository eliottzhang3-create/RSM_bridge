#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

SOURCE_CHECKPOINT="${RSMOL_5_10_5_SOURCE_CHECKPOINT:-${RSMOL_SOURCE_CHECKPOINT:-}}"
OUTPUT_DIR="${RSMOL_5_10_5_OUTPUT_DIR:-${RSMOL_RECURSIVE_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5}}"
if [[ -z "$SOURCE_CHECKPOINT" ]]; then
  echo "Set RSMOL_5_10_5_SOURCE_CHECKPOINT to an external local checkpoint" >&2
  exit 2
fi

echo "========== STEPWISE RECURSIVE CONVERSION JOB =========="
echo "ACTIVE_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "PYTHON=$(which python)"
echo "SOURCE_CHECKPOINT=$SOURCE_CHECKPOINT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "ALLOW_OVERWRITE=${RSMOL_5_10_5_ALLOW_OVERWRITE:-${RSMOL_ALLOW_OVERWRITE:-false}}"
echo "OFFLINE_MODE=true"

ARGS=(--source-checkpoint "$SOURCE_CHECKPOINT" --output-dir "$OUTPUT_DIR")
if [[ "${RSMOL_5_10_5_ALLOW_OVERWRITE:-${RSMOL_ALLOW_OVERWRITE:-false}}" == "true" ]]; then
  ARGS+=(--allow-overwrite)
fi
if [[ -n "${RSMOL_SOURCE_LAYER_INDICES:-}" ]]; then
  ARGS+=(--source-layer-indices "$RSMOL_SOURCE_LAYER_INDICES")
fi
python -u code/RSmol/scripts/convert_stepwise_5_10_5.py "${ARGS[@]}"
