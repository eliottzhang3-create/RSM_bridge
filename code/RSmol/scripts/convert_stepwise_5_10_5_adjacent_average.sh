#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

SOURCE_CHECKPOINT="${RSMOL_5_10_5_ADJAVG_SOURCE_CHECKPOINT:-}"
OUTPUT_DIR="${RSMOL_5_10_5_ADJAVG_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5-adjacent-average}"
SEED="${RSMOL_5_10_5_ADJAVG_SEED:-0}"
if [[ -z "$SOURCE_CHECKPOINT" ]]; then
  echo "Set RSMOL_5_10_5_ADJAVG_SOURCE_CHECKPOINT to the external 30-layer SmolLM2 checkpoint" >&2
  exit 2
fi
for path_value in "$SOURCE_CHECKPOINT" "$OUTPUT_DIR"; do
  case "$path_value" in
    /*) ;;
    *) echo "Adjacent-average source/output paths must be absolute: $path_value" >&2; exit 2 ;;
  esac
done
case "$OUTPUT_DIR" in
  /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM|/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/*)
    echo "Adjacent-average output must be outside the Git checkout: $OUTPUT_DIR" >&2
    exit 2
    ;;
esac

echo "========== 5-10-5 ADJACENT-AVERAGE CONVERSION =========="
echo "ACTIVE_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "PYTHON=$(which python)"
echo "SOURCE_CHECKPOINT=$SOURCE_CHECKPOINT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "SEED=$SEED"
echo "ALLOW_OVERWRITE=${RSMOL_5_10_5_ADJAVG_ALLOW_OVERWRITE:-false}"

ARGS=(
  --source-checkpoint "$SOURCE_CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --seed "$SEED"
)
if [[ "${RSMOL_5_10_5_ADJAVG_ALLOW_OVERWRITE:-false}" == "true" ]]; then
  ARGS+=(--allow-overwrite)
fi
python -u code/RSmol/scripts/convert_stepwise_5_10_5_adjacent_average.py "${ARGS[@]}"
