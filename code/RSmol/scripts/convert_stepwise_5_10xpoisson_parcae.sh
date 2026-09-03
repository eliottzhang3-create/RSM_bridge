#!/bin/bash
set -euo pipefail
USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
SOURCE_CHECKPOINT="${RSMOL_5_10XPOISSON_PARCAE_SOURCE_CHECKPOINT:?set source checkpoint directory}"
OUTPUT_DIR="${RSMOL_5_10XPOISSON_PARCAE_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10xpoisson-parcae}"
ARGS=(--source-checkpoint "$SOURCE_CHECKPOINT" --output-dir "$OUTPUT_DIR" --seed "${RSMOL_5_10XPOISSON_PARCAE_SEED:-0}")
[[ "${RSMOL_5_10XPOISSON_PARCAE_ALLOW_OVERWRITE:-0}" == "1" ]] && ARGS+=(--allow-overwrite)
python -u code/RSmol/scripts/convert_stepwise_5_10xpoisson_parcae.py "${ARGS[@]}"
