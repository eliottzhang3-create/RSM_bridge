#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RSMOL_MELLOW_TRAINING_ROUTE=adamw_cosine \
RSMOL_MELLOW_EVAL_MODE=smoke \
  bash "$SCRIPT_DIR/run_mmar_mellow_official_training_3090.sh" "$@"