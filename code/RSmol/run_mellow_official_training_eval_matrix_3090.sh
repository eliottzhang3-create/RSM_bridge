#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE="${1:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "usage: bash $0 [smoke|full]" >&2
  exit 2
fi

submit_one() {
  local benchmark="$1"
  local route="$2"
  if [[ "$benchmark" == "mmau" ]]; then
    RSMOL_MELLOW_TRAINING_ROUTE="$route" \
    RSMOL_MELLOW_EVAL_MODE="$MODE" \
      bash "$SCRIPT_DIR/run_mmau_test_mini_mellow_official_training_3090.sh"
  else
    RSMOL_MELLOW_TRAINING_ROUTE="$route" \
    RSMOL_MELLOW_EVAL_MODE="$MODE" \
      bash "$SCRIPT_DIR/run_mmar_mellow_official_training_3090.sh"
  fi
}

submit_one mmau c8204d8
submit_one mmau adamw_cosine
submit_one mmar c8204d8
submit_one mmar adamw_cosine

echo "submitted official Mellow training evaluation matrix mode=$MODE"