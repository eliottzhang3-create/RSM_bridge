#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export RSMOL_MMAU_MODE="smoke"
# One smoke is sufficient for the common evaluator; x5 exercises the longest
# route trace while the full jobs still audit every selected route strictly.
export RSMOL_MMAU_ROUTE="${RSMOL_MMAU_ROUTE:-x5_7slot}"
exec "$SCRIPT_DIR/run_mmau_test_mini_audio_mesh_shared_store_3090.sh" "$@"
