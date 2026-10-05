#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
ROUTE="${RSMOL_MMAU_8SLOT_ROUTE:-x4_8slot}"
case "$ROUTE" in
  x4_8slot)
    CHECKPOINT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x4_5_mesh_8slot_mellow_shared_store_configurable_epochs/formal_3epochs_20261004_x4_8slot_v2/checkpoint-011343"
    ;;
  x5_8slot)
    CHECKPOINT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x5_5_mesh_8slot_mellow_shared_store_configurable_epochs/formal_3epochs_20261004_x5_8slot_v1/checkpoint-011343"
    ;;
  *)
    echo "RSMOL_MMAU_8SLOT_ROUTE must be x4_8slot or x5_8slot, got: $ROUTE" >&2
    exit 2
    ;;
esac

OUTPUT_ROOT="${RSMOL_MMAU_8SLOT_OUTPUT_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mmau_audio_mesh_zero_slot}"
OUTPUT_DIR="${RSMOL_MMAU_8SLOT_OUTPUT_DIR:-$OUTPUT_ROOT/$ROUTE/full_${RUN_TAG}}"

export RSMOL_MMAU_ROUTE="$ROUTE"
export RSMOL_MMAU_MODE="full"
export RSMOL_MMAU_CHECKPOINT="$CHECKPOINT"
export RSMOL_MMAU_OUTPUT_DIR="$OUTPUT_DIR"

# This isolated entry point intentionally does not forward positional arguments:
# the two 8-slot routes are always submitted as the audited full evaluation.
exec bash "$SCRIPT_DIR/run_mmau_test_mini_audio_mesh_shared_store_3090.sh"
