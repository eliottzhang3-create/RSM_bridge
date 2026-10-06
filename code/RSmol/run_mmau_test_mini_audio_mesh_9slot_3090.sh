#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
CHECKPOINT="${RSMOL_MMAU_9SLOT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x5_5_mesh_9slot_mellow_shared_store_configurable_epochs/formal_3epochs_20261006_054353274259875-20/checkpoint-011343}"
OUTPUT_ROOT="${RSMOL_MMAU_9SLOT_OUTPUT_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mmau_audio_mesh_zero_slot}"
OUTPUT_DIR="${RSMOL_MMAU_9SLOT_OUTPUT_DIR:-$OUTPUT_ROOT/x5_9slot/full_${RUN_TAG}}"

export RSMOL_MMAU_ROUTE="x5_9slot"
export RSMOL_MMAU_MODE="full"
export RSMOL_MMAU_CHECKPOINT="$CHECKPOINT"
export RSMOL_MMAU_OUTPUT_DIR="$OUTPUT_DIR"

# The 9-slot entry point is always the complete, official-scored evaluation.
exec bash "$SCRIPT_DIR/run_mmau_test_mini_audio_mesh_shared_store_3090.sh"
