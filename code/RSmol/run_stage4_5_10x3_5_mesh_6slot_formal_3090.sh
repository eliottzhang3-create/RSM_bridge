#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export RSMOL_5_10X3_5_MESH_6SLOT_STAGE4_GATE=FORMAL
export RSMOL_5_10X3_5_MESH_6SLOT_RESUME_FROM="${RSMOL_5_10X3_5_MESH_6SLOT_FORMAL_RESUME_FROM:-}"
exec bash "$SCRIPT_DIR/run_stage4_5_10x3_5_mesh_6slot_3090.sh"
