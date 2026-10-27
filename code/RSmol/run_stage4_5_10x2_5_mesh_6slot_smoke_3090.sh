#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export RSMOL_5_10X2_5_MESH_6SLOT_RESUME_FROM=
export RSMOL_5_10X2_5_MESH_6SLOT_STAGE4_GATE=D
bash "$SCRIPT_DIR/run_stage4_5_10x2_5_mesh_6slot_3090.sh"
