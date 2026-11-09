#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export RSMOL_5_10X3_5_STAGE4_GATE=FORMAL
export RSMOL_5_10X3_5_RESUME_FROM="${RSMOL_5_10X3_5_FORMAL_RESUME_FROM:-}"
exec bash "$SCRIPT_DIR/run_stage4_5_10x3_5_3090.sh"
