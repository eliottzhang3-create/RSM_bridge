#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python "$SCRIPT_DIR/audit_audio_stage3_5_10x2_5_mesh_mellow.py" "$@"
