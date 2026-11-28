#!/usr/bin/env bash
# Compatibility entry for the current S3 L11-14 gated-repeat experiment.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_mizar_s3_l11_14x3_residual_gate_5090.sh" "$@"
