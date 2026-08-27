#!/bin/bash
set -euo pipefail

# Compatibility spelling matching the Stage 2 submit-wrapper convention.
# The delegated script contains the concrete ``vc submit`` command and the
# production resource contract (pdgpu-5090, -c 32 -m 256G -g 8 -n 1).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/run_stage4_5090.sh" "$@"
