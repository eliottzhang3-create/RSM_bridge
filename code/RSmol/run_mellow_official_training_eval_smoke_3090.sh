#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_mellow_official_training_eval_matrix_3090.sh" smoke