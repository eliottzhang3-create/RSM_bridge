#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${RSMOL_PYTHON:-python}
exec "$PYTHON_BIN" "$SCRIPT_DIR/audit_reasonaqa_paths.py" "$@"
