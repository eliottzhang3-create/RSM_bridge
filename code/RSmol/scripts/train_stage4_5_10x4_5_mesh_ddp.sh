#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
shift || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/stage_text_shared_store_5_10x4_5_mesh.sh" "$MODE" "$@"
