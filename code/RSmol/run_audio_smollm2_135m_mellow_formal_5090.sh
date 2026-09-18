#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "deprecated wrapper: forwarding to pdgpu-3090" >&2
exec bash "$SCRIPT_DIR/run_audio_smollm2_135m_mellow_formal_3090.sh" "$@"
