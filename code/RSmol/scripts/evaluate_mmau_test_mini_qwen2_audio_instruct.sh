#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python "$SCRIPT_DIR/evaluate_mmau_test_mini_qwen2_audio_instruct.py" "$@"
