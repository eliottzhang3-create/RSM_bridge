#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/submit_reasonaqa_stage_3090.sh" stage1 smoke "$@"
