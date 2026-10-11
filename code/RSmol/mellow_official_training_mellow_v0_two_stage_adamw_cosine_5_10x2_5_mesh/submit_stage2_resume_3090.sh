#!/usr/bin/env bash
set -euo pipefail
export MELLOW_STAGE2_PROFILE="${MELLOW_STAGE2_PROFILE:-stage2_gbs256_lr1e-4}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/submit_reasonaqa_stage_3090.sh" stage2 resume "$@"
