#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

REPORT="${RSMOL_AUDIO_STAGE0_REPORT:-${RSMOL_AUDIO_REPORT_DIR:-/tmp}/audio_stage0_5_10_5_mellow.json}"
ARGS=(--report-path "$REPORT")
[[ -n "${RSMOL_MELLOW_ROOT:-}" ]] && ARGS+=(--mellow-root "$RSMOL_MELLOW_ROOT")
[[ -n "${RSMOL_HTSAT_ROOT:-}" ]] && ARGS+=(--htsat-root "$RSMOL_HTSAT_ROOT")
[[ -n "${RSMOL_REASONAQA_ROOT:-}" ]] && ARGS+=(--reasonaqa-root "$RSMOL_REASONAQA_ROOT")
[[ -n "${RSMOL_AUDIOCAPS_ROOT:-}" ]] && ARGS+=(--audiocaps-root "$RSMOL_AUDIOCAPS_ROOT")
[[ -n "${RSMOL_CLOTHO_ROOT:-}" ]] && ARGS+=(--clotho-root "$RSMOL_CLOTHO_ROOT")
[[ -n "${RSMOL_HTSAT_CHECKPOINT:-}" ]] && ARGS+=(--htsat-checkpoint "$RSMOL_HTSAT_CHECKPOINT")
[[ -n "${RSMOL_BASE_CHECKPOINT:-}" ]] && ARGS+=(--base-checkpoint "$RSMOL_BASE_CHECKPOINT")
python -u code/RSmol/scripts/audit_audio_stage0_5_10_5_mellow.py "${ARGS[@]}"
