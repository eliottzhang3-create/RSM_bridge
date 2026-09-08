#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${RSMOL_REASONAQA_MANIFEST_OUTPUT_DIR:-/tmp/reasonaqa_5_10_5_mellow}"
ARGS=(--output-dir "$OUTPUT_DIR")
[[ -n "${RSMOL_REASONAQA_ROOT:-}" ]] && ARGS+=(--reasonaqa-root "$RSMOL_REASONAQA_ROOT")
[[ -n "${RSMOL_AUDIOCAPS_ROOT:-}" ]] && ARGS+=(--audiocaps-root "$RSMOL_AUDIOCAPS_ROOT")
[[ -n "${RSMOL_CLOTHO_AUDIO_ROOT:-}" ]] && ARGS+=(--clotho-audio-root "$RSMOL_CLOTHO_AUDIO_ROOT")
[[ -n "${RSMOL_CLOTHO_AQA_AUDIO_ROOT:-}" ]] && ARGS+=(--clotho-aqa-audio-root "$RSMOL_CLOTHO_AQA_AUDIO_ROOT")
[[ -n "${RSMOL_REASONAQA_REPORT:-}" ]] && ARGS+=(--report-path "$RSMOL_REASONAQA_REPORT")
[[ -n "${RSMOL_REASONAQA_COMBINED_MANIFEST:-}" ]] && ARGS+=(--manifest-path "$RSMOL_REASONAQA_COMBINED_MANIFEST")
[[ "${RSMOL_REASONAQA_DRY_RUN:-0}" == "1" ]] && ARGS+=(--dry-run)
if [[ "${RSMOL_REASONAQA_ALLOW_MISSING:-0}" == "1" ]]; then
  ARGS+=(--allow-missing)
elif [[ "${RSMOL_REASONAQA_DROP_UNRESOLVED:-1}" == "1" ]]; then
  ARGS+=(--drop-unresolved)
fi
python -u code/RSmol/scripts/prepare_reasonaqa_manifest_5_10_5_mellow.py "${ARGS[@]}"
