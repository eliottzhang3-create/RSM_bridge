#!/usr/bin/env bash
set -euo pipefail

# Runtime wrapper only.  It deliberately does not submit a job or choose a
# GPU; Stage 2 must be launched by the user's cluster scheduler.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

CHECKPOINT="${RSMOL_HTSAT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt}"
REPORT="${RSMOL_AUDIO_STAGE2_REPORT:-/tmp/audio_stage2_htsat_5_10_5_mellow.json}"
ARGS=(--htsat-checkpoint "$CHECKPOINT" --report-path "$REPORT" --device cuda)
[[ -n "${RSMOL_MELLOW_ROOT:-}" ]] && ARGS+=(--mellow-root "$RSMOL_MELLOW_ROOT")
[[ -n "${RSMOL_HTSAT_ROOT:-}" ]] && ARGS+=(--htsat-root "$RSMOL_HTSAT_ROOT")
if [[ -n "${RSMOL_AUDIO_PATH:-}" ]]; then
  ARGS+=(--audio-path "$RSMOL_AUDIO_PATH")
elif [[ -n "${RSMOL_AUDIO_MANIFEST:-}" ]]; then
  ARGS+=(--manifest "$RSMOL_AUDIO_MANIFEST")
else
  ARGS+=(--construction-only)
fi
[[ -n "${RSMOL_HTSAT_IMPLEMENTATION:-}" ]] && ARGS+=(--implementation "$RSMOL_HTSAT_IMPLEMENTATION")
python -u code/RSmol/scripts/audit_audio_stage2_htsat_5_10_5_mellow.py "${ARGS[@]}"
