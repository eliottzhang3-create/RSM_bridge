#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESUME_FROM="${RSMOL_5_10X2_5_MESH_6SLOT_RESUME_FROM:?set the smoke checkpoint-000010 path}"
for REQUIRED_FILE in checkpoint_complete.json checkpoint_manifest.json training_state.pt; do
  if [[ ! -f "$RESUME_FROM/$REQUIRED_FILE" ]]; then
    echo "incomplete resume checkpoint: $RESUME_FROM/$REQUIRED_FILE" >&2; exit 2
  fi
done
export RSMOL_5_10X2_5_MESH_6SLOT_STAGE4_GATE=E
bash "$SCRIPT_DIR/run_stage4_5_10x2_5_mesh_6slot_3090.sh"
