#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${RSMOL_QWEN2_AUDIO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen2Audio-Instruct}"
OUTPUT="${RSMOL_QWEN2_AUDIO_PREFLIGHT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/qwen2_audio_instruct_eval/artifact_preflight.json}"
PYTHON_BIN="${RSMOL_QWEN2_AUDIO_PYTHON:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/rsmol/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Qwen2-Audio Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/audit_qwen2_audio_instruct_artifact.py \
  --model-path "$MODEL_PATH" \
  --output "$OUTPUT"

echo "Qwen2-Audio Python: $PYTHON_BIN"
echo "Qwen2-Audio artifact preflight report: $OUTPUT"
