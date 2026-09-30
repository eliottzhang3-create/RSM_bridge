#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <PASS path-audit report.json> [persistent output directory]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATH_AUDIT_REPORT=$(readlink -f "$1")
OUTPUT_DIR="${2:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/training_preflight_$(date +%Y%m%d_%H%M%S%N)}"
REPORT_PATH="$OUTPUT_DIR/training_preflight_report.json"

if [[ ! -f "$PATH_AUDIT_REPORT" ]]; then
  echo "path-audit report is missing: $PATH_AUDIT_REPORT" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" && ! -d "$OUTPUT_DIR" ]]; then
  echo "output path exists and is not a directory: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
if find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "refusing non-empty output directory: $OUTPUT_DIR" >&2
  exit 2
fi

USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
TEXT_MODEL_DIR="${MELLOW_TEXT_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2}"
HTSAT_CHECKPOINT="${MELLOW_HTSAT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt}"

source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8

echo "[mellow-preflight] login-node audit; no waveform/CUDA/language-model load or torchrun" >&2
echo "[mellow-preflight] path_audit=$PATH_AUDIT_REPORT" >&2
echo "[mellow-preflight] text_model=$TEXT_MODEL_DIR" >&2
echo "[mellow-preflight] htsat=$HTSAT_CHECKPOINT" >&2
echo "[mellow-preflight] report=$REPORT_PATH" >&2

python "$SCRIPT_DIR/audit_8gpu_training_preflight.py" \
  --path-audit-report "$PATH_AUDIT_REPORT" \
  --text-model-dir "$TEXT_MODEL_DIR" \
  --htsat-checkpoint "$HTSAT_CHECKPOINT" \
  --report-path "$REPORT_PATH" \
  --gpus 8 \
  --cpus 32 \
  --memory-gib 256 \
  --workers-per-rank 4 \
  --candidate-batch-size-per-rank 8 \
  --gradient-accumulation-steps 4 \
  --staging-margin-gib 10

python - "$REPORT_PATH" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
assert report.get("status") == "PASS", report.get("hard_failures")
print(
    "[mellow-preflight] PASS "
    f"checks={len(report['checks'])} "
    f"warnings={len(report['warnings'])} "
    f"staged_payload_gib={report['mapping']['staged_payload_gib']:.3f} "
    f"report={path}",
    flush=True,
)
PY
