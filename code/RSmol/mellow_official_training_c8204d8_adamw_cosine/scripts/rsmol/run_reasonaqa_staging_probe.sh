#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <PASS path-audit report.json> <train-audio mapping.jsonl> [persistent output dir]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AUDIT_REPORT=$(readlink -f "$1")
MAPPING_JSONL=$(readlink -f "$2")
OUTPUT_DIR="${3:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5090/staging_probe_$(date +%Y%m%d_%H%M%S%N)}"

if [[ ! -f "$AUDIT_REPORT" ]]; then
  echo "PASS path-audit report is missing: $AUDIT_REPORT" >&2
  exit 2
fi
if [[ ! -f "$MAPPING_JSONL" ]]; then
  echo "train-audio mapping is missing: $MAPPING_JSONL" >&2
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
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8

RUN_ID="${MELLOW_STAGE_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}"
STAGE_ROOT="/dev/shm/mellow_adamw_cosine_reasonaqa_${RUN_ID}"
REPORT_PATH="$OUTPUT_DIR/staging_report.json"
WORKERS="${MELLOW_STAGE_WORKERS:-16}"
MARGIN_GIB="${MELLOW_STAGE_MARGIN_GIB:-10}"
DECODE_SAMPLES_PER_GROUP="${MELLOW_STAGE_DECODE_SAMPLES_PER_GROUP:-3}"
COPY_BATCH_SIZE="${MELLOW_STAGE_COPY_BATCH_SIZE:-1024}"

cleanup_stage() {
  if [[ "${MELLOW_KEEP_STAGE:-0}" == "1" ]]; then
    echo "[mellow-stage-cleanup] preserving debug stage because MELLOW_KEEP_STAGE=1: $STAGE_ROOT" >&2
    return
  fi
  case "${STAGE_ROOT:-}" in
    /dev/shm/mellow_adamw_cosine_reasonaqa_*)
      if [[ -e "$STAGE_ROOT" ]]; then
        echo "[mellow-stage-cleanup] removing $STAGE_ROOT" >&2
        rm -rf -- "$STAGE_ROOT"
      fi
      ;;
    *)
      echo "[mellow-stage-cleanup] refusing unsafe cleanup target: ${STAGE_ROOT:-unset}" >&2
      ;;
  esac
}
trap cleanup_stage EXIT INT TERM

echo "[mellow-stage-probe] hostname=$(hostname)" >&2
echo "[mellow-stage-probe] audit=$AUDIT_REPORT" >&2
echo "[mellow-stage-probe] mapping=$MAPPING_JSONL" >&2
echo "[mellow-stage-probe] stage_root=$STAGE_ROOT" >&2
echo "[mellow-stage-probe] output_dir=$OUTPUT_DIR" >&2
df -h /dev/shm >&2

export PYTHONUNBUFFERED=1
python "$SCRIPT_DIR/stage_reasonaqa_raw_audio.py" \
  --audit-report "$AUDIT_REPORT" \
  --mapping-jsonl "$MAPPING_JSONL" \
  --stage-root "$STAGE_ROOT" \
  --report-path "$REPORT_PATH" \
  --workers "$WORKERS" \
  --free-space-margin-gib "$MARGIN_GIB" \
  --decode-samples-per-group "$DECODE_SAMPLES_PER_GROUP" \
  --copy-batch-size "$COPY_BATCH_SIZE"

python - "$REPORT_PATH" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
assert report.get("status") == "PASS", report
assert report.get("file_count", 0) > 0, report
assert report.get("staged_payload_bytes", 0) > 0, report
assert report.get("decode_results"), report
print(
    "[mellow-stage-probe] PASS "
    f"files={report['file_count']} "
    f"payload_gib={report['staged_payload_gib']:.3f} "
    f"copy_seconds={report['copy_seconds']:.3f} "
    f"throughput_mib_s={report['copy_throughput_mib_s']:.2f} "
    f"report={path}",
    flush=True,
)
PY

df -h /dev/shm >&2
