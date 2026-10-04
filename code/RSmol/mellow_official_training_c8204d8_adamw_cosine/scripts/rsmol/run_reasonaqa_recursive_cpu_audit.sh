#!/usr/bin/env bash
set -u -o pipefail

# CPU-only preflight. This wrapper never stages audio and never submits a job.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${RSMOL_PYTHON:-python}

usage() {
  echo "usage: $0 <output-dir> [train-json] [checkpoint-dir] [audiocaps-root] [clotho-root] [clotho-aqa-root]" >&2
}

if [[ $# -lt 1 || $# -gt 6 ]]; then
  usage
  exit 2
fi

OUTPUT_DIR=$(readlink -m "$1")
TRAIN_JSON=$(readlink -m "${2:-/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/train.json}")
CHECKPOINT_DIR=$(readlink -m "${3:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244}")
AUDIOCAPS_ROOT=$(readlink -m "${4:-/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2}")
CLOTHO_ROOT=$(readlink -m "${5:-/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1}")
CLOTHO_AQA_ROOT=$(readlink -m "${6:-/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_audio/audio_files}")

mkdir -p "$OUTPUT_DIR"
PATH_REPORT="$OUTPUT_DIR/path_audit.json"
MAPPING_JSONL="$OUTPUT_DIR/train_audio_mapping.jsonl"
MANIFEST_REPORT="$OUTPUT_DIR/full_manifest_audit.json"
CHECKPOINT_REPORT="$OUTPUT_DIR/text_checkpoint_audit.json"

status=0
echo "[audit] path mapping: $TRAIN_JSON" >&2
"$PYTHON_BIN" "$SCRIPT_DIR/audit_reasonaqa_paths.py" \
  --train-json "$TRAIN_JSON" \
  --audiocaps-root "$AUDIOCAPS_ROOT" \
  --clotho-root "$CLOTHO_ROOT" \
  --clotho-aqa-root "$CLOTHO_AQA_ROOT" \
  --report-path "$PATH_REPORT" \
  --mapping-path "$MAPPING_JSONL" || status=1

echo "[audit] full manifest and mapping consistency" >&2
"$PYTHON_BIN" "$SCRIPT_DIR/audit_reasonaqa_full_manifest.py" \
  --manifest "$TRAIN_JSON" \
  --path-audit-report "$PATH_REPORT" \
  --mapping-jsonl "$MAPPING_JSONL" \
  --output-report "$MANIFEST_REPORT" \
  --world-size 8 \
  --per-rank-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --num-epochs 30 \
  --warmup-ratio 0.05 || status=1

echo "[audit] recursive text checkpoint: $CHECKPOINT_DIR" >&2
"$PYTHON_BIN" "$SCRIPT_DIR/audit_reasonaqa_recursive_text_checkpoint.py" \
  --checkpoint "$CHECKPOINT_DIR" \
  --output-report "$CHECKPOINT_REPORT" || status=1

echo "[audit] reports:" >&2
printf '  path=%s\n  manifest=%s\n  checkpoint=%s\n' "$PATH_REPORT" "$MANIFEST_REPORT" "$CHECKPOINT_REPORT" >&2
exit "$status"
