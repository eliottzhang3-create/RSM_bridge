#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <PASS audit-report.json> <mapping.jsonl> <output-dir> [smoke-rows]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
AUDIT_REPORT=$(readlink -f "$1")
MAPPING_JSONL=$(readlink -f "$2")
OUTPUT_DIR=$(readlink -m "$3")
SMOKE_ROWS=2048
if [[ $# -eq 4 ]]; then
  SMOKE_ROWS="$4"
fi

if [[ ! -f "$AUDIT_REPORT" || ! -f "$MAPPING_JSONL" ]]; then
  echo "audit report or mapping is missing" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

USER_CONDA_BASE="${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8

RUN_ID="${MELLOW_RUN_ID:-${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S%N)}"
STAGE_ROOT="/dev/shm/mellow_official_reasonaqa_$RUN_ID"
STAGING_REPORT="$OUTPUT_DIR/staging_report.json"
SMOKE_JSON="$OUTPUT_DIR/reasonaqa_smoke_$SMOKE_ROWS.json"
RUNTIME_CONFIG="$OUTPUT_DIR/runtime_smoke.yaml"
CHECKPOINT_ROOT="$OUTPUT_DIR/checkpoints"

cleanup() {
  if [[ -n "${STAGE_ROOT:-}" && -d "$STAGE_ROOT" ]]; then
    rm -rf -- "$STAGE_ROOT"
  fi
}
trap cleanup EXIT

echo "[mellow-smoke] staging raw audio into $STAGE_ROOT" >&2
python "$SCRIPT_DIR/stage_reasonaqa_raw_audio.py" \
  --audit-report "$AUDIT_REPORT" \
  --mapping-jsonl "$MAPPING_JSONL" \
  --stage-root "$STAGE_ROOT" \
  --report-path "$STAGING_REPORT" \
  --workers 16 \
  --free-space-margin-gib 10 \
  --decode-samples-per-group 3 \
  --copy-batch-size 1024 \
  --progress-every 1000

STAGED_TRAIN_JSON="$STAGE_ROOT/.mellow_stage/reasonaqa_train.json"
if [[ ! -f "$STAGED_TRAIN_JSON" || ! -f "$STAGE_ROOT/.mellow_stage/READY.json" ]]; then
  echo "staging did not produce READY.json and staged train metadata" >&2
  exit 1
fi

python "$SCRIPT_DIR/prepare_reasonaqa_smoke_metadata.py" \
  --input-json "$STAGED_TRAIN_JSON" \
  --output-json "$SMOKE_JSON" \
  --rows "$SMOKE_ROWS" \
  --seed 1234

python "$SCRIPT_DIR/write_reasonaqa_runtime_config.py" \
  --stage-root "$STAGE_ROOT" \
  --smoke-json "$SMOKE_JSON" \
  --output-config "$RUNTIME_CONFIG" \
  --save-dir "$CHECKPOINT_ROOT" \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --num-epochs 1 \
  --num-workers 4

export MELLOW_JOB_ID="mellow_official_reasonaqa_smoke_$RUN_ID"
cd "$ROUTE_ROOT"
echo "[mellow-smoke] launching torchrun world_size=8, batch=8, accumulation=4" >&2
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  train.py \
  --config "$RUNTIME_CONFIG" \
  --distributed-backend nccl \
  --save-dir "$CHECKPOINT_ROOT"

echo "[mellow-smoke] PASS: training exited successfully" >&2
echo "[mellow-smoke] staging_report=$STAGING_REPORT" >&2
echo "[mellow-smoke] runtime_config=$RUNTIME_CONFIG" >&2
echo "[mellow-smoke] checkpoints=$CHECKPOINT_ROOT/$MELLOW_JOB_ID" >&2
