#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "usage: $0 <smoke|resume|formal> [extra trainer arguments...]" >&2
  exit 2
fi
MODE="$1"
shift
case "$MODE" in
  smoke) GATE=D; MAX_STEPS=10 ;;
  resume) GATE=E; MAX_STEPS=2 ;;
  formal) GATE=FORMAL; MAX_STEPS=18488 ;;
  *) echo "invalid mode: $MODE" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-200000}"

SOURCE_DATA="${RSMOL_5_10X4_5_MESH_PERSISTENT_DATA_SOURCE:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset/data}"
MODEL="${RSMOL_5_10X4_5_MESH_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x4-5-mesh}"
OUTPUT="${RSMOL_5_10X4_5_MESH_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/${MODE}_2epochs_20260927_$(date +%Y%m%d_%H%M%S)}"
RESUME_FROM="${RSMOL_5_10X4_5_MESH_RESUME_FROM:-}"
RUN_ID="${RSMOL_5_10X4_5_MESH_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}"
STAGE_ROOT="/dev/shm/rsmol_text_5_10x4_5_$RUN_ID"
STAGED_DATA="$STAGE_ROOT/data"
SOURCE_REPORT="$OUTPUT/source_data_inventory.json"
STAGED_REPORT="$OUTPUT/staged_data_inventory.json"

if [[ ! -d "$SOURCE_DATA" ]]; then
  echo "persistent text data directory is missing: $SOURCE_DATA" >&2
  exit 2
fi
if [[ "$MODE" == resume && -z "$RESUME_FROM" ]]; then
  echo "resume mode requires RSMOL_5_10X4_5_MESH_RESUME_FROM" >&2
  exit 2
fi
if [[ -e "$STAGE_ROOT" ]]; then
  echo "refusing existing staging directory: $STAGE_ROOT" >&2
  exit 2
fi
if [[ "$MODE" != resume && -d "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing nonempty output directory: $OUTPUT" >&2
  exit 2
fi

mkdir -p "$OUTPUT"
python code/RSmol/scripts/audit_text_parquet_store_5_10x4_5_mesh.py \
  --data-dir "$SOURCE_DATA" \
  --report-path "$SOURCE_REPORT"

SOURCE_KIB="$(du -sk "$SOURCE_DATA" | awk '{print $1}')"
SHM_FREE_KIB="$(df -Pk /dev/shm | awk 'NR == 2 {print $4}')"
MARGIN_KIB=$((10 * 1024 * 1024))
REQUIRED_KIB=$((SOURCE_KIB + MARGIN_KIB))
if [[ -z "$SHM_FREE_KIB" || "$SHM_FREE_KIB" -lt "$REQUIRED_KIB" ]]; then
  echo "insufficient /dev/shm: free_kib=${SHM_FREE_KIB:-unknown} required_kib=$REQUIRED_KIB" >&2
  exit 2
fi

cleanup_stage() {
  if [[ -n "${STAGE_ROOT:-}" && "$STAGE_ROOT" == /dev/shm/rsmol_text_5_10x4_5_* && -d "$STAGE_ROOT" ]]; then
    rm -rf -- "$STAGE_ROOT"
  fi
}
trap cleanup_stage EXIT INT TERM

mkdir -p "$STAGED_DATA"
COPY_START_NS="$(date +%s%N)"
echo "[text-store-stage] copying $SOURCE_DATA to $STAGED_DATA" >&2
cp -a "$SOURCE_DATA"/. "$STAGED_DATA"/
COPY_END_NS="$(date +%s%N)"
COPY_SECONDS="$(awk -v start="$COPY_START_NS" -v end="$COPY_END_NS" 'BEGIN {printf "%.9f", (end-start)/1000000000}')"

python code/RSmol/scripts/audit_text_parquet_store_5_10x4_5_mesh.py \
  --data-dir "$STAGED_DATA" \
  --report-path "$STAGED_REPORT" \
  --compare-report "$SOURCE_REPORT"
echo "[text-store-stage] PASS copy_seconds=$COPY_SECONDS staged_data=$STAGED_DATA" >&2

ARGS=(
  --gate "$GATE"
  --model-path "$MODEL"
  --data-dir "$STAGED_DATA"
  --persistent-data-source "$SOURCE_DATA"
  --stage-report "$STAGED_REPORT"
  --output-dir "$OUTPUT"
  --world-size 8
  --micro-batch-size 8
  --gradient-accumulation-steps 16
  --context-length 1024
  --max-optimizer-steps "$MAX_STEPS"
  --scheduler-total-steps 18488
  --warmup-steps 925
  --max-lr 1e-3
  --min-lr 1e-4
  --save-every 500
  --steps-per-epoch 9244
  --epochs 2
  --seed "${RSMOL_5_10X4_5_MESH_SEED:-0}"
)
if [[ -n "$RESUME_FROM" ]]; then
  ARGS+=(--resume-from "$RESUME_FROM")
fi
if [[ -n "${RSMOL_5_10X4_5_MESH_TOKENIZER_PATH:-}" ]]; then
  ARGS+=(--tokenizer-path "$RSMOL_5_10X4_5_MESH_TOKENIZER_PATH")
fi
ARGS+=("$@")

torchrun --standalone --nproc_per_node=8 \
  code/RSmol/scripts/train_stage4_5_10x4_5_mesh_ddp.py \
  "${ARGS[@]}"
