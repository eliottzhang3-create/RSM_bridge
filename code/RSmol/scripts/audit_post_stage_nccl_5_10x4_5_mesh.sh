#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
cd "$REPO_ROOT"

SOURCE_DATA="${RSMOL_5_10X4_5_MESH_PERSISTENT_DATA_SOURCE:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset/data}"
OUTPUT="${RSMOL_5_10X4_5_MESH_POST_STAGE_AUDIT_OUTPUT:?set post-stage audit output directory}"
RUN_ID="${RSMOL_5_10X4_5_MESH_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}"
STAGE_ROOT="/dev/shm/rsmol_text_5_10x4_5_post_stage_audit_$RUN_ID"
STAGED_DATA="$STAGE_ROOT/data"
SOURCE_REPORT="$OUTPUT/source_data_inventory.json"
STAGED_REPORT="$OUTPUT/staged_data_inventory.json"
NCCL_OUTPUT="$OUTPUT/nccl_after_stage"

if [[ ! -d "$SOURCE_DATA" ]]; then
  echo "persistent text data directory is missing: $SOURCE_DATA" >&2
  exit 2
fi
if [[ -e "$STAGE_ROOT" ]]; then
  echo "refusing existing staging directory: $STAGE_ROOT" >&2
  exit 2
fi
if [[ -d "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing nonempty output directory: $OUTPUT" >&2
  exit 2
fi

mkdir -p "$OUTPUT"
df -Pk /dev/shm > "$OUTPUT/shm_before_stage.txt"
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
  if [[ -n "${STAGE_ROOT:-}" && "$STAGE_ROOT" == /dev/shm/rsmol_text_5_10x4_5_post_stage_audit_* && -d "$STAGE_ROOT" ]]; then
    rm -rf -- "$STAGE_ROOT"
  fi
}
trap cleanup_stage EXIT INT TERM

mkdir -p "$STAGED_DATA"
COPY_START_NS="$(date +%s%N)"
echo "[post-stage-nccl] copying $SOURCE_DATA to $STAGED_DATA" >&2
cp -a "$SOURCE_DATA"/. "$STAGED_DATA"/
COPY_END_NS="$(date +%s%N)"
COPY_SECONDS="$(awk -v start="$COPY_START_NS" -v end="$COPY_END_NS" 'BEGIN {printf "%.9f", (end-start)/1000000000}')"

python code/RSmol/scripts/audit_text_parquet_store_5_10x4_5_mesh.py \
  --data-dir "$STAGED_DATA" \
  --report-path "$STAGED_REPORT" \
  --compare-report "$SOURCE_REPORT"
df -Pk /dev/shm > "$OUTPUT/shm_after_stage.txt"
du -sk "$STAGE_ROOT" > "$OUTPUT/staged_size_kib.txt"
printf '%s\n' "$COPY_SECONDS" > "$OUTPUT/copy_seconds.txt"

echo "[post-stage-nccl] starting baseline NCCL audit with staged data resident" >&2
RSMOL_5_10X4_5_MESH_NCCL_AUDIT_OUTPUT="$NCCL_OUTPUT" \
  bash "$SCRIPT_DIR/audit_nccl_transport_5_10x4_5_mesh.sh" baseline

if [[ ! -f "$NCCL_OUTPUT/nccl_transport_report.json" ]]; then
  echo "post-stage NCCL audit did not create its final report" >&2
  exit 1
fi
echo "[post-stage-nccl] PASS copy_seconds=$COPY_SECONDS staged_data=$STAGED_DATA report=$NCCL_OUTPUT/nccl_transport_report.json" >&2
