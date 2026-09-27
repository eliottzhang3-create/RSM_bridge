#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
USER_CONDA_BASE="${{USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
cd "$REPO_ROOT"

OUTPUT="${{RSMOL_5_10X4_5_MESH_SAME_ALLOCATION_OUTPUT:?set same-allocation output directory}"
SOURCE_DATA="${{RSMOL_5_10X4_5_MESH_PERSISTENT_DATA_SOURCE:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset/data}"
X2_MODEL="${{RSMOL_5_10X4_5_MESH_X2_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x2-5-mesh}"
X4_MODEL="${{RSMOL_5_10X4_5_MESH_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x4-5-mesh}"
RUN_ID="${{RSMOL_5_10X4_5_MESH_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}"
STAGE_ROOT="/dev/shm/rsmol_text_5_10x4_5_same_allocation_$RUN_ID"
STAGED_DATA="$STAGE_ROOT/data"
STATUS_FILE="$OUTPUT/stage_status.tsv"
SOURCE_REPORT="$OUTPUT/source_data_inventory.json"
STAGED_REPORT="$OUTPUT/staged_data_inventory.json"
STAGE_TIMEOUT_SECONDS="${{RSMOL_5_10X4_5_MESH_AUDIT_STAGE_TIMEOUT_SECONDS:-600}"

for path in "$SOURCE_DATA" "$X2_MODEL" "$X4_MODEL"; do
  if [[ ! -e "$path" ]]; then
    echo "required path is missing: $path" >&2
    exit 2
  fi
done
if [[ -d "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing nonempty output directory: $OUTPUT" >&2
  exit 2
fi
if [[ -e "$STAGE_ROOT" ]]; then
  echo "refusing existing staging directory: $STAGE_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT/environment"
printf 'stage\tstatus\tstart_epoch\tend_epoch\texit_code\n' > "$STATUS_FILE"

cleanup_stage() {
  if [[ -n "${{STAGE_ROOT:-}" && "$STAGE_ROOT" == /dev/shm/rsmol_text_5_10x4_5_same_allocation_* && -d "$STAGE_ROOT" ]]; then
    rm -rf -- "$STAGE_ROOT"
  fi
}
trap cleanup_stage EXIT INT TERM

date -Ins > "$OUTPUT/environment/date.txt"
hostname -f > "$OUTPUT/environment/hostname.txt" 2>&1 || hostname > "$OUTPUT/environment/hostname.txt"
env | sort > "$OUTPUT/environment/environment.txt"
nvidia-smi -L > "$OUTPUT/environment/nvidia_smi_list.txt" 2>&1
nvidia-smi topo -m > "$OUTPUT/environment/nvidia_smi_topology.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total --format=csv,noheader \
  > "$OUTPUT/environment/gpu_inventory.csv" 2>&1
df -Pk /dev/shm > "$OUTPUT/environment/shm_initial.txt"

run_stage() {
  local stage="$1"
  shift
  local start end rc
  start="$(date +%s)"
  echo "[same-allocation] stage=$stage status=START" >&2
  set +e
  "$@"
  rc=$?
  set -e
  end="$(date +%s)"
  if (( rc == 0 )); then
    printf '%s\tPASS\t%s\t%s\t0\n' "$stage" "$start" "$end" >> "$STATUS_FILE"
    echo "[same-allocation] stage=$stage status=PASS seconds=$((end-start))" >&2
    return 0
  fi
  printf '%s\tFAIL\t%s\t%s\t%s\n' "$stage" "$start" "$end" "$rc" >> "$STATUS_FILE"
  echo "[same-allocation] stage=$stage status=FAIL exit_code=$rc seconds=$((end-start))" >&2
  return "$rc"
}

run_stage 01_model_free_nccl \
  timeout --signal=TERM "$STAGE_TIMEOUT_SECONDS" \
  env RSMOL_5_10X4_5_MESH_NCCL_AUDIT_OUTPUT="$OUTPUT/01_model_free_nccl" \
  bash "$SCRIPT_DIR/audit_nccl_transport_5_10x4_5_mesh.sh" baseline

run_stage 02_historical_x2_smoke \
  timeout --signal=TERM "$STAGE_TIMEOUT_SECONDS" \
  env \
    RSMOL_5_10X2_5_MESH_STAGE4_GATE=D \
    RSMOL_5_10X2_5_MESH_WORLD_SIZE=8 \
    RSMOL_5_10X2_5_MESH_MODEL_DIR="$X2_MODEL" \
    RSMOL_5_10X2_5_MESH_DATA_DIR="$SOURCE_DATA" \
    RSMOL_5_10X2_5_MESH_OUTPUT_DIR="$OUTPUT/02_historical_x2_smoke" \
    RSMOL_5_10X2_5_MESH_MICRO_BATCH_SIZE=1 \
    RSMOL_5_10X2_5_MESH_GRADIENT_ACCUMULATION_STEPS=1 \
    RSMOL_5_10X2_5_MESH_MAX_OPTIMIZER_STEPS=1 \
    RSMOL_5_10X2_5_MESH_SCHEDULER_TOTAL_STEPS=1 \
    RSMOL_5_10X2_5_MESH_WARMUP_STEPS=1 \
  bash "$SCRIPT_DIR/train_stage4_5_10x2_5_mesh_ddp.sh"

run_stage 03_source_inventory \
  python code/RSmol/scripts/audit_text_parquet_store_5_10x4_5_mesh.py \
    --data-dir "$SOURCE_DATA" \
    --report-path "$SOURCE_REPORT"

SOURCE_KIB="$(du -sk "$SOURCE_DATA" | awk '{print $1}')"
SHM_FREE_KIB="$(df -Pk /dev/shm | awk 'NR == 2 {print $4}')"
MARGIN_KIB=$((10 * 1024 * 1024))
REQUIRED_KIB=$((SOURCE_KIB + MARGIN_KIB))
if [[ -z "$SHM_FREE_KIB" || "$SHM_FREE_KIB" -lt "$REQUIRED_KIB" ]]; then
  echo "insufficient /dev/shm: free_kib=${{SHM_FREE_KIB:-unknown} required_kib=$REQUIRED_KIB" >&2
  exit 2
fi

mkdir -p "$STAGED_DATA"
run_stage 04_copy_to_shared_memory cp -a "$SOURCE_DATA"/. "$STAGED_DATA"/

run_stage 05_staged_inventory \
  python code/RSmol/scripts/audit_text_parquet_store_5_10x4_5_mesh.py \
    --data-dir "$STAGED_DATA" \
    --report-path "$STAGED_REPORT" \
    --compare-report "$SOURCE_REPORT"

df -Pk /dev/shm > "$OUTPUT/environment/shm_after_stage.txt"
du -sk "$STAGE_ROOT" > "$OUTPUT/environment/staged_size_kib.txt"

run_stage 06_post_stage_nccl \
  timeout --signal=TERM "$STAGE_TIMEOUT_SECONDS" \
  env RSMOL_5_10X4_5_MESH_NCCL_AUDIT_OUTPUT="$OUTPUT/06_post_stage_nccl" \
  bash "$SCRIPT_DIR/audit_nccl_transport_5_10x4_5_mesh.sh" baseline

run_stage 07_x4_smoke \
  timeout --signal=TERM "$STAGE_TIMEOUT_SECONDS" \
  torchrun --standalone --nproc_per_node=8 \
    code/RSmol/scripts/train_stage4_5_10x4_5_mesh_ddp.py \
    --gate D \
    --model-path "$X4_MODEL" \
    --data-dir "$STAGED_DATA" \
    --persistent-data-source "$SOURCE_DATA" \
    --stage-report "$STAGED_REPORT" \
    --output-dir "$OUTPUT/07_x4_smoke" \
    --world-size 8 \
    --micro-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --context-length 1024 \
    --max-optimizer-steps 1 \
    --scheduler-total-steps 1 \
    --warmup-steps 1 \
    --max-lr 1e-3 \
    --min-lr 1e-4 \
    --save-every 500 \
    --steps-per-epoch 9244 \
    --epochs 2 \
    --seed 0

touch "$OUTPUT/SAME_ALLOCATION_AUDIT_PASS"
echo "[same-allocation] status=PASS output=$OUTPUT" >&2
