#!/usr/bin/env bash
set -euo pipefail
[[ $# == 7 || $# == 8 ]] || { echo "run via code/RSmol/run_mizar_s1_gbs256_recursive_mesh_ced_small_layer36912_listen_5090.sh" >&2; exit 2; }
MODE=$1
RESOLVED_PATHS=$(readlink -f "$2")
OUTPUT_ROOT=$(readlink -m "$3")
RESUME_CHECKPOINT=$4
TEXT_MODEL_DIR=$(readlink -f "$5")
CED_MODEL_DIR=$(readlink -f "$6")
PREPARED_INDEX=$7
STAGE_WORKERS=${8:-4}
ROUTE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
[[ "$MODE" == formal ]] || { echo "this route runs formal training only" >&2; exit 2; }
CONDA_BASE=${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cd "$ROUTE_ROOT"
mkdir -p "$OUTPUT_ROOT"
RUN_ID="${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S%N)"
STAGE_ROOT=""
STAGE_TOKEN=""
STAGE_PID=""
cleanup() {
  local task_exit_code=$?
  # Stop/reap the producer before removing tmpfs files it might still write.
  if [[ -n "$STAGE_PID" ]]; then
    kill "$STAGE_PID" 2>/dev/null || true
    wait "$STAGE_PID" 2>/dev/null || true
    STAGE_PID=""
  fi
  if [[ -n "$STAGE_ROOT" ]]; then
    python "$ROUTE_ROOT/stage_data.py" --cleanup --stage-root "$STAGE_ROOT" --token "$STAGE_TOKEN"
  fi
  return "$task_exit_code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# The optional runtime checks are not part of this formal job.

run_one() {
  local run_mode=$1
  local run_dir="$OUTPUT_ROOT/$run_mode"
  mkdir -p "$run_dir"
  if [[ -z "$RESUME_CHECKPOINT" && -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "fresh $run_mode directory is not empty: $run_dir" >&2
    return 2
  fi
  STAGE_TOKEN="${run_mode}_${RUN_ID}"
  STAGE_ROOT="/dev/shm/mizar_s1_gbs256_recursive_mesh_ced_small_layer36912_listen_${STAGE_TOKEN}"
  local stage_args=(--manifest "$MANIFEST" --resolved-paths "$RESOLVED_PATHS" --prepared-index "$PREPARED_INDEX" \
    --stage-root "$STAGE_ROOT" --token "$STAGE_TOKEN" --mode "$run_mode" --output "$run_dir/staging_report.json" \
    --background --workers "$STAGE_WORKERS")
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    stage_args+=(--resume-checkpoint "$RESUME_CHECKPOINT")
  fi
  python stage_data.py "${stage_args[@]}" &
  STAGE_PID=$!
  python -m data.staging --report "$run_dir/staging_report.json" --stage-root "$STAGE_ROOT" --producer-pid "$STAGE_PID"
  local config_args=(--prepared "$run_dir/staging_report.json" --output "$run_dir/runtime.json" \
    --run-dir "$run_dir" --mode "$run_mode" --text-model-dir "$TEXT_MODEL_DIR" --ced-model-dir "$CED_MODEL_DIR")
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    config_args+=(--resume-checkpoint "$RESUME_CHECKPOINT")
  fi
  python write_config.py "${config_args[@]}"
  torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --config "$run_dir/runtime.json"
  wait "$STAGE_PID"
  STAGE_PID=""
  python - "$run_dir/training_report.json" "$run_mode" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = 17313
if report.get("status") != "PASS" or report.get("optimizer_steps") != expected:
    raise SystemExit("training report did not confirm all required optimizer updates")
print(f"[{sys.argv[2]}] PASS: {expected} optimizer updates; report={sys.argv[1]}")
PY
  cleanup
  STAGE_ROOT=""
  STAGE_TOKEN=""
}

run_one "$MODE"
