#!/usr/bin/env bash
set -euo pipefail
[[ $# == 7 ]] || { echo "run via code/RSmol/run_mizar_official_final_s1_gbs128_5_20x2_5_5090.sh" >&2; exit 2; }
MODE=$1
RESOLVED_PATHS=$(readlink -f "$2")
OUTPUT_ROOT=$(readlink -m "$3")
RESUME_CHECKPOINT=$4
MODEL_DIR=$(readlink -f "$5")
PREPARED_INDEX=$6
STAGE_WORKERS=$7
ROUTE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
[[ "$MODE" == formal && "$STAGE_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "requires formal mode and positive stage workers" >&2; exit 2; }
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
  # Reap the only producer before deleting files that its threads may write.
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

RUN_DIR="$OUTPUT_ROOT/formal"
mkdir -p "$RUN_DIR"
if [[ -z "$RESUME_CHECKPOINT" && -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "fresh formal directory is not empty: $RUN_DIR" >&2
  exit 2
fi
STAGE_TOKEN="formal_${RUN_ID}"
STAGE_ROOT="/dev/shm/mizar_official_final_s1_gbs128_5_20x2_5_${STAGE_TOKEN}"
STAGE_ARGS=(--manifest "$MANIFEST" --resolved-paths "$RESOLVED_PATHS" --prepared-index "$PREPARED_INDEX" \
  --stage-root "$STAGE_ROOT" --token "$STAGE_TOKEN" --mode formal --output "$RUN_DIR/staging_report.json" \
  --background --workers "$STAGE_WORKERS")
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  STAGE_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi
python stage_data.py "${STAGE_ARGS[@]}" &
STAGE_PID=$!
# Wait only for current-job path metadata; initial_steps=0, no audio prefill.
python -m data.staging --report "$RUN_DIR/staging_report.json" --stage-root "$STAGE_ROOT" --producer-pid "$STAGE_PID"
CONFIG_ARGS=(--prepared "$RUN_DIR/staging_report.json" --output "$RUN_DIR/runtime.json" \
  --run-dir "$RUN_DIR" --mode formal --model-dir "$MODEL_DIR")
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  CONFIG_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi
python write_config.py "${CONFIG_ARGS[@]}"
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --config "$RUN_DIR/runtime.json"
wait "$STAGE_PID"
STAGE_PID=""
python - "$RUN_DIR/training_report.json" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "PASS" or report.get("optimizer_steps") != 34626:
    raise SystemExit("training report did not confirm all 34626 real optimizer updates")
print(f"[formal] PASS: 34626 optimizer updates; report={sys.argv[1]}")
PY
