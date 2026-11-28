#!/usr/bin/env bash
set -euo pipefail
[[ $# == 6 ]] || { echo "use run_mizar_s3_l21_25x3_residual_gate_5090.sh" >&2; exit 2; }
DATA_DIR=$(readlink -f "$1")
OUTPUT_ROOT=$(readlink -m "$2")
MODEL_DIR=$(readlink -f "$3")
INDEX_DIR=$(readlink -f "$4")
STAGE_WORKERS=$5
RESUME=$6
ROUTE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONDA_BASE=${RSMOL_CONDA_BASE:-${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROUTE_ROOT"
mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.run.lock"
flock -n 9 || { echo "another job owns this output root" >&2; exit 2; }
RUN_DIR="$OUTPUT_ROOT/formal"
if [[ -z "$RESUME" && -d "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "fresh formal output is not empty" >&2; exit 2
fi
mkdir -p "$RUN_DIR"
STAGE_TOKEN="${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S%N)"
STAGE_ROOT="/dev/shm/mizar_s3_l21_25x3_gate_${STAGE_TOKEN}"
STAGE_PID=""
RUN_PHASE="staging_start"
cleanup() {
  local exit_code=$?
  trap - EXIT
  set +e
  if [[ -n "$STAGE_PID" ]]; then
    kill "$STAGE_PID" 2>/dev/null || true
    wait "$STAGE_PID" 2>/dev/null || true
    STAGE_PID=""
  fi
  python "$ROUTE_ROOT/stage_data.py" --cleanup --stage-root "$STAGE_ROOT" --token "$STAGE_TOKEN"
  local cleanup_code=$?
  if [[ $exit_code == 0 && $cleanup_code != 0 ]]; then exit_code=$cleanup_code; fi
  if [[ $exit_code != 0 ]]; then
    python "$ROUTE_ROOT/write_failure.py" "$RUN_DIR" "$exit_code" "$RUN_PHASE"
  fi
  exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
STAGE_ARGS=(--manifest "$DATA_DIR/train.json" --resolved-paths "$DATA_DIR/resolved_paths.tsv" \
  --prepared-index "$INDEX_DIR" --stage-root "$STAGE_ROOT" --token "$STAGE_TOKEN" \
  --mode formal --output "$RUN_DIR/staging_report.json" --background --workers "$STAGE_WORKERS")
[[ -z "$RESUME" ]] || STAGE_ARGS+=(--resume-checkpoint "$RESUME")
python -u stage_data.py "${STAGE_ARGS[@]}" &
STAGE_PID=$!
python -m data.staging --report "$RUN_DIR/staging_report.json" --stage-root "$STAGE_ROOT" --producer-pid "$STAGE_PID"
RUN_PHASE="write_config"
CONFIG_ARGS=(--prepared "$RUN_DIR/staging_report.json" --run-dir "$RUN_DIR" --model-dir "$MODEL_DIR" --output "$RUN_DIR/runtime.json")
[[ -z "$RESUME" ]] || CONFIG_ARGS+=(--resume-checkpoint "$RESUME")
python write_config.py "${CONFIG_ARGS[@]}"
RUN_PHASE="torchrun"
set +e
torchrun --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 train.py --config "$RUN_DIR/runtime.json"
TRAIN_EXIT=$?
set -e
if [[ $TRAIN_EXIT != 0 ]]; then
  exit "$TRAIN_EXIT"
fi
RUN_PHASE="producer_completion"
wait "$STAGE_PID"
STAGE_PID=""
RUN_PHASE="final_report"
python - "$RUN_DIR/training_report.json" "$RUN_DIR/runtime.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
c=json.load(open(sys.argv[2],encoding='utf-8'))
g=c['contract']['geometry']
if r.get('status')!='PASS' or r.get('optimizer_steps')!=g['total_steps'] or r.get('epochs')!=2 or not r.get('full_epoch_no_drop_no_repeat') or r.get('geometry')!=g:
    raise SystemExit('training report did not confirm two complete sealed S3 epochs')
print(f"PASS: {g['total_steps']} updates; {g['rows']} S3 rows per epoch; two complete epochs; frozen weights preserved; report="+sys.argv[1])
PY
