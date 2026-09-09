#!/usr/bin/env bash
set -euo pipefail

# This runtime script is the only place where the remote Conda environment is
# selected.  The submit wrapper does not run Python on the login host.
USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
if [[ ! -f "$USER_CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "Missing Conda shell hook: $USER_CONDA_BASE/etc/profile.d/conda.sh" >&2
  exit 2
fi
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
RSMOL_BASELINE_CONDA_ENV="${RSMOL_BASELINE_CONDA_ENV:-rsmol}"
conda activate "$RSMOL_BASELINE_CONDA_ENV"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${RSMOL_BASELINE_FALCON_MODEL:-/hpc_stor03/sjtu_home/jinwei.zhang/models/falcon90M}"
BENCHMARK_ROOT="${RSMOL_BASELINE_BENCHMARK_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets}"
OUTPUT_DIR="${RSMOL_BASELINE_FALCON_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage3-baseline-falcon90m-$(date +%Y%m%d_%H%M%S)-$$}"
DEVICE="${RSMOL_BASELINE_DEVICE:-cuda:0}"
DTYPE="${RSMOL_BASELINE_DTYPE:-bfloat16}"
BATCH_SIZE="${RSMOL_BASELINE_BATCH_SIZE:-1}"
SEED="${RSMOL_BASELINE_SEED:-0}"
TASKS="${RSMOL_BASELINE_TASKS:-hellaswag,mmlu,gsm8k,arc_easy,arc_challenge}"
CACHE_DIR="${RSMOL_BASELINE_CACHE_DIR:-/tmp/rsmol-stage3-baseline-falcon90m-cache-$$}"
LOG_ROOT="${RSMOL_BASELINE_LOG_ROOT:-$REPO_ROOT/code/RSmol/log}"

IFS=',' read -r -a RAW_TASK_ARRAY <<< "$TASKS"
TASK_ARRAY=()
for task in "${RAW_TASK_ARRAY[@]}"; do
  task="${task//[[:space:]]/}"
  [[ -n "$task" ]] && TASK_ARRAY+=("$task")
done
if [[ "${#TASK_ARRAY[@]}" -eq 0 ]]; then
  echo "RSMOL_BASELINE_TASKS must contain at least one task" >&2
  exit 2
fi

MODE_ARGS=()
[[ "${RSMOL_BASELINE_VALIDATION_ONLY:-0}" == "1" ]] && MODE_ARGS+=(--validation-only)
[[ "${RSMOL_BASELINE_SMOKE:-0}" == "1" ]] && MODE_ARGS+=(--smoke)
[[ "${RSMOL_BASELINE_NO_LOG_SAMPLES:-0}" == "1" ]] && MODE_ARGS+=(--no-log-samples)
[[ -n "${RSMOL_BASELINE_LIMIT:-}" ]] && MODE_ARGS+=(--limit "$RSMOL_BASELINE_LIMIT")

RUNTIME_LOG="$LOG_ROOT/falcon90m/runtime.log"
mkdir -p "$(dirname "$RUNTIME_LOG")"
python -u code/RSmol/scripts/evaluate_stage3_baseline.py \
  --model-key falcon90m \
  --model-path "$MODEL_PATH" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --cache-dir "$CACHE_DIR" \
  --log-root "$LOG_ROOT" \
  --tasks "${TASK_ARRAY[@]}" \
  "${MODE_ARGS[@]}" 2>&1 | tee -a "$RUNTIME_LOG"
