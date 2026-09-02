#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${RSMOL_STAGE3_5_10_5_LINEAR_MODEL:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5-linear}"
BENCHMARK_ROOT="${RSMOL_STAGE3_5_10_5_LINEAR_BENCHMARK_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets}"
OUTPUT_DIR="${RSMOL_STAGE3_5_10_5_LINEAR_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage3-5-10-5-linear-$(date +%Y%m%d_%H%M%S)-$$}"
DEVICE="${RSMOL_STAGE3_5_10_5_LINEAR_DEVICE:-cuda:0}"
BATCH_SIZE="${RSMOL_STAGE3_5_10_5_LINEAR_BATCH_SIZE:-1}"
SEED="${RSMOL_STAGE3_5_10_5_LINEAR_SEED:-0}"
TASKS="${RSMOL_STAGE3_5_10_5_LINEAR_TASKS:-hellaswag,mmlu,gsm8k,arc_easy,arc_challenge}"
CACHE_ROOT="${RSMOL_STAGE3_5_10_5_LINEAR_CACHE_ROOT:-/tmp/rsmol-stage3-5-10-5-linear-$$}"
LOG_ROOT="${RSMOL_STAGE3_5_10_5_LINEAR_LOG_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol/log}"

for value in "$MODEL_PATH" "$BENCHMARK_ROOT" "$OUTPUT_DIR" "$CACHE_ROOT" "$LOG_ROOT"; do
  case "$value" in /*) ;; *) echo "Stage 3 linear paths must be absolute: $value" >&2; exit 2 ;; esac
done
case "$OUTPUT_DIR" in
  /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM|/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/*)
    echo "Stage 3 linear output must be outside the Git checkout: $OUTPUT_DIR" >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_DIR" && ( ! -d "$OUTPUT_DIR" || -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ) ]]; then
  echo "Refusing to overwrite non-empty Stage 3 linear output: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR" "$LOG_ROOT"

IFS=',' read -r -a RAW_TASKS <<< "$TASKS"
TASK_ARGS=(--tasks)
for task in "${RAW_TASKS[@]}"; do
  task="${task//[[:space:]]/}"
  [[ -n "$task" ]] && TASK_ARGS+=("$task")
done
[[ "${#TASK_ARGS[@]}" -gt 1 ]] || { echo "No Stage 3 tasks selected" >&2; exit 2; }

MODE_ARGS=()
[[ -n "${RSMOL_STAGE3_5_10_5_LINEAR_DTYPE:-}" ]] && MODE_ARGS+=(--dtype "$RSMOL_STAGE3_5_10_5_LINEAR_DTYPE")
[[ "${RSMOL_STAGE3_5_10_5_LINEAR_VALIDATION_ONLY:-0}" == 1 ]] && MODE_ARGS+=(--validation-only)
[[ "${RSMOL_STAGE3_5_10_5_LINEAR_SMOKE:-0}" == 1 ]] && MODE_ARGS+=(--smoke)
[[ "${RSMOL_STAGE3_5_10_5_LINEAR_NO_LOG_SAMPLES:-0}" == 1 ]] && MODE_ARGS+=(--no-log-samples)
[[ -n "${RSMOL_STAGE3_5_10_5_LINEAR_LIMIT:-}" ]] && MODE_ARGS+=(--limit "$RSMOL_STAGE3_5_10_5_LINEAR_LIMIT")

RUNTIME_LOG="$LOG_ROOT/stage3_5_10_5_linear_runtime.log"
echo "========== STAGE3 MODEL: linear_5_10_5 ==========" | tee -a "$RUNTIME_LOG"
echo "MODEL_PATH=$MODEL_PATH" | tee -a "$RUNTIME_LOG"
echo "OUTPUT_DIR=$OUTPUT_DIR" | tee -a "$RUNTIME_LOG"
echo "TASKS=$TASKS DEVICE=$DEVICE BATCH_SIZE=$BATCH_SIZE" | tee -a "$RUNTIME_LOG"

python -u code/RSmol/scripts/evaluate_stage3_5_10_5_linear.py \
  --model-path "$MODEL_PATH" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" --batch-size "$BATCH_SIZE" --seed "$SEED" \
  --cache-dir "$CACHE_ROOT" --log-root "$LOG_ROOT" \
  "${TASK_ARGS[@]}" "${MODE_ARGS[@]}" 2>&1 | tee -a "$RUNTIME_LOG"

