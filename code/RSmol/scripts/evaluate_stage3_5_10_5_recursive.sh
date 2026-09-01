#!/bin/bash
set -euo pipefail

# Single-GPU runtime for the isolated 5-10-5 middle-loop-2 Stage 3
# evaluator.  This wrapper has its own namespace and never calls the linear
# evaluator or the 15R evaluator.
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

MODEL_PATH="${RSMOL_STAGE3_5_10_5_RECURSIVE_MODEL:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5}"
BENCHMARK_ROOT="${RSMOL_STAGE3_5_10_5_RECURSIVE_BENCHMARK_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets}"
OUTPUT_DIR="${RSMOL_STAGE3_5_10_5_RECURSIVE_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage3-5-10-5-recursive-$(date +%Y%m%d_%H%M%S)-$$}"
DEVICE="${RSMOL_STAGE3_5_10_5_RECURSIVE_DEVICE:-cuda:0}"
BATCH_SIZE="${RSMOL_STAGE3_5_10_5_RECURSIVE_BATCH_SIZE:-1}"
SEED="${RSMOL_STAGE3_5_10_5_RECURSIVE_SEED:-0}"
TASKS="${RSMOL_STAGE3_5_10_5_RECURSIVE_TASKS:-hellaswag,mmlu,gsm8k,arc_easy,arc_challenge}"
CACHE_ROOT="${RSMOL_STAGE3_5_10_5_RECURSIVE_CACHE_ROOT:-/tmp/rsmol-stage3-5-10-5-recursive-$$}"
LOG_ROOT="${RSMOL_STAGE3_5_10_5_RECURSIVE_LOG_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol/log}"

for path_value in "$MODEL_PATH" "$BENCHMARK_ROOT" "$OUTPUT_DIR" "$CACHE_ROOT"; do
  case "$path_value" in
    /*) ;;
    *) echo "recursive 5-10-5 model/benchmark/output/cache paths must be absolute: $path_value" >&2; exit 2 ;;
  esac
done
case "$OUTPUT_DIR" in
  /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM|/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/*)
    echo "recursive 5-10-5 Stage 3 output must be outside the Git checkout: $OUTPUT_DIR" >&2; exit 2 ;;
esac
case "$LOG_ROOT" in
  /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol/log|/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol/log/*) ;;
  /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM|/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/*)
    echo "recursive 5-10-5 checkout logs must be under code/RSmol/log: $LOG_ROOT" >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_DIR" && ( ! -d "$OUTPUT_DIR" || -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ) ]]; then
  echo "Refusing to overwrite non-empty recursive 5-10-5 Stage 3 output: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR" "$LOG_ROOT"

IFS=',' read -r -a RAW_TASKS <<< "$TASKS"
TASK_ARGS=(--tasks)
for task in "${RAW_TASKS[@]}"; do
  task="${task//[[:space:]]/}"
  [[ -n "$task" ]] && TASK_ARGS+=("$task")
done
if [[ "${#TASK_ARGS[@]}" -le 1 ]]; then
  echo "RSMOL_STAGE3_5_10_5_RECURSIVE_TASKS must contain at least one task" >&2
  exit 2
fi

MODE_ARGS=()
[[ -n "${RSMOL_STAGE3_5_10_5_RECURSIVE_DTYPE:-}" ]] && MODE_ARGS+=(--dtype "$RSMOL_STAGE3_5_10_5_RECURSIVE_DTYPE")
[[ "${RSMOL_STAGE3_5_10_5_RECURSIVE_VALIDATION_ONLY:-0}" == "1" ]] && MODE_ARGS+=(--validation-only)
[[ "${RSMOL_STAGE3_5_10_5_RECURSIVE_SMOKE:-0}" == "1" ]] && MODE_ARGS+=(--smoke)
[[ "${RSMOL_STAGE3_5_10_5_RECURSIVE_NO_LOG_SAMPLES:-0}" == "1" ]] && MODE_ARGS+=(--no-log-samples)
[[ -n "${RSMOL_STAGE3_5_10_5_RECURSIVE_LIMIT:-}" ]] && MODE_ARGS+=(--limit "$RSMOL_STAGE3_5_10_5_RECURSIVE_LIMIT")

RUNTIME_LOG="$LOG_ROOT/stage3_5_10_5_recursive_runtime.log"
echo "========== STAGE3 MODEL: recursive_5_10_5_middle_loop2 ==========" | tee -a "$RUNTIME_LOG"
echo "MODEL_PATH=$MODEL_PATH" | tee -a "$RUNTIME_LOG"
echo "BENCHMARK_ROOT=$BENCHMARK_ROOT" | tee -a "$RUNTIME_LOG"
echo "OUTPUT_DIR=$OUTPUT_DIR" | tee -a "$RUNTIME_LOG"
echo "TASKS=$TASKS DEVICE=$DEVICE BATCH_SIZE=$BATCH_SIZE" | tee -a "$RUNTIME_LOG"

python -u code/RSmol/scripts/evaluate_stage3_5_10_5_recursive.py \
  --model-path "$MODEL_PATH" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --cache-dir "$CACHE_ROOT" \
  --log-root "$LOG_ROOT" \
  "${TASK_ARGS[@]}" \
  "${MODE_ARGS[@]}" 2>&1 | tee -a "$RUNTIME_LOG"
