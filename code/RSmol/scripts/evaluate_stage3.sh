#!/bin/bash
set -euo pipefail

# Runtime layer for a single-GPU remote Stage 3 evaluation.  The submit
# wrapper is the only supported entry point for model/data/GPU work.
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

MODEL_SELECTOR="${RSMOL_STAGE3_MODEL:-original}"
ORIGINAL_MODEL="${RSMOL_STAGE3_ORIGINAL_MODEL:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2}"
RECURSIVE_MODEL="${RSMOL_STAGE3_RECURSIVE_MODEL:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-15R}"
BENCHMARK_ROOT="${RSMOL_STAGE3_BENCHMARK_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets}"
OUTPUT_ROOT="${RSMOL_STAGE3_OUTPUT_DIR:-${RSMOL_STAGE3_OUTPUT_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage3-$(date +%Y%m%d_%H%M%S)-$$}}"
DEVICE="${RSMOL_STAGE3_DEVICE:-cuda:0}"
BATCH_SIZE="${RSMOL_STAGE3_BATCH_SIZE:-1}"
SEED="${RSMOL_STAGE3_SEED:-0}"
TASKS="${RSMOL_STAGE3_TASKS:-hellaswag,mmlu,gsm8k,arc_easy,arc_challenge}"
CACHE_ROOT="${RSMOL_STAGE3_CACHE_ROOT:-/tmp/rsmol-stage3-cache-$$}"
# ``SCRIPT_DIR`` is code/RSmol/scripts; the established project log directory
# is its parent (code/RSmol/log), not scripts/log.
LOG_ROOT="${RSMOL_STAGE3_LOG_ROOT:-${RSMOL_STAGE3_SUBMIT_LOG_ROOT:-$REPO_ROOT/code/RSmol/log}}"

# The requested checkout log directory is an explicit exception for vc/task/
# runtime diagnostics only.  Model checkpoints, benchmark data, caches, and
# result directories remain subject to the external-path guards below.
case "$LOG_ROOT" in
  /*) ;;
  *)
    echo "RSMOL_STAGE3_LOG_ROOT must be an absolute path: $LOG_ROOT" >&2
    exit 2
    ;;
esac
case "$LOG_ROOT" in
  "$SCRIPT_DIR/log"|"$SCRIPT_DIR/log/"*|"$SCRIPT_DIR/../log"|"$SCRIPT_DIR/../log/"*|"$REPO_ROOT/code/RSmol/log"|"$REPO_ROOT/code/RSmol/log/"*) ;;
  "$REPO_ROOT"|"$REPO_ROOT"/*|/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM|/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/*)
    echo "RSMOL_STAGE3_LOG_ROOT inside the checkout must be under $SCRIPT_DIR/../log: $LOG_ROOT" >&2
    exit 2
    ;;
esac
mkdir -p "$LOG_ROOT"

case "$OUTPUT_ROOT" in
  /*) ;;
  *)
    echo "RSMOL_STAGE3_OUTPUT_DIR must be an absolute external path: $OUTPUT_ROOT" >&2
    exit 2
    ;;
esac
case "$OUTPUT_ROOT" in
  "$REPO_ROOT"|"$REPO_ROOT"/*|/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM|/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/*)
    echo "RSMOL_STAGE3_OUTPUT_DIR must be outside the Git checkout: $OUTPUT_ROOT" >&2
    exit 2
    ;;
esac

# Reserve a fresh root before either model starts.  The Python evaluator also
# guards each model directory, but this root-level check prevents a second
# invocation from adding files beside an earlier run when an explicit output
# root is reused.
if [[ -e "$OUTPUT_ROOT" ]]; then
  if [[ ! -d "$OUTPUT_ROOT" ]]; then
    echo "RSMOL_STAGE3_OUTPUT_DIR is not a directory: $OUTPUT_ROOT" >&2
    exit 2
  fi
  if [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty Stage 3 output root: $OUTPUT_ROOT" >&2
    exit 2
  fi
fi
mkdir -p "$OUTPUT_ROOT"
if [[ "$MODEL_SELECTOR" == "both" ]]; then
  for model_name in original recursive; do
    model_output="$OUTPUT_ROOT/$model_name"
    if [[ -e "$model_output" ]]; then
      if [[ ! -d "$model_output" ]]; then
        echo "Model output path is not a directory: $model_output" >&2
        exit 2
      fi
      if [[ -n "$(find "$model_output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to overwrite non-empty model output: $model_output" >&2
        exit 2
      fi
    fi
  done
fi

IFS=',' read -r -a RAW_TASK_ARRAY <<< "$TASKS"
TASK_ARRAY=()
for task in "${RAW_TASK_ARRAY[@]}"; do
  task="${task//[[:space:]]/}"
  [[ -n "$task" ]] && TASK_ARRAY+=("$task")
done
if [[ "${#TASK_ARRAY[@]}" -eq 0 ]]; then
  echo "RSMOL_STAGE3_TASKS must contain at least one task" >&2
  exit 2
fi

run_model() {
  local model_name="$1"
  local model_path="$2"
  local model_output="$OUTPUT_ROOT/$model_name"
  local model_cache="$CACHE_ROOT/$model_name"
  local task_args=(--tasks)
  local task
  for task in "${TASK_ARRAY[@]}"; do
    task_args+=("$task")
  done
  echo "========== STAGE3 MODEL: $model_name =========="
  echo "MODEL_PATH=$model_path"
  echo "BENCHMARK_ROOT=$BENCHMARK_ROOT"
  echo "OUTPUT_DIR=$model_output"
  echo "TASKS=$TASKS"
  echo "SEED=$SEED DEVICE=$DEVICE BATCH_SIZE=$BATCH_SIZE LOG_ROOT=$LOG_ROOT"
  local runtime_log="$LOG_ROOT/$model_name/runtime.log"
  mkdir -p "$(dirname "$runtime_log")"
  local mode_args=()
  local reference_args=()
  if [[ "$model_name" == "recursive" ]]; then
    reference_args+=(--reference-model-path "$ORIGINAL_MODEL")
  fi
  if [[ "${RSMOL_STAGE3_VALIDATION_ONLY:-0}" == "1" ]]; then
    mode_args+=(--validation-only)
  fi
  if [[ "${RSMOL_STAGE3_SMOKE:-0}" == "1" ]]; then
    mode_args+=(--smoke)
  fi
  if [[ "${RSMOL_STAGE3_NO_LOG_SAMPLES:-0}" == "1" ]]; then
    mode_args+=(--no-log-samples)
  fi
  if [[ -n "${RSMOL_STAGE3_LIMIT:-}" ]]; then
    mode_args+=(--limit "$RSMOL_STAGE3_LIMIT")
  fi
  python -u code/RSmol/scripts/evaluate_stage3.py \
    --model-path "$model_path" \
    --benchmark-root "$BENCHMARK_ROOT" \
    --output-dir "$model_output" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --cache-dir "$model_cache" \
    --log-root "$LOG_ROOT" \
    "${task_args[@]}" \
    "${reference_args[@]}" \
    "${mode_args[@]}" 2>&1 | tee -a "$runtime_log"
}

case "$MODEL_SELECTOR" in
  original)
    run_model original "$ORIGINAL_MODEL"
    ;;
  recursive)
    run_model recursive "$RECURSIVE_MODEL"
    ;;
  both)
    both_status=0
    original_status=0
    recursive_status=0
    run_model original "$ORIGINAL_MODEL" || original_status=$?
    run_model recursive "$RECURSIVE_MODEL" || recursive_status=$?
    if [[ "$original_status" -ne 0 ]]; then
      both_status="$original_status"
    elif [[ "$recursive_status" -ne 0 ]]; then
      both_status="$recursive_status"
    fi
    exit "$both_status"
    ;;
  *)
    echo "RSMOL_STAGE3_MODEL must be original, recursive, or both; got $MODEL_SELECTOR" >&2
    exit 2
    ;;
esac

echo "[result] status=PASS output_root=$OUTPUT_ROOT"
