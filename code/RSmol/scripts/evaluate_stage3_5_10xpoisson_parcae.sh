#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKOUT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$CHECKOUT_ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"

MODEL="${RSMOL_STAGE3_5_10XPOISSON_PARCAE_MODEL:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10xpoisson_parcae_lr8e-4_mb2_ga64_20260903_172047/checkpoint-009244}"
BENCHMARK_ROOT="${RSMOL_STAGE3_5_10XPOISSON_PARCAE_BENCHMARK_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets}"
OUTPUT_DIR="${RSMOL_STAGE3_5_10XPOISSON_PARCAE_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage3_eval_5_10xpoisson_parcae_009244}"
DEVICE="${RSMOL_STAGE3_5_10XPOISSON_PARCAE_DEVICE:-cuda:0}"
DTYPE="${RSMOL_STAGE3_5_10XPOISSON_PARCAE_DTYPE:-bfloat16}"
BATCH_SIZE="${RSMOL_STAGE3_5_10XPOISSON_PARCAE_BATCH_SIZE:-1}"
SEED="${RSMOL_STAGE3_5_10XPOISSON_PARCAE_SEED:-0}"
TASKS="${RSMOL_STAGE3_5_10XPOISSON_PARCAE_TASKS:-hellaswag mmlu gsm8k arc_easy arc_challenge}"
CACHE_DIR="${RSMOL_STAGE3_5_10XPOISSON_PARCAE_CACHE_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/eval_cache/stage3_5_10xpoisson_parcae_009244}"
LOG_ROOT="${RSMOL_STAGE3_5_10XPOISSON_PARCAE_LOG_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/eval_logs/stage3_5_10xpoisson_parcae_009244}"
LIMIT="${RSMOL_STAGE3_5_10XPOISSON_PARCAE_LIMIT:-}"

for path_value in "$MODEL" "$BENCHMARK_ROOT" "$OUTPUT_DIR" "$CACHE_DIR" "$LOG_ROOT"; do
  case "$path_value" in
    /*) ;;
    *) echo "Stage 3 Parcae paths must be absolute: $path_value" >&2; exit 2 ;;
  esac
done
[[ -n "$TASKS" ]] || { echo "RSMOL_STAGE3_5_10XPOISSON_PARCAE_TASKS must contain at least one task" >&2; exit 2; }

TASK_ARGS=(--tasks)
TASKS_NORMALIZED="${TASKS//,/ }"
read -r -a TASK_LIST <<< "$TASKS_NORMALIZED"
[[ "${#TASK_LIST[@]}" -gt 0 ]] || { echo "RSMOL_STAGE3_5_10XPOISSON_PARCAE_TASKS must contain at least one task" >&2; exit 2; }
TASK_ARGS+=("${TASK_LIST[@]}")

ARGS=(
  python -u code/RSmol/scripts/evaluate_stage3_5_10xpoisson_parcae.py
  --model-path "$MODEL"
  --benchmark-root "$BENCHMARK_ROOT"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --batch-size "$BATCH_SIZE"
  --seed "$SEED"
  --cache-dir "$CACHE_DIR"
  --log-root "$LOG_ROOT"
  "${TASK_ARGS[@]}"
)
if [[ -n "$LIMIT" ]]; then ARGS+=(--limit "$LIMIT"); fi
if [[ "${RSMOL_STAGE3_5_10XPOISSON_PARCAE_SMOKE:-0}" == "1" ]]; then ARGS+=(--smoke); fi
if [[ "${RSMOL_STAGE3_5_10XPOISSON_PARCAE_VALIDATION_ONLY:-0}" == "1" ]]; then ARGS+=(--validation-only); fi
if [[ "${RSMOL_STAGE3_5_10XPOISSON_PARCAE_NO_LOG_SAMPLES:-0}" == "1" ]]; then ARGS+=(--no-log-samples); fi

exec "${ARGS[@]}"
