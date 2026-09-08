#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKOUT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$CHECKOUT_ROOT"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
if [[ -f "$USER_CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$USER_CONDA_BASE/envs/rsmol"
fi

MODEL="${RSMOL_STAGE3_5_10X2_5_MESH_MODEL:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_resume_000500_nonfatal_router_20260907_115533/checkpoint-009244}"
TOKENIZER="${RSMOL_STAGE3_5_10X2_5_MESH_TOKENIZER_PATH:-${RSMOL_STAGE3_5_10X2_5_MESH_TOKENIZER:-}}"
BENCHMARK_ROOT="${RSMOL_STAGE3_5_10X2_5_MESH_BENCHMARK_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets}"
OUTPUT_DIR="${RSMOL_STAGE3_5_10X2_5_MESH_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage3_eval_5_10x2_5_mesh_009244}"
DEVICE="${RSMOL_STAGE3_5_10X2_5_MESH_DEVICE:-cuda:0}"
DTYPE="${RSMOL_STAGE3_5_10X2_5_MESH_DTYPE:-bfloat16}"
BATCH_SIZE="${RSMOL_STAGE3_5_10X2_5_MESH_BATCH_SIZE:-1}"
SEED="${RSMOL_STAGE3_5_10X2_5_MESH_SEED:-0}"
TASKS="${RSMOL_STAGE3_5_10X2_5_MESH_TASKS:-hellaswag mmlu gsm8k arc_easy arc_challenge}"
CACHE_DIR="${RSMOL_STAGE3_5_10X2_5_MESH_CACHE_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/eval_cache/stage3_5_10x2_5_mesh_009244}"
LOG_ROOT="${RSMOL_STAGE3_5_10X2_5_MESH_LOG_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/eval_logs/stage3_5_10x2_5_mesh_009244}"
REPORT_PATH="${RSMOL_STAGE3_5_10X2_5_MESH_REPORT_PATH:-}"
LIMIT="${RSMOL_STAGE3_5_10X2_5_MESH_LIMIT:-}"
MAX_NEW_TOKENS="${RSMOL_STAGE3_5_10X2_5_MESH_MAX_NEW_TOKENS:-2}"

for path_value in "$MODEL" "$BENCHMARK_ROOT" "$OUTPUT_DIR" "$CACHE_DIR" "$LOG_ROOT"; do
  case "$path_value" in /*) ;; *) echo "MeSH Stage 3 paths must be absolute: $path_value" >&2; exit 2 ;; esac
done
[[ -z "$TOKENIZER" ]] || case "$TOKENIZER" in /*) ;; *) echo "MeSH tokenizer path must be absolute: $TOKENIZER" >&2; exit 2 ;; esac
[[ -n "$TASKS" ]] || { echo "RSMOL_STAGE3_5_10X2_5_MESH_TASKS must contain at least one task" >&2; exit 2; }

TASK_ARGS=(--tasks)
TASKS_NORMALIZED="${TASKS//,/ }"
read -r -a TASK_LIST <<< "$TASKS_NORMALIZED"
[[ "${#TASK_LIST[@]}" -gt 0 ]] || { echo "RSMOL_STAGE3_5_10X2_5_MESH_TASKS must contain at least one task" >&2; exit 2; }
TASK_ARGS+=("${TASK_LIST[@]}")

ARGS=(
  python -u code/RSmol/scripts/evaluate_stage3_5_10x2_5_mesh.py
  --model-path "$MODEL"
  --benchmark-root "$BENCHMARK_ROOT"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --batch-size "$BATCH_SIZE"
  --seed "$SEED"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --cache-dir "$CACHE_DIR"
  --log-root "$LOG_ROOT"
  "${TASK_ARGS[@]}"
)
[[ -z "$TOKENIZER" ]] || ARGS+=(--tokenizer-path "$TOKENIZER")
[[ -z "$REPORT_PATH" ]] || ARGS+=(--report-path "$REPORT_PATH")
[[ -z "$LIMIT" ]] || ARGS+=(--limit "$LIMIT")
[[ "${RSMOL_STAGE3_5_10X2_5_MESH_SMOKE:-0}" == "1" ]] && ARGS+=(--smoke)
[[ "${RSMOL_STAGE3_5_10X2_5_MESH_VALIDATION_ONLY:-0}" == "1" ]] && ARGS+=(--validation-only)
[[ "${RSMOL_STAGE3_5_10X2_5_MESH_NO_LOG_SAMPLES:-0}" == "1" ]] && ARGS+=(--no-log-samples)

exec "${ARGS[@]}"
