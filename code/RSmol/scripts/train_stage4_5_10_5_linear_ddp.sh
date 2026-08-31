#!/bin/bash
set -euo pipefail

# Runtime wrapper for the isolated 5-10-5 linear Stage 4 pipeline.  Only the
# linear-prefixed variables below are accepted, so stale 15R/recursive
# environment variables cannot silently select another architecture.
USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
TASK_TMP="${RSMOL_5_10_5_LINEAR_TASK_TMP:-/tmp/rsmol-stage4-5-10-5-linear-${USER:-unknown}-${SLURM_JOB_ID:-local}}"
mkdir -p "$TASK_TMP/hf-datasets" "$TASK_TMP/hf-home" "$TASK_TMP/modelscope"
export HF_DATASETS_CACHE="$TASK_TMP/hf-datasets"
export HF_HOME="$TASK_TMP/hf-home"
export MODELSCOPE_CACHE="$TASK_TMP/modelscope"
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

GATE="${RSMOL_5_10_5_LINEAR_STAGE4_GATE:-D}"
GATE="${GATE^^}"
WORLD_SIZE="${RSMOL_5_10_5_LINEAR_WORLD_SIZE:-8}"
MODEL_PATH="${RSMOL_5_10_5_LINEAR_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5-linear}"
RESUME_PATH="${RSMOL_5_10_5_LINEAR_RESUME_FROM:-}"
DATA_DIR="${RSMOL_5_10_5_LINEAR_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset}"
OUTPUT_DIR="${RSMOL_5_10_5_LINEAR_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5_linear/$(date +%Y%m%d_%H%M%S)}"
REPORT_PATH="${RSMOL_5_10_5_LINEAR_REPORT:-$OUTPUT_DIR/stage4_report.json}"
AUDIT_REPORT="${RSMOL_5_10_5_LINEAR_AUDIT_REPORT:-}"
TOKENIZER_PATH="${RSMOL_5_10_5_LINEAR_TOKENIZER_PATH:-}"
DRY_RUN="${RSMOL_5_10_5_LINEAR_DRY_RUN:-0}"

if [[ "$GATE" == "B" || "$GATE" == "C" ]]; then
  echo "Stage 4 5-10-5 linear Gate $GATE is unsupported; reuse the external Gate B JSON and run Gate A, D, E, or FORMAL." >&2
  exit 2
fi
if [[ "$GATE" == "FORMAL" ]]; then
  MAX_OPTIMIZER_STEPS=9244
  WARMUP_STEPS=463
  SCHEDULER_TOTAL_STEPS=9244
  SAVE_EVERY=500
else
  MAX_OPTIMIZER_STEPS="${RSMOL_5_10_5_LINEAR_MAX_OPTIMIZER_STEPS:-10}"
  WARMUP_STEPS="${RSMOL_5_10_5_LINEAR_WARMUP_STEPS:-0}"
  SCHEDULER_TOTAL_STEPS="${RSMOL_5_10_5_LINEAR_SCHEDULER_TOTAL_STEPS:-}"
  SAVE_EVERY="${RSMOL_5_10_5_LINEAR_SAVE_EVERY:-500}"
fi

if [[ "$GATE" == "D" && "$DRY_RUN" != "1" && -z "$AUDIT_REPORT" ]]; then
  echo "Run Gate B first and set RSMOL_5_10_5_LINEAR_AUDIT_REPORT to its external JSON report" >&2
  exit 2
fi
if [[ ("$GATE" == "E" || "$GATE" == "FORMAL") && "$DRY_RUN" != "1" ]]; then
  if [[ -z "$RESUME_PATH" ]]; then
    if [[ "$GATE" == "E" ]]; then
      echo "Gate E requires RSMOL_5_10_5_LINEAR_RESUME_FROM" >&2
      exit 2
    fi
  elif [[ ! -d "$RESUME_PATH" || ! -f "$RESUME_PATH/config.json" || ! -f "$RESUME_PATH/training_state.pt" || ! -f "$RESUME_PATH/checkpoint_complete.json" ]]; then
    echo "$GATE resume path is incomplete: $RESUME_PATH" >&2
    exit 2
  fi
fi

echo "========== STAGE 4 5-10-5 LINEAR ${GATE} =========="
echo "ACTIVE_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "PYTHON=$(which python)"
echo "GATE=$GATE WORLD_SIZE=$WORLD_SIZE"
echo "MODEL_PATH=$MODEL_PATH"
echo "RESUME_PATH=$RESUME_PATH"
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "REPORT_PATH=$REPORT_PATH"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

ARGS=(
  --gate "$GATE"
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --report-path "$REPORT_PATH"
  --world-size "$WORLD_SIZE"
  --micro-batch-size "${RSMOL_5_10_5_LINEAR_MICRO_BATCH_SIZE:-8}"
  --gradient-accumulation-steps "${RSMOL_5_10_5_LINEAR_GRADIENT_ACCUMULATION_STEPS:-16}"
  --learning-rate "${RSMOL_5_10_5_LINEAR_LEARNING_RATE:-2e-4}"
  --max-lr "${RSMOL_5_10_5_LINEAR_MAX_LR:-2e-4}"
  --min-lr "${RSMOL_5_10_5_LINEAR_MIN_LR:-2e-5}"
  --context-length "${RSMOL_5_10_5_LINEAR_CONTEXT_LENGTH:-1024}"
  --warmup-steps "$WARMUP_STEPS"
  --max-optimizer-steps "$MAX_OPTIMIZER_STEPS"
  --formal-optimizer-steps "${RSMOL_5_10_5_LINEAR_FORMAL_OPTIMIZER_STEPS:-9244}"
  --log-interval-steps "${RSMOL_5_10_5_LINEAR_LOG_INTERVAL_STEPS:-10}"
  --seed "${RSMOL_5_10_5_LINEAR_SEED:-0}"
  --weight-decay "${RSMOL_5_10_5_LINEAR_WEIGHT_DECAY:-0.1}"
  --max-grad-norm "${RSMOL_5_10_5_LINEAR_MAX_GRAD_NORM:-1.0}"
  --record-buffer-size "${RSMOL_5_10_5_LINEAR_RECORD_BUFFER_SIZE:-4096}"
  --save-every "$SAVE_EVERY"
  --checkpoint-retention "${RSMOL_5_10_5_LINEAR_CHECKPOINT_RETENTION:-3}"
  --monitor-interval-seconds "${RSMOL_5_10_5_LINEAR_MONITOR_INTERVAL:-60}"
  --backend "${RSMOL_5_10_5_LINEAR_BACKEND:-nccl}"
)
if [[ -n "$AUDIT_REPORT" ]]; then ARGS+=(--audit-report "$AUDIT_REPORT"); fi
if [[ -n "$SCHEDULER_TOTAL_STEPS" ]]; then ARGS+=(--scheduler-total-steps "$SCHEDULER_TOTAL_STEPS"); fi
if [[ -n "$MODEL_PATH" ]]; then ARGS+=(--model-path "$MODEL_PATH"); fi
if [[ -n "$TOKENIZER_PATH" ]]; then ARGS+=(--tokenizer-path "$TOKENIZER_PATH"); fi
if [[ -n "$RESUME_PATH" ]]; then ARGS+=(--resume-from "$RESUME_PATH"); fi
if [[ "${RSMOL_5_10_5_LINEAR_ALLOW_NON8:-0}" == "1" ]]; then ARGS+=(--allow-non8); fi
if [[ "$DRY_RUN" == "1" ]]; then ARGS+=(--dry-run); fi

torchrun --standalone --nproc_per_node="$WORLD_SIZE" \
  code/RSmol/scripts/train_stage4_5_10_5_linear_ddp.py "${ARGS[@]}"
echo "[result] status=PASS"
