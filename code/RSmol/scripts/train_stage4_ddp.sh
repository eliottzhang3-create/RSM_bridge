#!/bin/bash
set -euo pipefail

# Remote runtime wrapper.  The submit wrapper is the only supported entry
# point on the cluster; this script is intentionally ordinary, reviewable bash
# once vc has placed it in the job container.
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
# If datasets is imported by a future diagnostic, all caches remain task-local
# and are never created under the Git checkout or transfer directory.
TASK_TMP="${RSMOL_STAGE4_TASK_TMP:-/tmp/rsmol-stage4-${USER:-unknown}-${SLURM_JOB_ID:-local}}"
mkdir -p "$TASK_TMP/hf-datasets" "$TASK_TMP/hf-home" "$TASK_TMP/modelscope"
export HF_DATASETS_CACHE="$TASK_TMP/hf-datasets"
export HF_HOME="$TASK_TMP/hf-home"
export MODELSCOPE_CACHE="$TASK_TMP/modelscope"
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

GATE="${RSMOL_STAGE4_GATE:-D}"
WORLD_SIZE="${RSMOL_STAGE4_WORLD_SIZE:-8}"
MODEL_PATH="${RSMOL_RECURSIVE_OUTPUT_DIR:-}"
RESUME_PATH="${RSMOL_STAGE4_RESUME_FROM:-}"
DATA_DIR="${RSMOL_STAGE4_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset}"
OUTPUT_DIR="${RSMOL_STAGE4_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4/$(date +%Y%m%d_%H%M%S)}"
REPORT_PATH="${RSMOL_STAGE4_REPORT:-$OUTPUT_DIR/stage4_report.json}"

if [[ "$GATE" == "B" ]]; then
  echo "Gate B is CPU-only and is not handled by the DDP runtime; run python -u code/RSmol/scripts/audit_stage4_dataset.py directly." >&2
  exit 2
fi

echo "========== STAGE 4 DDP ${GATE} =========="
echo "ACTIVE_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "PYTHON=$(which python)"
echo "GATE=$GATE"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "MODEL_PATH=$MODEL_PATH"
echo "RESUME_PATH=$RESUME_PATH"
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "REPORT_PATH=$REPORT_PATH"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_HOME=$HF_HOME"
echo "MODELSCOPE_CACHE=$MODELSCOPE_CACHE"

if [[ "$GATE" == "E" && "${RSMOL_STAGE4_DRY_RUN:-0}" != "1" ]]; then
  if [[ -z "$RESUME_PATH" ]]; then
    echo "Gate E requires RSMOL_STAGE4_RESUME_FROM: an external complete Stage 4 checkpoint" >&2
    exit 2
  fi
  if [[ ! -d "$RESUME_PATH" || ! -f "$RESUME_PATH/config.json" || ! -f "$RESUME_PATH/training_state.pt" || ! -f "$RESUME_PATH/checkpoint_complete.json" ]]; then
    echo "Gate E resume path is incomplete; expected config.json, training_state.pt, checkpoint_complete.json: $RESUME_PATH" >&2
    exit 2
  fi
elif [[ "$GATE" != "A" && "${RSMOL_STAGE4_DRY_RUN:-0}" != "1" && -z "$MODEL_PATH" ]]; then
  echo "Set RSMOL_RECURSIVE_OUTPUT_DIR to the external Stage 1 recursive checkpoint" >&2
  exit 2
fi
if [[ ("$GATE" == "C" || "$GATE" == "D") && "${RSMOL_STAGE4_DRY_RUN:-0}" != "1" && -z "${RSMOL_STAGE4_AUDIT_REPORT:-}" ]]; then
  echo "Run Gate B first and set RSMOL_STAGE4_AUDIT_REPORT to its external JSON report" >&2
  exit 2
fi
ARGS=(
  --gate "$GATE"
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --report-path "$REPORT_PATH"
  --world-size "$WORLD_SIZE"
  --micro-batch-size "${RSMOL_STAGE4_MICRO_BATCH_SIZE:-8}"
  --gradient-accumulation-steps "${RSMOL_STAGE4_GRADIENT_ACCUMULATION_STEPS:-16}"
  --learning-rate "${RSMOL_STAGE4_LEARNING_RATE:-2e-4}"
  --max-lr "${RSMOL_STAGE4_MAX_LR:-2e-4}"
  --min-lr "${RSMOL_STAGE4_MIN_LR:-2e-5}"
  --context-length "${RSMOL_STAGE4_CONTEXT_LENGTH:-1024}"
  --warmup-steps "${RSMOL_STAGE4_WARMUP_STEPS:-0}"
  --max-optimizer-steps "${RSMOL_STAGE4_MAX_OPTIMIZER_STEPS:-10}"
  --formal-optimizer-steps "${RSMOL_STAGE4_FORMAL_OPTIMIZER_STEPS:-9244}"
  --log-interval-steps "${RSMOL_STAGE4_LOG_INTERVAL_STEPS:-10}"
  --seed "${RSMOL_STAGE4_SEED:-0}"
  --weight-decay "${RSMOL_STAGE4_WEIGHT_DECAY:-0.1}"
  --max-grad-norm "${RSMOL_STAGE4_MAX_GRAD_NORM:-1.0}"
  --record-buffer-size "${RSMOL_STAGE4_RECORD_BUFFER_SIZE:-4096}"
  --save-every "${RSMOL_STAGE4_SAVE_EVERY:-500}"
  --monitor-interval-seconds "${RSMOL_STAGE4_MONITOR_INTERVAL:-60}"
  --backend "${RSMOL_STAGE4_BACKEND:-nccl}"
)
if [[ -n "${RSMOL_STAGE4_AUDIT_REPORT:-}" ]]; then
  ARGS+=(--audit-report "$RSMOL_STAGE4_AUDIT_REPORT")
fi
if [[ -n "${RSMOL_STAGE4_SCHEDULER_TOTAL_STEPS:-}" ]]; then
  ARGS+=(--scheduler-total-steps "$RSMOL_STAGE4_SCHEDULER_TOTAL_STEPS")
fi
if [[ -n "$MODEL_PATH" ]]; then
  ARGS+=(--model-path "$MODEL_PATH")
fi
if [[ -n "${RSMOL_STAGE4_TOKENIZER_PATH:-}" ]]; then
  ARGS+=(--tokenizer-path "$RSMOL_STAGE4_TOKENIZER_PATH")
fi
if [[ -n "$RESUME_PATH" ]]; then
  ARGS+=(--resume-from "$RESUME_PATH")
fi
if [[ "${RSMOL_STAGE4_ALLOW_NON8:-0}" == "1" ]]; then
  ARGS+=(--allow-non8)
fi
if [[ "${RSMOL_STAGE4_DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

torchrun --standalone --nproc_per_node="$WORLD_SIZE" \
  code/RSmol/scripts/train_stage4_ddp.py "${ARGS[@]}"
echo "[result] status=PASS"
