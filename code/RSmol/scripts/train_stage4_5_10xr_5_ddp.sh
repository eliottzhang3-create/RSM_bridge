#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONFAULTHANDLER=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1 TORCH_DISTRIBUTED_DEBUG=DETAIL

GATE="${RSMOL_5_10XR_5_STAGE4_GATE:-D}"
GATE="${GATE^^}"
TASK_TMP="${RSMOL_5_10XR_5_TASK_TMP:-/tmp/rsmol-5-10xr-5-${USER:-unknown}-${SLURM_JOB_ID:-local}}"
mkdir -p "$TASK_TMP/hf-datasets" "$TASK_TMP/hf-home" "$TASK_TMP/modelscope"
export HF_DATASETS_CACHE="$TASK_TMP/hf-datasets" HF_HOME="$TASK_TMP/hf-home" MODELSCOPE_CACHE="$TASK_TMP/modelscope"

MODEL_PATH="${RSMOL_5_10XR_5_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10xr-5-poisson}"
TOKENIZER_PATH="${RSMOL_5_10XR_5_TOKENIZER_PATH:-}"
DATA_DIR="${RSMOL_5_10XR_5_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset}"
OUTPUT_DIR="${RSMOL_5_10XR_5_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10xr_5_poisson/$(date +%Y%m%d_%H%M%S)}"
REPORT_PATH="${RSMOL_5_10XR_5_REPORT:-$OUTPUT_DIR/stage4_report.json}"
RESUME_PATH="${RSMOL_5_10XR_5_RESUME_FROM:-}"
WORLD_SIZE="${RSMOL_5_10XR_5_WORLD_SIZE:-8}"
AUDIT_REPORT="${RSMOL_5_10XR_5_AUDIT_REPORT:-}"

if [[ "$GATE" == "B" || "$GATE" == "C" ]]; then
  echo "Stage 4 5-10xr-5 Gate $GATE is unsupported; reuse the existing Gate B JSON." >&2
  exit 2
fi
if [[ "$GATE" != "A" && -z "$RESUME_PATH" && ! -d "$MODEL_PATH" ]]; then
  echo "Missing external 5-10xr-5 model directory: $MODEL_PATH" >&2
  exit 2
fi
if [[ ("$GATE" == "D" || "$GATE" == "FORMAL") && -z "$RESUME_PATH" && -z "$AUDIT_REPORT" && "${RSMOL_5_10XR_5_DRY_RUN:-0}" != "1" ]]; then
  echo "$GATE requires RSMOL_5_10XR_5_AUDIT_REPORT from the existing Gate B audit (unless resuming)" >&2
  exit 2
fi
if [[ ("$GATE" == "D" || "$GATE" == "FORMAL") && -n "$AUDIT_REPORT" && ! -f "$AUDIT_REPORT" && "${RSMOL_5_10XR_5_DRY_RUN:-0}" != "1" ]]; then
  echo "$GATE Gate-B audit report does not exist: $AUDIT_REPORT" >&2
  exit 2
fi

if [[ "$GATE" == "FORMAL" ]]; then
  MAX_STEPS=9244; WARMUP_STEPS=463; SCHEDULER_STEPS=9244; SAVE_EVERY=500
else
  MAX_STEPS="${RSMOL_5_10XR_5_MAX_OPTIMIZER_STEPS:-10}"
  WARMUP_STEPS="${RSMOL_5_10XR_5_WARMUP_STEPS:-0}"
  SCHEDULER_STEPS="${RSMOL_5_10XR_5_SCHEDULER_TOTAL_STEPS:-$MAX_STEPS}"
  SAVE_EVERY="${RSMOL_5_10XR_5_SAVE_EVERY:-500}"
fi

ARGS=(--gate "$GATE" --world-size "$WORLD_SIZE" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" --report-path "$REPORT_PATH"
  --micro-batch-size 2 --gradient-accumulation-steps 64 --learning-rate 8e-4 --max-lr 8e-4 --min-lr 8e-5
  --context-length 1024 --warmup-steps "$WARMUP_STEPS" --max-optimizer-steps "$MAX_STEPS"
  --formal-optimizer-steps 9244 --scheduler-total-steps "$SCHEDULER_STEPS" --log-interval-steps 10
  --seed "${RSMOL_5_10XR_5_SEED:-0}" --weight-decay 0.1 --max-grad-norm "${RSMOL_5_10XR_5_MAX_GRAD_NORM:-1.0}"
  --record-buffer-size "${RSMOL_5_10XR_5_RECORD_BUFFER_SIZE:-4096}" --save-every "$SAVE_EVERY"
  --checkpoint-retention 3 --backend nccl)
[[ -n "$MODEL_PATH" ]] && ARGS+=(--model-path "$MODEL_PATH")
[[ -n "$TOKENIZER_PATH" ]] && ARGS+=(--tokenizer-path "$TOKENIZER_PATH")
[[ -n "$RESUME_PATH" ]] && ARGS+=(--resume-from "$RESUME_PATH")
[[ -n "$AUDIT_REPORT" ]] && ARGS+=(--audit-report "$AUDIT_REPORT")
[[ "${RSMOL_5_10XR_5_ALLOW_NON8:-0}" == "1" ]] && ARGS+=(--allow-non8)
[[ "${RSMOL_5_10XR_5_DRY_RUN:-0}" == "1" ]] && ARGS+=(--dry-run)

echo "========== STAGE 4 5-10xr-5 $GATE =========="
echo "MODEL_PATH=$MODEL_PATH OUTPUT_DIR=$OUTPUT_DIR WORLD_SIZE=$WORLD_SIZE MAX_STEPS=$MAX_STEPS WARMUP_STEPS=$WARMUP_STEPS"
torchrun --standalone --nproc_per_node="$WORLD_SIZE" code/RSmol/scripts/train_stage4_5_10xr_5_ddp.py "${ARGS[@]}"
echo "[result] status=PASS"
