#!/bin/bash
set -euo pipefail

# Dedicated formal runtime. Only this route's IO variables are read.
CONDA_BASE="${RSMOL_5_20X2_5_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONFAULTHANDLER=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
TASK_TMP="${RSMOL_5_20X2_5_TASK_TMP:-/tmp/rsmol-text-5-20x2-5-${USER:-unknown}-${SLURM_JOB_ID:-local}-$$}"
mkdir -p "$TASK_TMP/hf-datasets" "$TASK_TMP/hf-home" "$TASK_TMP/modelscope"
export HF_DATASETS_CACHE="$TASK_TMP/hf-datasets" HF_HOME="$TASK_TMP/hf-home" MODELSCOPE_CACHE="$TASK_TMP/modelscope"
MODEL_DIR="${RSMOL_5_20X2_5_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-20x2-5}"
DATA_DIR="${RSMOL_5_20X2_5_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset}"
OUTPUT_DIR="${RSMOL_5_20X2_5_OUTPUT_DIR:?Set a new external formal output directory}"
RESUME_FROM="${RSMOL_5_20X2_5_RESUME_FROM:-}"
AUDIT_REPORT="${RSMOL_5_20X2_5_AUDIT_REPORT:-}"
DRY_RUN="${RSMOL_5_20X2_5_DRY_RUN:-0}"
if [[ "$DRY_RUN" != 1 ]]; then
  if [[ -n "$RESUME_FROM" ]]; then
    for name in config.json training_state.pt checkpoint_complete.json; do
      [[ -f "$RESUME_FROM/$name" ]] || { echo "Incomplete resume: $RESUME_FROM/$name" >&2; exit 2; }
    done
  else
    [[ -f "$MODEL_DIR/recursive_conversion_metadata.json" ]] || { echo "Run CPU conversion first: $MODEL_DIR" >&2; exit 2; }
    [[ -n "$AUDIT_REPORT" && -f "$AUDIT_REPORT" ]] || { echo "CPU Gate B audit required" >&2; exit 2; }
  fi
fi
ARGS=(
  --gate FORMAL --model-path "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR"
  --world-size 8 --micro-batch-size 4 --gradient-accumulation-steps 16
  --learning-rate 8e-4 --max-lr 8e-4 --min-lr 8e-5 --context-length 1024
  --max-optimizer-steps 18488 --formal-optimizer-steps 18488
  --scheduler-total-steps 18488 --warmup-steps 925
  --seed 0 --weight-decay 0.1 --max-grad-norm 1.0 --record-buffer-size 4096
  --save-every 500 --checkpoint-retention 3 --log-interval-steps 10
  --monitor-interval-seconds 60 --backend nccl
)
[[ -z "$AUDIT_REPORT" ]] || ARGS+=(--audit-report "$AUDIT_REPORT")
[[ -z "$RESUME_FROM" ]] || ARGS+=(--resume-from "$RESUME_FROM")
if [[ "$DRY_RUN" == 1 ]]; then
  python -u code/RSmol/scripts/train_stage4_5_20x2_5_ddp.py "${ARGS[@]}" --dry-run
else
  torchrun --standalone --nproc_per_node=8 code/RSmol/scripts/train_stage4_5_20x2_5_ddp.py "${ARGS[@]}"
fi
