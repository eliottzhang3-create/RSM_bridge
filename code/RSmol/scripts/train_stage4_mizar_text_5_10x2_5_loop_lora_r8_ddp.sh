#!/bin/bash
set -euo pipefail
CONDA_BASE="${RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_BASE/envs/rsmol"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONFAULTHANDLER=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
TASK_TMP="${RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_TASK_TMP:-/tmp/rsmol-mizar-text-51025-lora8-${USER:-unknown}-${SLURM_JOB_ID:-local}-$$}"
mkdir -p "$TASK_TMP/hf-datasets" "$TASK_TMP/hf-home" "$TASK_TMP/modelscope"
export HF_DATASETS_CACHE="$TASK_TMP/hf-datasets" HF_HOME="$TASK_TMP/hf-home" MODELSCOPE_CACHE="$TASK_TMP/modelscope"
MODEL_DIR="${RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-text-5-10x2-5-loop-lora-r8}"
DATA_DIR="${RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_DATA_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset}"
OUTPUT_DIR="${RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_OUTPUT_DIR:?Set a new external output directory}"
MAX_LR="${RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_MAX_LR:-8e-4}"
MIN_LR="${RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_MIN_LR:-}"
RESUME_FROM="${RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_RESUME_FROM:-}"
DRY_RUN="${RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_DRY_RUN:-0}"
if [[ "$DRY_RUN" != 1 ]]; then
  if [[ -n "$RESUME_FROM" ]]; then
    for name in config.json training_state.pt checkpoint_complete.json; do
      [[ -f "$RESUME_FROM/$name" ]] || { echo "Incomplete resume: $RESUME_FROM/$name" >&2; exit 2; }
    done
  else
    [[ -f "$MODEL_DIR/config.json" ]] || { echo "Model config missing: $MODEL_DIR/config.json" >&2; exit 2; }
  fi
fi
ARGS=(
  --gate FORMAL --model-path "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR"
  --world-size 8 --micro-batch-size 8 --gradient-accumulation-steps 16
  --learning-rate "$MAX_LR" --max-lr "$MAX_LR" --context-length 1024
  --max-optimizer-steps 9244 --formal-optimizer-steps 9244
  --scheduler-total-steps 9244 --warmup-steps 463
  --seed 0 --weight-decay 0.1 --max-grad-norm 1.0 --record-buffer-size 4096
  --save-every 500 --checkpoint-retention 3 --log-interval-steps 10
  --monitor-interval-seconds 60 --backend nccl
)
[[ -z "$MIN_LR" ]] || ARGS+=(--min-lr "$MIN_LR")
[[ -z "$RESUME_FROM" ]] || ARGS+=(--resume-from "$RESUME_FROM")
echo "[mizar-text-loop-lora-r8] max_lr=$MAX_LR min_lr=${MIN_LR:-0.1*max_lr}; one epoch: 9244 updates, GBS1024"
if [[ "$DRY_RUN" == 1 ]]; then
  python -u code/RSmol/scripts/train_stage4_mizar_text_5_10x2_5_loop_lora_r8_ddp.py "${ARGS[@]}" --dry-run
else
  torchrun --standalone --nproc_per_node=8 code/RSmol/scripts/train_stage4_mizar_text_5_10x2_5_loop_lora_r8_ddp.py "${ARGS[@]}"
fi
