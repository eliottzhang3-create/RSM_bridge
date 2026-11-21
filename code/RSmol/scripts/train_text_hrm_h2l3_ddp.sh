#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT/code/RSmol"
ARGS=(--model-dir "${RSMOL_HRM_MODEL_DIR:?}" --data-dir "${RSMOL_HRM_DATA_DIR:?}" --output-dir "${RSMOL_HRM_OUTPUT_DIR:?}")
[[ -z "${RSMOL_HRM_RESUME_FROM:-}" ]] || ARGS+=(--resume-from "$RSMOL_HRM_RESUME_FROM")
if [[ "${RSMOL_HRM_DRY_RUN:-0}" == 1 ]]; then
  exec python -B -m text_hrm_h2l3.train "${ARGS[@]}" --dry-run
fi
CONDA_BASE="${RSMOL_HRM_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_BASE/envs/rsmol"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1
exec torchrun --standalone --nnodes=1 --nproc_per_node=8 --max_restarts=0 \
  --module text_hrm_h2l3.train "${ARGS[@]}"
