#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
PREFIX=RSMOL_MIZAR_TEXT_7_14X3_7_MESH
read_var() { local key="${PREFIX}_$1"; printf '%s' "${!key:-$2}"; }
GATE="$(read_var STAGE4_GATE FORMAL)"
WORLD="$(read_var WORLD_SIZE 8)"
MICRO="$(read_var MICRO_BATCH_SIZE 8)"
GA="$(read_var GRADIENT_ACCUMULATION_STEPS 16)"
MODEL="$(read_var MODEL_DIR /hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-text-7-14x3-7-mesh-6slot)"
DATA="$(read_var DATA_DIR /hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset)"
OUTPUT="$(read_var OUTPUT_DIR /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x3_7_mesh/formal_$(date +%Y%m%d_%H%M%S)_$$)"
RESUME="$(read_var RESUME_FROM '')"
STEPS=9244
case "$GATE" in
  FORMAL)
    [[ "$WORLD" == 8 && ( "$MICRO:$GA" == 8:16 || "$MICRO:$GA" == 4:32 ) ]] || { echo 'FORMAL requires 8 ranks and micro8/GA16 or micro4/GA32' >&2; exit 2; };;
  D) STEPS=10;;
  E) STEPS=2; [[ -n "$RESUME" ]] || { echo 'Gate E requires RESUME_FROM' >&2; exit 2; };;
  A) STEPS=2;;
  *) echo "Unknown gate: $GATE" >&2; exit 2;;
esac
ARGS=(--gate "$GATE" --model-path "$MODEL" --data-dir "$DATA" --output-dir "$OUTPUT"
  --world-size "$WORLD" --micro-batch-size "$MICRO" --gradient-accumulation-steps "$GA"
  --context-length 1024 --max-optimizer-steps "$STEPS" --scheduler-total-steps 9244
  --warmup-steps 463 --steps-per-epoch 9244 --epochs 1 --max-lr 6e-4 --min-lr 6e-5
  --save-every 400 --seed "$(read_var SEED 0)")
[[ -z "$RESUME" ]] || ARGS+=(--resume-from "$RESUME")
COMMAND=(torchrun --standalone --nproc_per_node="$WORLD" code/RSmol/scripts/train_stage4_mizar_text_7_14x3_7_mesh_ddp.py "${ARGS[@]}" "$@")
if [[ "$(read_var DRY_RUN 0)" == 1 ]]; then
  printf 'COMMAND '; printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0
fi
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || { echo "Refusing existing output: $OUTPUT" >&2; exit 2; }
if [[ "$GATE" != A ]]; then
  [[ -d "$DATA" ]] || { echo "Missing persistent data: $DATA" >&2; exit 2; }
fi
CONDA_BASE="$(read_var CONDA_BASE /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_BASE/envs/rsmol"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-200000}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
echo "[Mizar MeSH] $GATE; 28 physical / 56 executions; 6 slots / 4 groups; BF16 forward"
echo "[Mizar MeSH] world=$WORLD micro=$MICRO GA=$GA; output=$OUTPUT; resume=${RESUME:-none}"
exec "${COMMAND[@]}"
