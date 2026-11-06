#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
# ``vc submit`` does not reliably inherit the caller's shell environment.
# Explicitly forward the MeSH Stage 4 controls so Gate E/resume paths cannot
# silently fall back to the runtime defaults (Gate D, fresh output, etc.).
RUN_TAG="$(date +%Y%m%d_%H%M%S%N)"
export RSMOL_5_10X2_5_MESH_7SLOT_V2_STAGE4_GATE="${RSMOL_5_10X2_5_MESH_7SLOT_V2_STAGE4_GATE:-FORMAL}"
GATE="$RSMOL_5_10X2_5_MESH_7SLOT_V2_STAGE4_GATE"
export RSMOL_5_10X2_5_MESH_7SLOT_V2_OUTPUT_DIR="${RSMOL_5_10X2_5_MESH_7SLOT_V2_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh_7slot_v2/${GATE,,}_1epoch_bf16_$RUN_TAG}"
REMOTE_CMD="set -euo pipefail; "
for name in \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_STAGE4_GATE \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_WORLD_SIZE \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_MODEL_DIR \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_TOKENIZER_PATH \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_DATA_DIR \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_OUTPUT_DIR \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_RESUME_FROM \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_MICRO_BATCH_SIZE \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_GRADIENT_ACCUMULATION_STEPS \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_CONTEXT_LENGTH \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_MAX_OPTIMIZER_STEPS \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_SCHEDULER_TOTAL_STEPS \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_WARMUP_STEPS \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_STEPS_PER_EPOCH \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_EPOCHS \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_MAX_LR \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_MIN_LR \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_SAVE_EVERY \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_SEED \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_MAX_MICROBATCHES \
  RSMOL_5_10X2_5_MESH_7SLOT_V2_HEARTBEAT_MICRO_INTERVAL \
  TORCH_NCCL_ASYNC_ERROR_HANDLING TORCH_NCCL_DUMP_ON_TIMEOUT \
  TORCH_NCCL_TRACE_BUFFER_SIZE TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG NCCL_DEBUG_SUBSYS; do
  if [[ "${!name+x}" == "x" ]]; then
    printf -v quoted_value '%q' "${!name}"
    REMOTE_CMD+="export ${name}=${quoted_value}; "
  fi
done
REMOTE_CMD+="bash scripts/train_stage4_5_10x2_5_mesh_7slot_v2_ddp.sh"
vc submit -p "${RSMOL_5_10X2_5_MESH_7SLOT_V2_QUEUE:-pdgpu-3090}" -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 -c 64 -m 256G -g 8 -n 1 -j "text-x2-7s-v2-${GATE,,}-$RUN_TAG" -d "$SCRIPT_DIR" JOB=1:1 "$SCRIPT_DIR/log/text_x2_7s_v2_${GATE,,}.$RUN_TAG.JOB.log" --cmd "$REMOTE_CMD"

echo "output: $RSMOL_5_10X2_5_MESH_7SLOT_V2_OUTPUT_DIR"
