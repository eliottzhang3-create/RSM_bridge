#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
# ``vc submit`` does not reliably inherit the caller's shell environment.
# Explicitly forward the MeSH Stage 4 controls so Gate E/resume paths cannot
# silently fall back to the runtime defaults (Gate D, fresh output, etc.).
REMOTE_CMD="set -euo pipefail; "
for name in \
  RSMOL_5_10X2_5_MESH_STAGE4_GATE \
  RSMOL_5_10X2_5_MESH_WORLD_SIZE \
  RSMOL_5_10X2_5_MESH_MODEL_DIR \
  RSMOL_5_10X2_5_MESH_TOKENIZER_PATH \
  RSMOL_5_10X2_5_MESH_DATA_DIR \
  RSMOL_5_10X2_5_MESH_OUTPUT_DIR \
  RSMOL_5_10X2_5_MESH_RESUME_FROM \
  RSMOL_5_10X2_5_MESH_MICRO_BATCH_SIZE \
  RSMOL_5_10X2_5_MESH_GRADIENT_ACCUMULATION_STEPS \
  RSMOL_5_10X2_5_MESH_CONTEXT_LENGTH \
  RSMOL_5_10X2_5_MESH_MAX_OPTIMIZER_STEPS \
  RSMOL_5_10X2_5_MESH_SCHEDULER_TOTAL_STEPS \
  RSMOL_5_10X2_5_MESH_WARMUP_STEPS \
  RSMOL_5_10X2_5_MESH_MAX_LR \
  RSMOL_5_10X2_5_MESH_MIN_LR \
  RSMOL_5_10X2_5_MESH_SAVE_EVERY \
  RSMOL_5_10X2_5_MESH_SEED \
  RSMOL_5_10X2_5_MESH_MAX_MICROBATCHES; do
  if [[ "${!name+x}" == "x" ]]; then
    printf -v quoted_value '%q' "${!name}"
    REMOTE_CMD+="export ${name}=${quoted_value}; "
  fi
done
REMOTE_CMD+="bash scripts/train_stage4_5_10x2_5_mesh_ddp.sh"
vc submit -p "${RSMOL_5_10X2_5_MESH_QUEUE:-pdgpu-3090}" -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 -c 32 -m 256G -g 8 -n 1 -j "stage4-5-10x2-5-mesh-$(date +%m%d%H%M)" -d "$SCRIPT_DIR" JOB=1:1 "$SCRIPT_DIR/log/stage4_5_10x2_5_mesh.JOB.log" --cmd "$REMOTE_CMD"
