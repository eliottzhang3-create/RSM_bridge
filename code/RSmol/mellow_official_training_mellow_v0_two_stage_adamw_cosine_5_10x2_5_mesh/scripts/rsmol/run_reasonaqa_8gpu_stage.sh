#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 7 ]]; then
  echo "usage: $0 <stage1|stage2> <smoke|resume|formal> <PASS audit-report.json> <mapping.jsonl> <output-dir> [source-checkpoint] [smoke-rows=5120]" >&2
  exit 2
fi

STAGE="$1"
MODE="$2"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
AUDIT_REPORT=$(readlink -f "$3")
MAPPING_JSONL=$(readlink -f "$4")
OUTPUT_DIR=$(readlink -m "$5")
SOURCE_CHECKPOINT=""
SMOKE_ROWS=5120
if [[ $# -ge 6 ]]; then
  if [[ "$6" =~ ^[0-9]+$ ]]; then
    SMOKE_ROWS="$6"
  else
    SOURCE_CHECKPOINT="$6"
    SMOKE_ROWS=${7:-5120}
  fi
fi

[[ "$STAGE" == stage1 || "$STAGE" == stage2 ]] || { echo "invalid stage: $STAGE" >&2; exit 2; }
[[ "$MODE" == smoke || "$MODE" == resume || "$MODE" == formal ]] || { echo "invalid mode: $MODE" >&2; exit 2; }
[[ -f "$AUDIT_REPORT" && -f "$MAPPING_JSONL" ]] || { echo "audit report or mapping is missing" >&2; exit 2; }
if [[ "$MODE" != formal && "$SMOKE_ROWS" -ne 5120 ]]; then
  echo "smoke and resume contracts require exactly 5120 rows" >&2
  exit 2
fi
if [[ "$MODE" == resume || "$STAGE" == stage2 ]]; then
  [[ -n "$SOURCE_CHECKPOINT" && -f "$SOURCE_CHECKPOINT" ]] || { echo "source checkpoint is required" >&2; exit 2; }
  SOURCE_CHECKPOINT=$(readlink -f "$SOURCE_CHECKPOINT")
fi

USER_CONDA_BASE="${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
CONDA_ENV="${MELLOW_CONDA_ENV:-mellow_c8204d8}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

MELLOW_INIT_ROOT="${MELLOW_V0_INIT_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/converted/mellow_v0_5_10x2_5_mesh_epoch17_routers}"
RUN_ID="${MELLOW_RUN_ID:-${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S%N)}"
STAGE_ROOT="/dev/shm/mellow_v0_two_stage_reasonaqa_${STAGE}_${MODE}_$RUN_ID"
STAGING_REPORT="$OUTPUT_DIR/staging_report.json"
CHECKPOINT_ROOT="$OUTPUT_DIR/checkpoints"
RUNTIME_CONFIG="$OUTPUT_DIR/runtime_${STAGE}_${MODE}.yaml"
SMOKE_JSON="$OUTPUT_DIR/reasonaqa_smoke_${SMOKE_ROWS}.json"
mkdir -p "$OUTPUT_DIR"

cleanup() {
  if [[ -n "${STAGE_ROOT:-}" && -d "$STAGE_ROOT" ]]; then rm -rf -- "$STAGE_ROOT"; fi
}
trap cleanup EXIT

python "$SCRIPT_DIR/stage_reasonaqa_raw_audio.py" \
  --audit-report "$AUDIT_REPORT" \
  --mapping-jsonl "$MAPPING_JSONL" \
  --stage-root "$STAGE_ROOT" \
  --report-path "$STAGING_REPORT" \
  --workers 16 \
  --free-space-margin-gib 10 \
  --decode-samples-per-group 3 \
  --copy-batch-size 1024 \
  --progress-every 1000

STAGED_TRAIN_JSON="$STAGE_ROOT/.mellow_stage/reasonaqa_train.json"
[[ -f "$STAGED_TRAIN_JSON" && -f "$STAGE_ROOT/.mellow_stage/READY.json" ]] || { echo "staging did not produce READY.json" >&2; exit 1; }

DATA_JSON="$STAGED_TRAIN_JSON"
if [[ "$MODE" != formal ]]; then
python "$SCRIPT_DIR/prepare_reasonaqa_smoke_metadata.py" \
    --input-json "$STAGED_TRAIN_JSON" \
    --output-json "$SMOKE_JSON" \
    --rows "$SMOKE_ROWS" \
    --seed 1234
  DATA_JSON="$SMOKE_JSON"
fi

NUM_EPOCHS=2
MAX_STEPS=0
if [[ "$MODE" == smoke ]]; then MAX_STEPS=20; fi
if [[ "$MODE" == resume ]]; then MAX_STEPS=2; fi
if [[ "$MODE" == formal && "$STAGE" == stage1 ]]; then NUM_EPOCHS=5; fi
if [[ "$MODE" == formal && "$STAGE" == stage2 ]]; then NUM_EPOCHS=10; fi

CONFIG_ARGS=(
  --stage-root "$STAGE_ROOT"
  --data-json "$DATA_JSON"
  --output-config "$RUNTIME_CONFIG"
  --save-dir "$CHECKPOINT_ROOT"
  --training-stage "$STAGE"
  --mellow-init-root "$MELLOW_INIT_ROOT"
  --num-epochs "$NUM_EPOCHS"
  --max-optimizer-steps "$MAX_STEPS"
  --num-workers 4
)
if [[ "$MODE" == resume ]]; then
  python "$SCRIPT_DIR/audit_stage_checkpoint.py" "$SOURCE_CHECKPOINT" --stage "$STAGE" --expected-epochs 2 --expected-total-step 20
  CONFIG_ARGS+=(--resume-checkpoint "$SOURCE_CHECKPOINT")
elif [[ "$STAGE" == stage2 ]]; then
  python "$SCRIPT_DIR/audit_stage_checkpoint.py" "$SOURCE_CHECKPOINT" --stage stage1
  CONFIG_ARGS+=(--init-model-checkpoint "$SOURCE_CHECKPOINT")
fi

python "$SCRIPT_DIR/write_reasonaqa_runtime_config.py" "${CONFIG_ARGS[@]}"

export MELLOW_JOB_ID="mellow_v0_two_stage_${STAGE}_${MODE}_$RUN_ID"
cd "$ROUTE_ROOT"
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config "$RUNTIME_CONFIG" \
  --distributed-backend nccl \
  --save-dir "$CHECKPOINT_ROOT"

if [[ "$MODE" == smoke ]]; then
  FINAL_CHECKPOINT="$CHECKPOINT_ROOT/$MELLOW_JOB_ID/model--step-20.ckpt"
  python "$SCRIPT_DIR/audit_stage_checkpoint.py" "$FINAL_CHECKPOINT" --stage "$STAGE" --expected-epochs 2 --expected-total-step 20
elif [[ "$MODE" == resume ]]; then
  FINAL_CHECKPOINT="$CHECKPOINT_ROOT/$MELLOW_JOB_ID/model--step-22.ckpt"
  python "$SCRIPT_DIR/audit_stage_checkpoint.py" "$FINAL_CHECKPOINT" --stage "$STAGE" --expected-epochs 2 --expected-total-step 22
else
  EXPECTED_EPOCHS="$([[ "$STAGE" == stage1 ]] && echo 5 || echo 10)"
  FINAL_CHECKPOINT="$CHECKPOINT_ROOT/$MELLOW_JOB_ID/model--epo-${EXPECTED_EPOCHS}.ckpt"
  python "$SCRIPT_DIR/audit_stage_checkpoint.py" "$FINAL_CHECKPOINT" --stage "$STAGE" --expected-epochs "$EXPECTED_EPOCHS"
fi
echo "PASS: stage=$STAGE mode=$MODE checkpoint=$FINAL_CHECKPOINT" >&2
