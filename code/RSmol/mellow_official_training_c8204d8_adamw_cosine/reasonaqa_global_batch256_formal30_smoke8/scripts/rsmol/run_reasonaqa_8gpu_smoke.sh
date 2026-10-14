#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then echo "usage: $0 <audit> <mapping> <output>" >&2; exit 2; fi
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROUTE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
FORMAL_ROOT=$(cd "$ROUTE_ROOT/../reasonaqa_global_batch256_formal_30epochs" && pwd)
BASE_ROOT=$(cd "$ROUTE_ROOT/../reasonaqa_global_batch256_lr_ablation_5epochs" && pwd)
AUDIT_REPORT=$(readlink -f "$1")
MAPPING_JSONL=$(readlink -f "$2")
OUTPUT_DIR=$(readlink -m "$3")
[[ -f "$AUDIT_REPORT" && -f "$MAPPING_JSONL" ]] || { echo "audit or mapping missing" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" || ( -d "$OUTPUT_DIR" && -z "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ) ]] || { echo "output must be empty" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
USER_CONDA_BASE="/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8
RUN_ID="$(date +%Y%m%d_%H%M%S%N)"
STAGE_ROOT="/dev/shm/mellow_adamw_cosine_reasonaqa_gbs256_lr_smoke8_$RUN_ID"
STAGING_REPORT="$OUTPUT_DIR/staging_report.json"
RUNTIME_CONFIG="$OUTPUT_DIR/runtime_smoke8.yaml"
CHECKPOINT_ROOT="$OUTPUT_DIR/checkpoints"
cleanup() { if [[ -d "$STAGE_ROOT" ]]; then rm -rf -- "$STAGE_ROOT"; fi; }
trap cleanup EXIT
python "$BASE_ROOT/scripts/rsmol/stage_reasonaqa_raw_audio.py" \
  --audit-report "$AUDIT_REPORT" --mapping-jsonl "$MAPPING_JSONL" \
  --stage-root "$STAGE_ROOT" --report-path "$STAGING_REPORT" \
  --workers 16 --free-space-margin-gib 10 --decode-samples-per-group 3 \
  --copy-batch-size 1024 --progress-every 1000
STAGED_TRAIN_JSON="$STAGE_ROOT/.mellow_stage/reasonaqa_train.json"
[[ -f "$STAGED_TRAIN_JSON" && -f "$STAGE_ROOT/.mellow_stage/READY.json" ]] || { echo "stage not READY" >&2; exit 1; }
python "$SCRIPT_DIR/write_reasonaqa_runtime_config.py" \
  --stage-root "$STAGE_ROOT" --data-json "$STAGED_TRAIN_JSON" \
  --output-config "$RUNTIME_CONFIG" --save-dir "$CHECKPOINT_ROOT" \
  --batch-size 8 --gradient-accumulation-steps 4 --num-epochs 30 \
  --max-lr 2e-4 --min-lr 1e-5 --max-epochs-this-run 0 \
  --max-optimizer-steps 8 --num-workers 4
export MELLOW_JOB_ID="mellow_adamw_cosine_reasonaqa_gbs256_smoke8_$RUN_ID"
export PYTHONPATH="$ROUTE_ROOT:$FORMAL_ROOT:$BASE_ROOT:${PYTHONPATH:-}"
cd "$ROUTE_ROOT"
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config "$RUNTIME_CONFIG" --distributed-backend nccl --save-dir "$CHECKPOINT_ROOT" \
  --reraise-exceptions
CHECKPOINT="$CHECKPOINT_ROOT/$MELLOW_JOB_ID/model--step-8.ckpt"
python "$SCRIPT_DIR/audit_smoke8.py" --output-dir "$OUTPUT_DIR" --checkpoint "$CHECKPOINT"
echo "[mellow-smoke8] PASS: $OUTPUT_DIR" >&2
