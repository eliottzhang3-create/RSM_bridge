#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 <PASS audit-report.json> <mapping.jsonl> <output-dir> [num-epochs] [resume-checkpoint]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
AUDIT_REPORT=$(readlink -f "$1")
MAPPING_JSONL=$(readlink -f "$2")
OUTPUT_DIR=$(readlink -m "$3")
NUM_EPOCHS=${4:-5}
RESUME_CHECKPOINT=${5:-}
RESUME_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  RESUME_CHECKPOINT=$(readlink -f "$RESUME_CHECKPOINT")
  [[ -f "$RESUME_CHECKPOINT" ]] || { echo "resume checkpoint not found: $RESUME_CHECKPOINT" >&2; exit 2; }
  RESUME_ARGS=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

if [[ ! -f "$AUDIT_REPORT" || ! -f "$MAPPING_JSONL" ]]; then
  echo "audit report or mapping is missing" >&2
  exit 2
fi
if [[ "$NUM_EPOCHS" -ne 5 ]]; then
  echo "formal route requires exactly 5 epochs" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

USER_CONDA_BASE="${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8

RUN_ID="${MELLOW_RUN_ID:-${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S%N)}"
STAGE_ROOT="/dev/shm/mellow_adamw_cosine_reasonaqa_gbs32_$RUN_ID"
STAGING_REPORT="$OUTPUT_DIR/staging_report.json"
RUNTIME_CONFIG="$OUTPUT_DIR/runtime_5epochs.yaml"
CHECKPOINT_ROOT="$OUTPUT_DIR/checkpoints"

cleanup() {
  if [[ -n "${STAGE_ROOT:-}" && -d "$STAGE_ROOT" ]]; then
    rm -rf -- "$STAGE_ROOT"
  fi
}
trap cleanup EXIT

echo "[mellow-formal] staging raw audio into $STAGE_ROOT" >&2
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
if [[ ! -f "$STAGED_TRAIN_JSON" || ! -f "$STAGE_ROOT/.mellow_stage/READY.json" ]]; then
  echo "staging did not produce READY.json and staged train metadata" >&2
  exit 1
fi

python "$SCRIPT_DIR/write_reasonaqa_runtime_config.py" \
  --stage-root "$STAGE_ROOT" \
  --data-json "$STAGED_TRAIN_JSON" \
  --output-config "$RUNTIME_CONFIG" \
  --save-dir "$CHECKPOINT_ROOT" \
  --batch-size 4 \
  --gradient-accumulation-steps 1 \
  --num-epochs "$NUM_EPOCHS" \
  --max-epochs-this-run 0 \
  --num-workers 4 \
  "${RESUME_ARGS[@]}"

export MELLOW_JOB_ID="mellow_adamw_cosine_reasonaqa_gbs32_formal_$RUN_ID"
cd "$ROUTE_ROOT"
echo "[mellow-formal] launching 5-epoch torchrun world_size=8, batch=4, accumulation=1, global_batch=32" >&2
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config "$RUNTIME_CONFIG" \
  --distributed-backend nccl \
  --save-dir "$CHECKPOINT_ROOT"

echo "[mellow-formal] PASS: training exited successfully" >&2
echo "[mellow-formal] staging_report=$STAGING_REPORT" >&2
echo "[mellow-formal] runtime_config=$RUNTIME_CONFIG" >&2
echo "[mellow-formal] checkpoints=$CHECKPOINT_ROOT/$MELLOW_JOB_ID" >&2
find "$CHECKPOINT_ROOT/$MELLOW_JOB_ID" -maxdepth 1 -type f -name '*.ckpt' -print | sort >&2

FINAL_CHECKPOINT="$CHECKPOINT_ROOT/$MELLOW_JOB_ID/model--epo-${NUM_EPOCHS}.ckpt"
[[ -f "$FINAL_CHECKPOINT" ]] || {
  echo "[mellow-formal] final checkpoint not found: $FINAL_CHECKPOINT" >&2
  exit 1
}
python - "$FINAL_CHECKPOINT" "$NUM_EPOCHS" <<'PY'
import sys
import torch

checkpoint_path, expected_epochs = sys.argv[1], int(sys.argv[2])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
required = {
    "schema_version", "state_dict", "optimizer", "optimizer_contract", "scheduler", "grad_scaler",
    "grad_norm_tracker", "loss_tracker", "epoch_completed", "total_step",
    "num_epochs", "batch_geometry", "random_state_by_rank",
}
missing = sorted(required.difference(checkpoint))
if missing:
    raise SystemExit(f"missing V2 fields: {missing}")
if checkpoint["schema_version"] != 2:
    raise SystemExit(f"unexpected checkpoint schema: {checkpoint['schema_version']!r}")
if int(checkpoint["epoch_completed"]) != expected_epochs or int(checkpoint["num_epochs"]) != expected_epochs:
    raise SystemExit("final checkpoint does not match the requested epoch horizon")
if checkpoint["batch_geometry"] != {
    "per_rank_batch_size": 4,
    "world_size": 8,
    "gradient_accumulation_steps": 1,
}:
    raise SystemExit(f"unexpected batch geometry: {checkpoint['batch_geometry']!r}")
contract = checkpoint["optimizer_contract"]
if contract.get("type") != "AdamW" or tuple(contract.get("betas", ())) != (0.9, 0.95):
    raise SystemExit(f"unexpected optimizer contract: {contract!r}")
scheduler = checkpoint["scheduler"]
if scheduler.get("scheduler_type") != "step_cosine_warmup":
    raise SystemExit(f"unexpected scheduler contract: {scheduler!r}")
if abs(float(scheduler.get("max_lr", -1.0)) - 1e-3) > 1e-12 or abs(float(scheduler.get("min_lr", -1.0)) - 5e-5) > 1e-12:
    raise SystemExit(f"unexpected scheduler learning-rate bounds: {scheduler!r}")
if not (0 < int(scheduler.get("warmup_steps", 0)) < int(scheduler.get("total_steps", 0))):
    raise SystemExit(f"invalid scheduler step contract: {scheduler!r}")
if len(checkpoint["random_state_by_rank"]) != 8 or int(checkpoint["total_step"]) <= 0:
    raise SystemExit("final checkpoint is missing complete distributed resume state")
print("[mellow-formal] V2 final checkpoint audit: PASS")
PY
