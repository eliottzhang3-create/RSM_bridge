#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 <PASS audit-report.json> <mapping.jsonl> <resume-checkpoint.ckpt> <output-dir> [smoke-rows]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
AUDIT_REPORT=$(readlink -f "$1")
MAPPING_JSONL=$(readlink -f "$2")
RESUME_CHECKPOINT=$(readlink -f "$3")
OUTPUT_DIR=$(readlink -m "$4")
SMOKE_ROWS=${5:-2048}

[[ -f "$AUDIT_REPORT" && -f "$MAPPING_JSONL" && -f "$RESUME_CHECKPOINT" ]] || {
  echo "audit, mapping, or resume checkpoint is missing" >&2
  exit 2
}
mkdir -p "$OUTPUT_DIR"

USER_CONDA_BASE="${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8

RUN_ID="${MELLOW_RUN_ID:-${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S%N)}"
STAGE_ROOT="/dev/shm/mellow_official_reasonaqa_resume_$RUN_ID"
STAGING_REPORT="$OUTPUT_DIR/staging_report.json"
SMOKE_JSON="$OUTPUT_DIR/reasonaqa_smoke_${SMOKE_ROWS}.json"
RUNTIME_CONFIG="$OUTPUT_DIR/runtime_resume_smoke.yaml"
CHECKPOINT_ROOT="$OUTPUT_DIR/checkpoints"

cleanup() {
  if [[ -n "${STAGE_ROOT:-}" && -d "$STAGE_ROOT" ]]; then
    rm -rf -- "$STAGE_ROOT"
  fi
}
trap cleanup EXIT

echo "[mellow-resume] staging raw audio into $STAGE_ROOT" >&2
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
python "$SCRIPT_DIR/prepare_reasonaqa_smoke_metadata.py" \
  --input-json "$STAGED_TRAIN_JSON" \
  --output-json "$SMOKE_JSON" \
  --rows "$SMOKE_ROWS" \
  --seed 1234

python "$SCRIPT_DIR/write_reasonaqa_runtime_config.py" \
  --stage-root "$STAGE_ROOT" \
  --data-json "$SMOKE_JSON" \
  --output-config "$RUNTIME_CONFIG" \
  --save-dir "$CHECKPOINT_ROOT" \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --num-epochs 2 \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --num-workers 4

export MELLOW_JOB_ID="mellow_official_reasonaqa_resume_$RUN_ID"
cd "$ROUTE_ROOT"
echo "[mellow-resume] launching torchrun world_size=8 from epoch checkpoint" >&2
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config "$RUNTIME_CONFIG" \
  --distributed-backend nccl \
  --save-dir "$CHECKPOINT_ROOT"

echo "[mellow-resume] PASS: resume training exited successfully" >&2
echo "[mellow-resume] staging_report=$STAGING_REPORT" >&2
echo "[mellow-resume] runtime_config=$RUNTIME_CONFIG" >&2
echo "[mellow-resume] checkpoints=$CHECKPOINT_ROOT/$MELLOW_JOB_ID" >&2
find "$CHECKPOINT_ROOT/$MELLOW_JOB_ID" -maxdepth 1 -type f -name '*.ckpt' -print | sort >&2

CHECKPOINT_PATH=$(find "$CHECKPOINT_ROOT/$MELLOW_JOB_ID" -maxdepth 1 -type f -name '*.ckpt' -print -quit)
if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "[mellow-resume] FAIL: no checkpoint was published" >&2
  exit 1
fi
python - "$CHECKPOINT_PATH" "$RESUME_CHECKPOINT" <<'PY'
import sys
import torch

new_path, source_path = sys.argv[1:]
source = torch.load(source_path, map_location="cpu", weights_only=False)
new = torch.load(new_path, map_location="cpu", weights_only=False)
if source.get("schema_version") != 2 or new.get("schema_version") != 2:
    raise SystemExit("resume audit requires schema_version=2 checkpoints")
required = {
    "state_dict", "optimizer", "scheduler", "grad_scaler", "grad_norm_tracker", "loss_tracker",
    "epoch_completed", "total_step", "num_epochs", "batch_geometry",
    "random_state_by_rank",
}
for label, checkpoint in (("source", source), ("new", new)):
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise SystemExit(f"{label} checkpoint is missing fields: {missing}")
    if checkpoint["batch_geometry"] != {
        "per_rank_batch_size": 8,
        "world_size": 8,
        "gradient_accumulation_steps": 4,
    }:
        raise SystemExit(f"{label} checkpoint has unexpected batch geometry")
    if len(checkpoint["random_state_by_rank"]) != 8:
        raise SystemExit(f"{label} checkpoint does not contain RNG state for all 8 ranks")
if source.get("epoch_completed") != 1 or new.get("epoch_completed") != 2:
    raise SystemExit(
        f"resume did not advance exactly one epoch: source={source.get('epoch_completed')} "
        f"new={new.get('epoch_completed')}"
    )
if source.get("num_epochs") != 2 or new.get("num_epochs") != 2:
    raise SystemExit("resume checkpoints do not preserve the 2-epoch smoke horizon")
if new.get("total_step", 0) <= source.get("total_step", 0):
    raise SystemExit("resume total_step did not advance")
print(f"[mellow-resume] checkpoint continuity audit PASS: {new_path}", file=sys.stderr)
PY
