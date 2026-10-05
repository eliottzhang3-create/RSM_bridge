#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 <PASS audit-report.json> <mapping.jsonl> <step-20 checkpoint.ckpt> <output-dir> [smoke-rows=5120]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
AUDIT_REPORT=$(readlink -f "$1")
MAPPING_JSONL=$(readlink -f "$2")
RESUME_CHECKPOINT=$(readlink -f "$3")
OUTPUT_DIR=$(readlink -m "$4")
SMOKE_ROWS=${5:-5120}
MESH_TEXT_CHECKPOINT=${MELLOW_MESH_TEXT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244}

if [[ "$SMOKE_ROWS" -ne 5120 ]]; then
  echo "the 20-to-22-step resume contract requires exactly 5120 rows" >&2
  exit 2
fi
if [[ ! -f "$AUDIT_REPORT" || ! -f "$MAPPING_JSONL" || ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "audit, mapping, or resume checkpoint is missing" >&2
  exit 2
fi
if [[ ! -d "$MESH_TEXT_CHECKPOINT" ]]; then
  echo "mesh text checkpoint directory is missing: $MESH_TEXT_CHECKPOINT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

USER_CONDA_BASE="${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8

python - "$RESUME_CHECKPOINT" <<'PY'
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
required = {
    "schema_version", "state_dict", "optimizer", "optimizer_contract", "scheduler", "grad_scaler",
    "grad_norm_tracker", "loss_tracker", "epoch_completed", "total_step", "num_epochs",
    "batch_geometry", "random_state_by_rank", "loss_reduction", "route_contract", "text_model_contract",
}
if not isinstance(checkpoint, dict):
    raise SystemExit("resume checkpoint is not a dictionary")
missing = sorted(required.difference(checkpoint))
if checkpoint.get("schema_version") != 2 or missing:
    raise SystemExit(f"resume requires schema_version=2 full checkpoint; missing={missing}")
if checkpoint["route_contract"] != "mellow_official_adamw_cosine_5_10x2_5_mesh_v1" or checkpoint["text_model_contract"] != "logical_30_physical_20_5_10x2_5":
    raise SystemExit("resume checkpoint belongs to another route or text-model contract")
if checkpoint["loss_reduction"] != "global_token_mean":
    raise SystemExit("resume checkpoint does not use global_token_mean")
if checkpoint["batch_geometry"] != {
    "per_rank_batch_size": 8, "world_size": 8, "gradient_accumulation_steps": 4,
}:
    raise SystemExit(f"resume checkpoint batch geometry mismatch: {checkpoint['batch_geometry']!r}")
if checkpoint.get("epoch_completed") != 1 or checkpoint.get("total_step") != 20 or checkpoint.get("num_epochs") != 2:
    raise SystemExit(
        "resume source must be the audited smoke checkpoint at epoch=1, step=20, horizon=2: "
        f"epoch={checkpoint.get('epoch_completed')} step={checkpoint.get('total_step')} epochs={checkpoint.get('num_epochs')}"
    )
contract = checkpoint["optimizer_contract"]
if contract.get("type") != "AdamW" or tuple(contract.get("betas", ())) != (0.9, 0.95):
    raise SystemExit(f"unexpected optimizer contract: {contract!r}")
scheduler = checkpoint["scheduler"]
if scheduler.get("scheduler_type") != "step_cosine_warmup" or int(scheduler.get("total_steps", -1)) != 40 or int(scheduler.get("last_step", -1)) != 20:
    raise SystemExit(f"unexpected source scheduler state: {scheduler!r}")
if abs(float(scheduler.get("max_lr", -1.0)) - 1e-3) > 1e-12 or abs(float(scheduler.get("min_lr", -1.0)) - 5e-5) > 1e-12:
    raise SystemExit(f"unexpected source scheduler bounds: {scheduler!r}")
if len(checkpoint["random_state_by_rank"]) != 8:
    raise SystemExit("resume checkpoint does not contain RNG state for all ranks")
print(f"[mesh-resume] source checkpoint audit PASS: {path}", file=sys.stderr)
PY

RUN_ID="${MELLOW_RUN_ID:-${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S%N)}"
STAGE_ROOT="/dev/shm/mellow_adamw_cosine_reasonaqa_mesh_resume_$RUN_ID"
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

echo "[mesh-resume] staging raw audio into $STAGE_ROOT" >&2
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
  --mesh-text-checkpoint "$MESH_TEXT_CHECKPOINT" \
  --output-config "$RUNTIME_CONFIG" \
  --save-dir "$CHECKPOINT_ROOT" \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --num-epochs 2 \
  --max-epochs-this-run 0 \
  --max-optimizer-steps 2 \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --max-lr 1e-3 \
  --min-lr 5e-5 \
  --warmup-ratio 0.05 \
  --num-workers 4

export MELLOW_JOB_ID="mellow_adamw_cosine_reasonaqa_mesh_resume_$RUN_ID"
cd "$ROUTE_ROOT"
echo "[mesh-resume] launching torchrun for exactly two additional optimizer steps" >&2
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config "$RUNTIME_CONFIG" \
  --distributed-backend nccl \
  --save-dir "$CHECKPOINT_ROOT"

CHECKPOINT_PATH="$CHECKPOINT_ROOT/$MELLOW_JOB_ID/model--step-22.ckpt"
[[ -f "$CHECKPOINT_PATH" ]] || {
  echo "[mesh-resume] expected checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
}
python - "$CHECKPOINT_PATH" "$RESUME_CHECKPOINT" <<'PY'
import sys
import torch

new_path, source_path = sys.argv[1:]
source = torch.load(source_path, map_location="cpu", weights_only=False)
new = torch.load(new_path, map_location="cpu", weights_only=False)
for label, checkpoint in (("source", source), ("new", new)):
    if checkpoint.get("schema_version") != 2:
        raise SystemExit(f"{label} checkpoint is not schema v2")
    if checkpoint.get("loss_reduction") != "global_token_mean":
        raise SystemExit(f"{label} checkpoint has wrong loss reduction")
    if checkpoint.get("route_contract") != "mellow_official_adamw_cosine_5_10x2_5_mesh_v1":
        raise SystemExit(f"{label} checkpoint has wrong route contract")
    if checkpoint.get("text_model_contract") != "logical_30_physical_20_5_10x2_5":
        raise SystemExit(f"{label} checkpoint has wrong text-model contract")
    if checkpoint.get("batch_geometry") != {
        "per_rank_batch_size": 8, "world_size": 8, "gradient_accumulation_steps": 4,
    }:
        raise SystemExit(f"{label} checkpoint has wrong batch geometry")
    if len(checkpoint.get("random_state_by_rank", [])) != 8:
        raise SystemExit(f"{label} checkpoint does not contain RNG state for all ranks")
    scheduler = checkpoint.get("scheduler") or {}
    if int(scheduler.get("total_steps", -1)) != 40 or int(scheduler.get("warmup_steps", -1)) != 2:
        raise SystemExit(f"{label} checkpoint has wrong scheduler horizon")
    if abs(float(scheduler.get("max_lr", -1.0)) - 1e-3) > 1e-12 or abs(float(scheduler.get("min_lr", -1.0)) - 5e-5) > 1e-12:
        raise SystemExit(f"{label} checkpoint has wrong scheduler bounds")
if source.get("total_step") != 20 or new.get("total_step") != 22:
    raise SystemExit(f"resume did not advance 20 -> 22: source={source.get('total_step')} new={new.get('total_step')}")
if source.get("epoch_completed") != 1 or new.get("epoch_completed") != 1:
    raise SystemExit(f"resume changed the completed epoch unexpectedly: source={source.get('epoch_completed')} new={new.get('epoch_completed')}")
if source.get("num_epochs") != 2 or new.get("num_epochs") != 2:
    raise SystemExit("resume did not preserve the 2-epoch smoke horizon")
if int((new.get("scheduler") or {}).get("last_step", -1)) != 22:
    raise SystemExit("resume scheduler did not advance to step 22")
print(f"[mesh-resume] checkpoint continuity audit PASS: {new_path}", file=sys.stderr)
PY
echo "[mesh-resume] PASS: checkpoint=$CHECKPOINT_PATH" >&2
