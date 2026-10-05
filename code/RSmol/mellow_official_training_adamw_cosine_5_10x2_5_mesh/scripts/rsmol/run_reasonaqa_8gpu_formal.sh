#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <PASS audit-report.json> <mapping.jsonl> <output-dir>" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
AUDIT_REPORT=$(readlink -f "$1")
MAPPING_JSONL=$(readlink -f "$2")
OUTPUT_DIR=$(readlink -m "$3")
MESH_TEXT_CHECKPOINT=${MELLOW_MESH_TEXT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244}

if [[ ! -f "$AUDIT_REPORT" || ! -f "$MAPPING_JSONL" ]]; then
  echo "audit report or mapping is missing" >&2
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

RUN_ID="${MELLOW_RUN_ID:-${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S%N)}"
STAGE_ROOT="/dev/shm/mellow_adamw_cosine_reasonaqa_mesh_$RUN_ID"
STAGING_REPORT="$OUTPUT_DIR/staging_report.json"
RUNTIME_CONFIG="$OUTPUT_DIR/runtime_30epochs.yaml"
CHECKPOINT_ROOT="$OUTPUT_DIR/checkpoints"

cleanup() {
  if [[ -n "${STAGE_ROOT:-}" && -d "$STAGE_ROOT" ]]; then
    rm -rf -- "$STAGE_ROOT"
  fi
}
trap cleanup EXIT

echo "[mesh-formal] staging raw audio into $STAGE_ROOT" >&2
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
  --mesh-text-checkpoint "$MESH_TEXT_CHECKPOINT" \
  --output-config "$RUNTIME_CONFIG" \
  --save-dir "$CHECKPOINT_ROOT" \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --num-epochs 30 \
  --max-epochs-this-run 0 \
  --max-optimizer-steps 0 \
  --max-lr 1e-3 \
  --min-lr 5e-5 \
  --warmup-ratio 0.05 \
  --num-workers 4

export MELLOW_JOB_ID="mellow_adamw_cosine_reasonaqa_mesh_formal_$RUN_ID"
cd "$ROUTE_ROOT"
echo "[mesh-formal] launching 30-epoch torchrun world_size=8, batch=8, accumulation=4" >&2
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config "$RUNTIME_CONFIG" \
  --distributed-backend nccl \
  --save-dir "$CHECKPOINT_ROOT"

FINAL_CHECKPOINT="$CHECKPOINT_ROOT/$MELLOW_JOB_ID/model--epo-30.ckpt"
[[ -f "$FINAL_CHECKPOINT" ]] || {
  echo "[mesh-formal] final checkpoint not found: $FINAL_CHECKPOINT" >&2
  exit 1
}
python - "$FINAL_CHECKPOINT" <<'PY'
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
required = {
    "schema_version", "state_dict", "optimizer", "optimizer_contract", "scheduler", "grad_scaler",
    "grad_norm_tracker", "loss_tracker", "epoch_completed", "total_step", "num_epochs",
    "batch_geometry", "random_state_by_rank", "loss_reduction", "route_contract", "text_model_contract",
}
missing = sorted(required.difference(checkpoint))
if checkpoint.get("schema_version") != 2 or missing:
    raise SystemExit(f"invalid formal checkpoint: schema={checkpoint.get('schema_version')} missing={missing}")
if checkpoint["epoch_completed"] != 30 or checkpoint["num_epochs"] != 30:
    raise SystemExit(f"formal checkpoint has wrong epoch horizon: epoch={checkpoint['epoch_completed']} epochs={checkpoint['num_epochs']}")
if checkpoint["loss_reduction"] != "global_token_mean":
    raise SystemExit("formal checkpoint does not declare global_token_mean")
if checkpoint["route_contract"] != "mellow_official_adamw_cosine_5_10x2_5_mesh_v1":
    raise SystemExit("formal checkpoint route contract mismatch")
if checkpoint["text_model_contract"] != "logical_30_physical_20_5_10x2_5":
    raise SystemExit("formal checkpoint text-model contract mismatch")
if checkpoint["batch_geometry"] != {
    "per_rank_batch_size": 8, "world_size": 8, "gradient_accumulation_steps": 4,
}:
    raise SystemExit(f"unexpected formal batch geometry: {checkpoint['batch_geometry']!r}")
contract = checkpoint["optimizer_contract"]
if contract.get("type") != "AdamW" or tuple(contract.get("betas", ())) != (0.9, 0.95):
    raise SystemExit(f"unexpected optimizer contract: {contract!r}")
scheduler = checkpoint["scheduler"]
if scheduler.get("scheduler_type") != "step_cosine_warmup":
    raise SystemExit(f"unexpected scheduler contract: {scheduler!r}")
if abs(float(scheduler.get("max_lr", -1.0)) - 1e-3) > 1e-12 or abs(float(scheduler.get("min_lr", -1.0)) - 5e-5) > 1e-12:
    raise SystemExit(f"unexpected formal scheduler bounds: {scheduler!r}")
if int(scheduler.get("total_steps", -1)) != 113430 or int(scheduler.get("warmup_steps", -1)) != 5672:
    raise SystemExit(f"unexpected formal scheduler horizon: {scheduler!r}")
if int(checkpoint.get("total_step", 0)) != 113430 or int(scheduler.get("last_step", 0)) != 113430:
    raise SystemExit("formal scheduler and optimizer step are inconsistent")
if len(checkpoint["random_state_by_rank"]) != 8:
    raise SystemExit("formal checkpoint does not contain RNG state for all ranks")
print(f"[mesh-formal] final checkpoint audit PASS: {path}", file=sys.stderr)
PY
echo "[mesh-formal] PASS: checkpoint=$FINAL_CHECKPOINT" >&2
