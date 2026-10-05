#!/usr/bin/env bash
set -euo pipefail
STAGE_ROOT=""
cleanup() { if [[ -n "${STAGE_ROOT:-}" && -d "$STAGE_ROOT" && "$STAGE_ROOT" == /dev/shm/mellow_adamw_cosine_reasonaqa_mcq_* ]]; then rm -rf -- "$STAGE_ROOT"; fi; }
trap cleanup EXIT
if [[ $# -ne 5 ]]; then echo "usage: $0 <mcq-manifest.json> <smoke-full-checkpoint.ckpt> <output-dir> <audit-report.json> <mapping.jsonl>" >&2; exit 2; fi
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); ROUTE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
MCQ_JSON=$(readlink -f "$1"); RESUME_CKPT=$(readlink -f "$2"); OUTPUT_DIR=$(readlink -m "$3"); AUDIT_REPORT=$(readlink -f "$4"); MAPPING_JSONL=$(readlink -f "$5")
[[ -f "$MCQ_JSON" && -f "$RESUME_CKPT" ]] || { echo "manifest or resume checkpoint missing" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"; USER_CONDA_BASE="${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"; source "$USER_CONDA_BASE/etc/profile.d/conda.sh"; conda activate mellow_c8204d8
python - "$RESUME_CKPT" <<'PY'
import sys, torch
c=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
required={"schema_version","state_dict","optimizer","optimizer_contract","scheduler","grad_scaler","grad_norm_tracker","loss_tracker","epoch_completed","total_step","num_epochs","batch_geometry","random_state_by_rank","loss_reduction"}
missing=required.difference(c) if isinstance(c,dict) else required
assert not missing, f"incomplete resume checkpoint, missing={sorted(missing)}"
assert c.get("schema_version")==2, f"unexpected schema_version={c.get('schema_version')!r}"
assert c.get("num_epochs")==2, f"unexpected num_epochs={c.get('num_epochs')!r}"
assert c.get("total_step")==20, f"resume requires step-20 checkpoint, got total_step={c.get('total_step')!r}"
assert c.get("loss_reduction")=="global_token_mean", f"wrong loss reduction={c.get('loss_reduction')!r}"
assert c.get("batch_geometry")=={"per_rank_batch_size":8,"world_size":8,"gradient_accumulation_steps":4}, f"unexpected batch geometry={c.get('batch_geometry')!r}"
contract=c["optimizer_contract"]
assert contract.get("type")=="AdamW" and tuple(contract.get("betas",()))==(0.9,0.95), f"unexpected optimizer contract={contract!r}"
s=c["scheduler"]
assert s.get("scheduler_type")=="step_cosine_warmup" and s.get("last_step")==20, f"unexpected scheduler state={s!r}"
assert abs(float(s.get("max_lr",-1))-1e-4)<1e-12 and abs(float(s.get("min_lr",-1))-1e-5)<1e-12, f"unexpected scheduler bounds={s!r}"
assert int(s.get("total_steps",0))>20 and int(s.get("warmup_steps",0))>0, f"invalid scheduler horizon={s!r}"
assert len(c["random_state_by_rank"])==8, f"expected RNG state for 8 ranks, got {len(c['random_state_by_rank'])}"
PY
RUN_ID="${MELLOW_RUN_ID:-${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S%N)_${RANDOM}}"; STAGE_ROOT="/dev/shm/mellow_adamw_cosine_reasonaqa_mcq_resume_$RUN_ID"; CHECKPOINT_ROOT="$OUTPUT_DIR/checkpoints"
python "$SCRIPT_DIR/stage_reasonaqa_mcq_raw_audio.py" --manifest-json "$MCQ_JSON" --audit-report "$AUDIT_REPORT" --mapping-jsonl "$MAPPING_JSONL" --stage-root "$STAGE_ROOT" --report-path "$OUTPUT_DIR/staging_report.json"
STAGE_ROOT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage_root"])' "$OUTPUT_DIR/staging_report.json")
python "$SCRIPT_DIR/write_reasonaqa_runtime_config.py" --stage-root "$STAGE_ROOT" --data-json "$STAGE_ROOT/.mellow_stage/reasonaqa_mcq_train.json" --output-config "$OUTPUT_DIR/runtime_mcq_resume.yaml" --save-dir "$CHECKPOINT_ROOT" --batch-size 8 --gradient-accumulation-steps 4 --num-epochs 2 --max-epochs-this-run 0 --max-optimizer-steps 2 --resume-checkpoint "$RESUME_CKPT" --max-lr 1e-4 --min-lr 1e-5 --warmup-ratio 0.05 --num-workers 4
export MELLOW_JOB_ID="mellow_adamw_cosine_reasonaqa_mcq_resume_$RUN_ID"; cd "$ROUTE_ROOT"; torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --config "$OUTPUT_DIR/runtime_mcq_resume.yaml" --distributed-backend nccl --save-dir "$CHECKPOINT_ROOT"; test -f "$CHECKPOINT_ROOT/$MELLOW_JOB_ID/model--step-22.ckpt"; rm -rf "$STAGE_ROOT"
