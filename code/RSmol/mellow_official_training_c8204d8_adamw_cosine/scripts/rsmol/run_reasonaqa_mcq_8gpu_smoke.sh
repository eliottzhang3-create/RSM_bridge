#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then echo "usage: $0 <mcq-manifest.json> <init-model.ckpt> <output-dir>" >&2; exit 2; fi
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
MCQ_JSON=$(readlink -f "$1"); INIT_CKPT=$(readlink -f "$2"); OUTPUT_DIR=$(readlink -m "$3")
[[ -f "$MCQ_JSON" && -f "$INIT_CKPT" ]] || { echo "manifest or initialization checkpoint missing" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
USER_CONDA_BASE="${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"; conda activate mellow_c8204d8
RUN_ID="${MELLOW_RUN_ID:-${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S%N)}"
STAGE_ROOT="/dev/shm/mellow_adamw_cosine_reasonaqa_mcq_$RUN_ID"; CHECKPOINT_ROOT="$OUTPUT_DIR/checkpoints"
 mkdir -p "$STAGE_ROOT/.mellow_stage"
 python "$SCRIPT_DIR/stage_reasonaqa_mcq_raw_audio.py" --manifest-json "$MCQ_JSON" --stage-root "$STAGE_ROOT" --report-path "$OUTPUT_DIR/staging_report.json"
python "$SCRIPT_DIR/write_reasonaqa_runtime_config.py" --stage-root "$STAGE_ROOT" --data-json "$STAGE_ROOT/.mellow_stage/reasonaqa_mcq_train.json" --output-config "$OUTPUT_DIR/runtime_mcq_smoke.yaml" --save-dir "$CHECKPOINT_ROOT" --batch-size 8 --gradient-accumulation-steps 4 --num-epochs 1 --max-epochs-this-run 1 --init-model-checkpoint "$INIT_CKPT" --max-lr 1e-4 --min-lr 1e-5 --warmup-ratio 0.05 --num-workers 4
export MELLOW_JOB_ID="mellow_adamw_cosine_reasonaqa_mcq_smoke_$RUN_ID"
cd "$ROUTE_ROOT"; torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --config "$OUTPUT_DIR/runtime_mcq_smoke.yaml" --distributed-backend nccl --save-dir "$CHECKPOINT_ROOT"
python - "$CHECKPOINT_ROOT/$MELLOW_JOB_ID/model--epo-1.ckpt" <<'PY'
import sys, torch
p=sys.argv[1]; c=torch.load(p,map_location="cpu",weights_only=False)
assert c.get("schema_version")==2 and c.get("num_epochs")==1 and c.get("epoch_completed")==1
assert c["optimizer_contract"]["type"]=="AdamW" and tuple(c["optimizer_contract"]["betas"])==(0.9,0.95)
s=c["scheduler"]; assert s["max_lr"]==1e-4 and s["min_lr"]==1e-5 and s["scheduler_type"]=="step_cosine_warmup"
assert len(c["random_state_by_rank"])==8 and c["total_step"]>0
print("MCQ smoke checkpoint audit PASS")
PY
rm -rf "$STAGE_ROOT"
