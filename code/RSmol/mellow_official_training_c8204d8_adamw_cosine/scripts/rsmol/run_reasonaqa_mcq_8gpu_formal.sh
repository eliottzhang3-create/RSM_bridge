#!/usr/bin/env bash
set -euo pipefail
STAGE_ROOT=""
cleanup() { if [[ -n "${STAGE_ROOT:-}" && -d "$STAGE_ROOT" && "$STAGE_ROOT" == /dev/shm/mellow_adamw_cosine_reasonaqa_mcq_* ]]; then rm -rf -- "$STAGE_ROOT"; fi; }
trap cleanup EXIT
if [[ $# -ne 5 ]]; then echo "usage: $0 <mcq-manifest.json> <init-model.ckpt> <output-dir> <audit-report.json> <mapping.jsonl>" >&2; exit 2; fi
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); ROUTE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
MCQ_JSON=$(readlink -f "$1"); INIT_CKPT=$(readlink -f "$2"); OUTPUT_DIR=$(readlink -m "$3"); AUDIT_REPORT=$(readlink -f "$4"); MAPPING_JSONL=$(readlink -f "$5"); [[ -f "$MCQ_JSON" && -f "$INIT_CKPT" ]] || exit 2; mkdir -p "$OUTPUT_DIR"
USER_CONDA_BASE="${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"; source "$USER_CONDA_BASE/etc/profile.d/conda.sh"; conda activate mellow_c8204d8
RUN_ID="${MELLOW_RUN_ID:-${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S%N)_${RANDOM}}"; STAGE_ROOT="/dev/shm/mellow_adamw_cosine_reasonaqa_mcq_formal_$RUN_ID"; CHECKPOINT_ROOT="$OUTPUT_DIR/checkpoints"; python "$SCRIPT_DIR/stage_reasonaqa_mcq_raw_audio.py" --manifest-json "$MCQ_JSON" --audit-report "$AUDIT_REPORT" --mapping-jsonl "$MAPPING_JSONL" --stage-root "$STAGE_ROOT" --report-path "$OUTPUT_DIR/staging_report.json"; STAGE_ROOT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage_root"])' "$OUTPUT_DIR/staging_report.json")
python "$SCRIPT_DIR/write_reasonaqa_runtime_config.py" --stage-root "$STAGE_ROOT" --data-json "$STAGE_ROOT/.mellow_stage/reasonaqa_mcq_train.json" --output-config "$OUTPUT_DIR/runtime_mcq_5epochs.yaml" --save-dir "$CHECKPOINT_ROOT" --batch-size 8 --gradient-accumulation-steps 4 --num-epochs 5 --init-model-checkpoint "$INIT_CKPT" --max-lr 1e-4 --min-lr 1e-5 --warmup-ratio 0.05 --num-workers 4
export MELLOW_JOB_ID="mellow_adamw_cosine_reasonaqa_mcq_formal_$RUN_ID"; cd "$ROUTE_ROOT"; torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --config "$OUTPUT_DIR/runtime_mcq_5epochs.yaml" --distributed-backend nccl --save-dir "$CHECKPOINT_ROOT"; test -f "$CHECKPOINT_ROOT/$MELLOW_JOB_ID/model--epo-5.ckpt"; rm -rf "$STAGE_ROOT"
