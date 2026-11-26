#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTE_ROOT="$SCRIPT_DIR/mizar_l14_17x3_strong_reason"
DATA_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/l9_12x3_strong_reason_v1
MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M
OUTPUT_DIR=""
RESUME=""
INDEX_CACHE_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_final_l14_17x3_strong_reason_gbs128_5090/index_cache
STAGE_WORKERS=4
CHECK=0
usage() { echo "usage: $0 [--data-dir DIR] [--model-dir DIR] [--output-dir OUTER_RUN_ROOT] [--resume-checkpoint FILE] [--index-cache-root DIR] [--stage-workers N] [--check-args]" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
  if [[ $1 == --check-args ]]; then CHECK=1; shift; continue; fi
  [[ $# -ge 2 ]] || usage
  case "$1" in
    --data-dir) DATA_DIR=$2 ;;
    --model-dir) MODEL_DIR=$2 ;;
    --output-dir) OUTPUT_DIR=$2 ;;
    --resume-checkpoint) RESUME=$2 ;;
    --index-cache-root) INDEX_CACHE_ROOT=$2 ;;
    --stage-workers) STAGE_WORKERS=$2 ;;
    *) usage ;;
  esac
  shift 2
done
[[ "$STAGE_WORKERS" =~ ^[1-9][0-9]*$ ]] || usage
RUN_TAG=$(date +%Y%m%d_%H%M%S%N)
if [[ -n "$RESUME" && -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR=$(dirname "$(dirname "$(dirname "$RESUME")")")
fi
OUTPUT_DIR=${OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_final_l14_17x3_strong_reason_gbs128_5090/formal_${RUN_TAG}}
LOG_PATH="$SCRIPT_DIR/log/mizar-l14-core.${RUN_TAG}.JOB.log"
if [[ $CHECK == 1 ]]; then
  printf 'queue=pdgpu-5090 cpus=32 memory=256G gpus=8 nodes=1\ndata_dir=%s\nmodel_dir=%s\noutput_root=%s\nresume=%s\nstage_workers=%s\nlog_path=%s\n' "$DATA_DIR" "$MODEL_DIR" "$OUTPUT_DIR" "$RESUME" "$STAGE_WORKERS" "$LOG_PATH"
  exit 0
fi
CONDA_BASE=${RSMOL_CONDA_BASE:-${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[[ -f "$DATA_DIR/manifest.json" ]] || { echo "Run prepare_mizar_l9_12x3_strong_reason.sh first; missing $DATA_DIR/manifest.json" >&2; exit 2; }
mkdir -p "$SCRIPT_DIR/log" "$INDEX_CACHE_ROOT"
INDEX_PATH_FILE=$(mktemp)
trap 'rm -f "$INDEX_PATH_FILE"' EXIT
CUDA_VISIBLE_DEVICES='' python "$ROUTE_ROOT/prepare_index.py" \
  --manifest "$DATA_DIR/train.jsonl" --resolved-paths "$DATA_DIR/resolved_paths.tsv" \
  --cache-root "$INDEX_CACHE_ROOT" --path-output "$INDEX_PATH_FILE"
INDEX_DIR=$(tr -d '\r\n' < "$INDEX_PATH_FILE")
ADMISSION="$SCRIPT_DIR/log/mizar-l14-core.${RUN_TAG}.admission.json"
ADMIT_ARGS=(--data-dir "$DATA_DIR" --index-dir "$INDEX_DIR" --model-dir "$MODEL_DIR" --output-root "$OUTPUT_DIR" --report "$ADMISSION")
[[ -z "$RESUME" ]] || ADMIT_ARGS+=(--resume-checkpoint "$RESUME")
CUDA_VISIBLE_DEVICES='' python "$ROUTE_ROOT/admission.py" "${ADMIT_ARGS[@]}"
ARGS=("$DATA_DIR" "$OUTPUT_DIR" "$MODEL_DIR" "$INDEX_DIR" "$STAGE_WORKERS" "$RESUME")
printf -v CMD_ARGS '%q ' "${ARGS[@]}"
vc submit -p pdgpu-5090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 -j "mizar-l14-core-${RUN_TAG}" -d "$SCRIPT_DIR" \
  JOB=1:1 "$LOG_PATH" --cmd "bash mizar_l14_17x3_strong_reason/run.sh $CMD_ARGS"
printf 'Submitted once.\nmanifest=%s\noutput=%s/formal\nlog_path=%s\nadmission=%s\n' "$DATA_DIR/manifest.json" "$OUTPUT_DIR" "$LOG_PATH" "$ADMISSION"
