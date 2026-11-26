#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTE_ROOT="$SCRIPT_DIR/mizar_l14_17x3_strong_reason"
DATA_DIR=/mnt/cloudstorfs/sjtu_home/jinwei.zhang/data/Mizar/l9_12x4_audiomcq_only_v1
SOURCE=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
RESOLVED_PATHS=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M
OUTPUT_DIR=""
RESUME=""
INDEX_CACHE_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_final_l14_17x4_audiomcq_gbs128_3epochs_3090/index_cache
STAGE_WORKERS=4
CHECK=0
PREPARE_ONLY=0
usage() { echo "usage: $0 [--source HISTORICAL_HYBRID] [--resolved-paths TSV] [--data-dir DIR] [--model-dir DIR] [--output-dir OUTER_RUN_ROOT] [--resume-checkpoint FILE] [--index-cache-root DIR] [--stage-workers N] [--check-args] [--prepare-only]" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
  if [[ $1 == --check-args ]]; then CHECK=1; shift; continue; fi
  if [[ $1 == --prepare-only ]]; then PREPARE_ONLY=1; shift; continue; fi
  [[ $# -ge 2 ]] || usage
  case "$1" in
    --data-dir) DATA_DIR=$2 ;;
    --source) SOURCE=$2 ;;
    --resolved-paths) RESOLVED_PATHS=$2 ;;
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
OUTPUT_DIR=${OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_final_l14_17x4_audiomcq_gbs128_3epochs_3090/formal_${RUN_TAG}}
LOG_PATH="$SCRIPT_DIR/log/mizar-l14-audiomcq-x4.${RUN_TAG}.JOB.log"
if [[ $CHECK == 1 ]]; then
  printf 'queue=pdgpu-3090 cpus=64 memory=256G gpus=8 nodes=1\ndata_dir=%s\nmodel_dir=%s\noutput_root=%s\nresume=%s\nstage_workers=%s\nlog_path=%s\n' "$DATA_DIR" "$MODEL_DIR" "$OUTPUT_DIR" "$RESUME" "$STAGE_WORKERS" "$LOG_PATH"
  printf 'source=%s\nresolved_paths=%s\nrecipe=AudioMCQ720524 ReasonAQA0 L14-17x4 no_injection GBS128 epochs3 max_lr2e-5 min_lr2e-6 warmup3pct keep4\n' "$SOURCE" "$RESOLVED_PATHS"
  exit 0
fi
CONDA_BASE=${RSMOL_CONDA_BASE:-${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES='' python -u "$ROUTE_ROOT/prepare_manifest.py" --source "$SOURCE" --resolved-paths "$RESOLVED_PATHS" --output-dir "$DATA_DIR"
if [[ $PREPARE_ONLY == 1 ]]; then printf 'manifest=%s/manifest.json\n' "$DATA_DIR"; exit 0; fi
mkdir -p "$SCRIPT_DIR/log" "$INDEX_CACHE_ROOT"
INDEX_PATH_FILE=$(mktemp)
trap 'rm -f "$INDEX_PATH_FILE"' EXIT
CUDA_VISIBLE_DEVICES='' python "$ROUTE_ROOT/prepare_index.py" \
  --manifest "$DATA_DIR/train.jsonl" --resolved-paths "$DATA_DIR/resolved_paths.tsv" \
  --cache-root "$INDEX_CACHE_ROOT" --path-output "$INDEX_PATH_FILE"
INDEX_DIR=$(tr -d '\r\n' < "$INDEX_PATH_FILE")
ADMISSION="$SCRIPT_DIR/log/mizar-l14-audiomcq-x4.${RUN_TAG}.admission.json"
ADMIT_ARGS=(--data-dir "$DATA_DIR" --index-dir "$INDEX_DIR" --model-dir "$MODEL_DIR" --output-root "$OUTPUT_DIR" --report "$ADMISSION")
[[ -z "$RESUME" ]] || ADMIT_ARGS+=(--resume-checkpoint "$RESUME")
CUDA_VISIBLE_DEVICES='' python "$ROUTE_ROOT/admission.py" "${ADMIT_ARGS[@]}"
ARGS=("$DATA_DIR" "$OUTPUT_DIR" "$MODEL_DIR" "$INDEX_DIR" "$STAGE_WORKERS" "$RESUME")
printf -v CMD_ARGS '%q ' "${ARGS[@]}"
vc submit -p pdgpu-3090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 64 -m 256G -g 8 -n 1 -j "mizar-l14-audiomcq-x4-${RUN_TAG}" -d "$SCRIPT_DIR" \
  JOB=1:1 "$LOG_PATH" --cmd "bash mizar_l14_17x3_strong_reason/run.sh $CMD_ARGS"
printf 'Submitted once.\nmanifest=%s\noutput=%s/formal\nlog_path=%s\nadmission=%s\n' "$DATA_DIR/manifest.json" "$OUTPUT_DIR" "$LOG_PATH" "$ADMISSION"
