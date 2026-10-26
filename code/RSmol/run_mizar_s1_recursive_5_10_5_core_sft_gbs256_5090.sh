#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 [--output-dir DIR] [--init-checkpoint PATH] [--resume-checkpoint PATH] [--resolved-paths TSV] [--text-model-dir DIR] [--htsat-root DIR] [--index-cache-root DIR] [--stage-workers N]" >&2
  exit 2
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTE=mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_2epochs_recursive_5_10_5_core_sft
RESOLVED_PATHS=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
TEXT_MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244
HTSAT_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT
MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
INIT_CHECKPOINT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_recursive_5_10_5_5090/formal_20261025_160348996763174/formal/checkpoints/model--step-17313.ckpt
INDEX_CACHE_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_recursive_5_10_5_core_sft_gbs256_5090/index_cache
OUTPUT_DIR=""
RESUME_CHECKPOINT=""
STAGE_WORKERS=4
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || usage
  case "$1" in
    --output-dir) OUTPUT_DIR=$2 ;;
    --init-checkpoint) INIT_CHECKPOINT=$2 ;;
    --resume-checkpoint) RESUME_CHECKPOINT=$2 ;;
    --resolved-paths) RESOLVED_PATHS=$2 ;;
    --text-model-dir) TEXT_MODEL_DIR=$2 ;;
    --htsat-root) HTSAT_ROOT=$2 ;;
    --index-cache-root) INDEX_CACHE_ROOT=$2 ;;
    --stage-workers) STAGE_WORKERS=$2 ;;
    *) usage ;;
  esac
  shift 2
done
[[ "$STAGE_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "stage workers must be positive" >&2; exit 2; }
[[ -z "$RESUME_CHECKPOINT" || -f "$RESUME_CHECKPOINT" ]] || { echo "resume checkpoint missing" >&2; exit 2; }
[[ -f "$INIT_CHECKPOINT" && -f "$MANIFEST" && -f "$RESOLVED_PATHS" ]] || { echo "init checkpoint, manifest or resolved paths missing" >&2; exit 2; }
[[ -f "$TEXT_MODEL_DIR/config.json" && -f "$HTSAT_ROOT/HTSAT_AudioSet_Saved_1.ckpt" ]] || { echo "text config/tokenizer or HTSAT asset missing" >&2; exit 2; }
RUN_TAG=$(date +%Y%m%d_%H%M%S%N)
OUTPUT_DIR=${OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_recursive_5_10_5_core_sft_gbs256_5090/formal_${RUN_TAG}}
[[ ! -e "$OUTPUT_DIR" || -d "$OUTPUT_DIR" ]] || { echo "output is not a directory" >&2; exit 2; }
if [[ -z "$RESUME_CHECKPOINT" && -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "fresh run requires an empty output directory: $OUTPUT_DIR" >&2; exit 2
fi
mkdir -p "$SCRIPT_DIR/log"
INDEX_PATH_FILE=$(mktemp)
cleanup_index_path() { rm -f "$INDEX_PATH_FILE"; }
trap cleanup_index_path EXIT
python "$SCRIPT_DIR/$ROUTE/prepare_index.py" --manifest "$MANIFEST" \
  --resolved-paths "$RESOLVED_PATHS" --cache-root "$INDEX_CACHE_ROOT" --path-output "$INDEX_PATH_FILE"
INDEX_DIR=$(tr -d '\r\n' < "$INDEX_PATH_FILE")
[[ -d "$INDEX_DIR" ]] || { echo "CPU index missing: $INDEX_DIR" >&2; exit 2; }
ARGS=(formal "$RESOLVED_PATHS" "$OUTPUT_DIR" "$RESUME_CHECKPOINT" "$TEXT_MODEL_DIR" "$HTSAT_ROOT" "$INDEX_DIR" "$STAGE_WORKERS" "$INIT_CHECKPOINT")
printf -v CMD_ARGS '%q ' "${ARGS[@]}"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "mizar-s1-recursive-core-sft-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mizar_s1_recursive_core_sft.${RUN_TAG}.JOB.log" \
  --cmd "bash $ROUTE/run.sh $CMD_ARGS"
echo "Submitted formal non-MeSH Core SFT baseline: 8 GPUs, global batch 256, two epochs / 11542 updates; max_lr=5e-5 min_lr=5e-6; output=$OUTPUT_DIR"
