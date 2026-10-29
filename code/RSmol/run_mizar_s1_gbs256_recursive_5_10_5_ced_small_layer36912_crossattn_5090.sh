#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 [--output-dir DIR] [--resume-checkpoint PATH] [--resolved-paths TSV] [--text-model-dir DIR] [--ced-model-dir DIR] [--index-cache-root DIR] [--stage-workers N]" >&2
  exit 2
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESOLVED_PATHS=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
TEXT_MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244
CED_MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small
MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
OUTPUT_DIR=""
RESUME_CHECKPOINT=""
INDEX_CACHE_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer36912_crossattn_5090/index_cache
STAGE_WORKERS=4
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || usage
  case "$1" in
    --output-dir) OUTPUT_DIR=$2 ;;
    --resume-checkpoint) RESUME_CHECKPOINT=$2 ;;
    --resolved-paths) RESOLVED_PATHS=$2 ;;
    --text-model-dir) TEXT_MODEL_DIR=$2 ;;
    --ced-model-dir) CED_MODEL_DIR=$2 ;;
    --index-cache-root) INDEX_CACHE_ROOT=$2 ;;
    --stage-workers) STAGE_WORKERS=$2 ;;
    *) usage ;;
  esac
  shift 2
done
[[ "$STAGE_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "staging workers must be a positive integer" >&2; exit 2; }
[[ -z "$RESUME_CHECKPOINT" || -f "$RESUME_CHECKPOINT" ]] || { echo "formal resume checkpoint does not exist" >&2; exit 2; }
[[ -f "$MANIFEST" && -f "$RESOLVED_PATHS" ]] || { echo "filtered manifest or resolved_paths.tsv missing" >&2; exit 2; }
[[ -d "$TEXT_MODEL_DIR" && -f "$TEXT_MODEL_DIR/config.json" && -d "$CED_MODEL_DIR" ]] || { echo "recursive text checkpoint or CED components missing" >&2; exit 2; }
python "$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_crossattn/check_ced_assets.py" --ced-model-dir "$CED_MODEL_DIR"
RUN_TAG=$(date +%Y%m%d_%H%M%S%N)
OUTPUT_DIR=${OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer36912_crossattn_5090/formal_${RUN_TAG}}
if [[ -e "$OUTPUT_DIR" && ! -d "$OUTPUT_DIR" ]]; then
  echo "output path is not a directory: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ -z "$RESUME_CHECKPOINT" && -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "fresh run requires an empty output directory: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$SCRIPT_DIR/log"
INDEX_PATH_FILE=$(mktemp)
cleanup_index_path() { rm -f "$INDEX_PATH_FILE"; }
trap cleanup_index_path EXIT
python "$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_crossattn/prepare_index.py" \
  --manifest "$MANIFEST" --resolved-paths "$RESOLVED_PATHS" \
  --cache-root "$INDEX_CACHE_ROOT" --path-output "$INDEX_PATH_FILE"
INDEX_DIR=$(tr -d '\r\n' < "$INDEX_PATH_FILE")
[[ -d "$INDEX_DIR" ]] || { echo "CPU-prepared index was not created: $INDEX_DIR" >&2; exit 2; }
ARGS=(formal "$RESOLVED_PATHS" "$OUTPUT_DIR" "$RESUME_CHECKPOINT" "$TEXT_MODEL_DIR" "$CED_MODEL_DIR" "$INDEX_DIR" "$STAGE_WORKERS")
printf -v CMD_ARGS '%q ' "${ARGS[@]}"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "mizar-ced-xattn-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mizar_s1_gbs256_recursive_5_10_5_ced_small_layer36912_crossattn.${RUN_TAG}.JOB.log" \
  --cmd "bash mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_crossattn/run.sh $CMD_ARGS"
echo "Submitted Mizar S1 / CED-Small GBS256 / recursive 5-10-5 + CED layers3/6/9/12 cross attention: 8 GPUs x microbatch8 x accumulation4; output=$OUTPUT_DIR"
echo "Raw audio: shared-path training starts immediately; background local copy then collective switch; workers=$STAGE_WORKERS; index=$INDEX_DIR"
