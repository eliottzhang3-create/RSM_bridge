#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 [--model-dir DIR] [--output-dir DIR] [--resume-checkpoint PATH] [--resolved-paths TSV] [--index-cache-root DIR] [--stage-workers N]" >&2
  exit 2
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTE_ROOT="$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_official_final_s1_gbs128_5_20x2_5_input_only"
MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-5-20x2-5-input-reinject
RESOLVED_PATHS=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
OUTPUT_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_official_final_s1_gbs128_5_20x2_5_input_only_5090
INDEX_CACHE_ROOT="$OUTPUT_BASE/index_cache"
OUTPUT_DIR=""
RESUME_CHECKPOINT=""
STAGE_WORKERS=4
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || usage
  case "$1" in
    --model-dir) MODEL_DIR=$2 ;;
    --output-dir) OUTPUT_DIR=$2 ;;
    --resume-checkpoint) RESUME_CHECKPOINT=$2 ;;
    --resolved-paths) RESOLVED_PATHS=$2 ;;
    --index-cache-root) INDEX_CACHE_ROOT=$2 ;;
    --stage-workers) STAGE_WORKERS=$2 ;;
    *) usage ;;
  esac
  shift 2
done
[[ "$STAGE_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "staging workers must be a positive integer" >&2; exit 2; }
[[ -f "$MANIFEST" && -f "$RESOLVED_PATHS" ]] || { echo "filtered manifest or resolved_paths.tsv missing" >&2; exit 2; }
[[ -f "$MODEL_DIR/manifest.json" && -f "$MODEL_DIR/model.ckpt" && -d "$MODEL_DIR/tokenizer" && -d "$MODEL_DIR/ced-small" ]] || { echo "complete converted official Mizar bundle missing: $MODEL_DIR" >&2; exit 2; }
# This is a read-only stdlib file audit, not a smoke job or tensor prerequisite.
python "$ROUTE_ROOT/check_model.py" --model-dir "$MODEL_DIR"
RUN_TAG=$(date +%Y%m%d_%H%M%S%N)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  [[ -f "$RESUME_CHECKPOINT" ]] || { echo "formal resume checkpoint does not exist" >&2; exit 2; }
  RESUME_CHECKPOINT=$(readlink -f "$RESUME_CHECKPOINT")
  RESUME_ROOT=$(dirname "$(dirname "$(dirname "$RESUME_CHECKPOINT")")")
  OUTPUT_DIR=${OUTPUT_DIR:-$RESUME_ROOT}
  [[ "$(readlink -m "$OUTPUT_DIR")/formal/checkpoints" == "$(dirname "$RESUME_CHECKPOINT")" ]] || { echo "resume requires the original output root/formal/checkpoints" >&2; exit 2; }
fi
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_BASE/formal_${RUN_TAG}}
if [[ -e "$OUTPUT_DIR" && ! -d "$OUTPUT_DIR" ]]; then
  echo "output path is not a directory: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ -z "$RESUME_CHECKPOINT" && -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "fresh run requires an empty output directory: $OUTPUT_DIR" >&2
  exit 2
fi
LOG_ROOT="$SCRIPT_DIR/log/mizar_official_final_s1_gbs128_5_20x2_5_input_only"
mkdir -p "$LOG_ROOT"
INDEX_PATH_FILE=$(mktemp)
cleanup_index_path() { rm -f "$INDEX_PATH_FILE"; }
trap cleanup_index_path EXIT
python "$ROUTE_ROOT/prepare_index.py" --manifest "$MANIFEST" --resolved-paths "$RESOLVED_PATHS" \
  --cache-root "$INDEX_CACHE_ROOT" --path-output "$INDEX_PATH_FILE"
INDEX_DIR=$(tr -d '\r\n' < "$INDEX_PATH_FILE")
[[ -d "$INDEX_DIR" ]] || { echo "CPU-prepared index was not created: $INDEX_DIR" >&2; exit 2; }
ARGS=(formal "$RESOLVED_PATHS" "$OUTPUT_DIR" "$RESUME_CHECKPOINT" "$MODEL_DIR" "$INDEX_DIR" "$STAGE_WORKERS")
printf -v CMD_ARGS '%q ' "${ARGS[@]}"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "mizar-official-20x2-input-g128-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$LOG_ROOT/formal.${RUN_TAG}.JOB.log" \
  --cmd "bash mellow_official_training_c8204d8_adamw_cosine/mizar_official_final_s1_gbs128_5_20x2_5_input_only/run.sh $CMD_ARGS"
echo "Submitted official-final Mizar S1: 5+20x2+5 / input-only reinjection; 8 x microbatch4 x GA4 = GBS128; 34626 updates; output=$OUTPUT_DIR"
echo "Audio reads shared paths immediately while staging runs; all ranks switch at an optimizer boundary after complete copy; workers=$STAGE_WORKERS"
