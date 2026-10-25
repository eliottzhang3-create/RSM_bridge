#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 [--mode smoke|formal] [--output-dir DIR] [--init-checkpoint PATH] [--resume-checkpoint PATH] [--resolved-paths TSV] [--text-model-dir DIR] [--htsat-root DIR] [--index-cache-root DIR] [--stage-workers N]" >&2
  exit 2
}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTE=mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_1epoch_audio_memory_recursive_mesh
RESOLVED_PATHS=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
TEXT_MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244
HTSAT_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT
MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
INIT_CHECKPOINT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_recursive_mesh_5090/formal_recursive_mesh_padfix_20261025_003124507033014/formal/checkpoints/model--step-17313.ckpt
INDEX_CACHE_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_audio_memory_gbs256_5090/index_cache
MODE=formal
OUTPUT_DIR=""
RESUME_CHECKPOINT=""
STAGE_WORKERS=4
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || usage
  case "$1" in
    --mode) MODE=$2 ;;
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
[[ "$MODE" == formal || "$MODE" == smoke ]] || usage
[[ "$STAGE_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "stage workers must be positive" >&2; exit 2; }
[[ "$MODE" != smoke || -z "$RESUME_CHECKPOINT" ]] || { echo "smoke cannot resume" >&2; exit 2; }
[[ -z "$RESUME_CHECKPOINT" || -f "$RESUME_CHECKPOINT" ]] || { echo "resume checkpoint missing" >&2; exit 2; }
[[ -f "$INIT_CHECKPOINT" && -f "$MANIFEST" && -f "$RESOLVED_PATHS" ]] || { echo "init checkpoint, manifest or resolved paths missing" >&2; exit 2; }
[[ -f "$TEXT_MODEL_DIR/config.json" && -f "$HTSAT_ROOT/HTSAT_AudioSet_Saved_1.ckpt" ]] || { echo "text config/tokenizer or HTSAT asset missing" >&2; exit 2; }
RUN_TAG=$(date +%Y%m%d_%H%M%S%N)
OUTPUT_DIR=${OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_audio_memory_gbs256_5090/${MODE}_${RUN_TAG}}
[[ ! -e "$OUTPUT_DIR" || -d "$OUTPUT_DIR" ]] || { echo "output is not a directory" >&2; exit 2; }
if [[ -z "$RESUME_CHECKPOINT" && -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "fresh run requires an empty output directory: $OUTPUT_DIR" >&2; exit 2
fi
mkdir -p "$SCRIPT_DIR/log"
INDEX_PATH_FILE=$(mktemp)
cleanup_index_path() { rm -f "$INDEX_PATH_FILE"; }
trap cleanup_index_path EXIT
# Metadata/path indexing only: no audio copy, decode, or encoder cache.
python "$SCRIPT_DIR/$ROUTE/prepare_index.py" --manifest "$MANIFEST" \
  --resolved-paths "$RESOLVED_PATHS" --cache-root "$INDEX_CACHE_ROOT" --path-output "$INDEX_PATH_FILE"
INDEX_DIR=$(tr -d '\r\n' < "$INDEX_PATH_FILE")
[[ -d "$INDEX_DIR" ]] || { echo "CPU index missing: $INDEX_DIR" >&2; exit 2; }
ARGS=("$MODE" "$RESOLVED_PATHS" "$OUTPUT_DIR" "$RESUME_CHECKPOINT" "$TEXT_MODEL_DIR" "$HTSAT_ROOT" "$INDEX_DIR" "$STAGE_WORKERS" "$INIT_CHECKPOINT")
printf -v CMD_ARGS '%q ' "${ARGS[@]}"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "mizar-s1-audio-memory-${MODE}-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mizar_s1_audio_memory.${MODE}.${RUN_TAG}.JOB.log" \
  --cmd "bash $ROUTE/run.sh $CMD_ARGS"
echo "Submitted $MODE: audio memory MeSH, 8 GPUs x microbatch8 x accumulation4; output=$OUTPUT_DIR"
if [[ "$MODE" == smoke ]]; then
  echo "Smoke: exactly five optimizer updates, direct original audio, no staging; schedule horizon=5771."
else
  echo "Formal: one epoch / 5771 updates; shared audio starts immediately, background copy then collective local switch."
fi
