#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 [--output-dir DIR] [--resume-checkpoint PATH] [--resolved-paths TSV] [--text-model-dir DIR] [--htsat-root DIR] [--stage-start-steps N] [--stage-workers N]" >&2
  exit 2
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESOLVED_PATHS=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_audiocaps_csv_retry1/resolved_paths.tsv
TEXT_MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2
HTSAT_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT
MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_filtered.json
OUTPUT_DIR=""
RESUME_CHECKPOINT=""
STAGE_START_STEPS=64
STAGE_WORKERS=16
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || usage
  case "$1" in
    --output-dir) OUTPUT_DIR=$2 ;;
    --resume-checkpoint) RESUME_CHECKPOINT=$2 ;;
    --resolved-paths) RESOLVED_PATHS=$2 ;;
    --text-model-dir) TEXT_MODEL_DIR=$2 ;;
    --htsat-root) HTSAT_ROOT=$2 ;;
    --stage-start-steps) STAGE_START_STEPS=$2 ;;
    --stage-workers) STAGE_WORKERS=$2 ;;
    *) usage ;;
  esac
  shift 2
done
[[ "$STAGE_START_STEPS" =~ ^[1-9][0-9]*$ && "$STAGE_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "staging steps/workers must be positive integers" >&2; exit 2; }
[[ -z "$RESUME_CHECKPOINT" || -f "$RESUME_CHECKPOINT" ]] || { echo "formal resume checkpoint does not exist" >&2; exit 2; }
[[ -f "$MANIFEST" && -f "$RESOLVED_PATHS" ]] || { echo "filtered manifest or resolved_paths.tsv missing" >&2; exit 2; }
[[ -d "$TEXT_MODEL_DIR" && -f "$HTSAT_ROOT/HTSAT_AudioSet_Saved_1.ckpt" ]] || { echo "local pretrained components missing" >&2; exit 2; }
RUN_TAG=$(date +%Y%m%d_%H%M%S%N)
OUTPUT_DIR=${OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_5090/formal_${RUN_TAG}}
if [[ -e "$OUTPUT_DIR" && ! -d "$OUTPUT_DIR" ]]; then
  echo "output path is not a directory: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ -z "$RESUME_CHECKPOINT" && -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "fresh run requires an empty output directory: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$SCRIPT_DIR/log"
ARGS=(formal "$RESOLVED_PATHS" "$OUTPUT_DIR" "$RESUME_CHECKPOINT" "$TEXT_MODEL_DIR" "$HTSAT_ROOT" "$STAGE_START_STEPS" "$STAGE_WORKERS")
printf -v CMD_ARGS '%q ' "${ARGS[@]}"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "mizar-s1-gbs256-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mizar_s1_gbs256.${RUN_TAG}.JOB.log" \
  --cmd "bash mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs/run.sh $CMD_ARGS"
echo "Submitted Mizar S1 / HTSAT GBS256: 8 GPUs x microbatch8 x accumulation4; output=$OUTPUT_DIR"
echo "Raw audio staging: first $STAGE_START_STEPS optimizer updates, then background copy; workers=$STAGE_WORKERS"
