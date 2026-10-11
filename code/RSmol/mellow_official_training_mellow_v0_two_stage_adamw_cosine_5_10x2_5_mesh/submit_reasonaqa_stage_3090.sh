#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 7 ]]; then
  echo "usage: $0 <stage1|stage2> <smoke|resume|formal> [audit-report] [mapping-jsonl] [output-dir] [source-checkpoint] [smoke-rows=5120]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE="$1"
MODE="$2"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
AUDIT_REPORT="${MELLOW_REASONAQA_AUDIT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/path_audit.json}"
MAPPING_JSONL="${MELLOW_REASONAQA_MAPPING_JSONL:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/train_audio_mapping.jsonl}"
STAGE2_PROFILE="${MELLOW_STAGE2_PROFILE:-stage2_gbs256_lr1e-4_min1e-5}"
STAGE2_OUTPUT_SUFFIX=""
if [[ "$STAGE" == stage2 ]]; then STAGE2_OUTPUT_SUFFIX="_${STAGE2_PROFILE}"; fi
OUTPUT_DIR="${MELLOW_V0_TWO_STAGE_OUTPUT_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0_two_stage_reasonaqa}/${STAGE}_${MODE}${STAGE2_OUTPUT_SUFFIX}_$RUN_TAG"
JOB_PROFILE_TAG=""
if [[ "$STAGE" == stage2 ]]; then
  case "$STAGE2_PROFILE" in
    stage2_gbs32_lr5e-4) JOB_PROFILE_TAG="g32" ;;
    stage2_gbs256_lr1e-4) JOB_PROFILE_TAG="g256-old" ;;
    stage2_gbs256_lr1e-4_min1e-5) JOB_PROFILE_TAG="g256-l14m15" ;;
    *) echo "unsupported MELLOW_STAGE2_PROFILE: $STAGE2_PROFILE" >&2; exit 2 ;;
  esac
fi
JOB_NAME="mellow-v0-${STAGE}-${MODE}"
if [[ -n "$JOB_PROFILE_TAG" ]]; then JOB_NAME="${JOB_NAME}-${JOB_PROFILE_TAG}"; fi
JOB_NAME="${JOB_NAME}-3090-${RUN_TAG}"
if [[ ${#JOB_NAME} -gt 60 ]]; then
  echo "job name exceeds vc limit of 60 characters (${#JOB_NAME}): $JOB_NAME" >&2
  exit 2
fi
SOURCE_CHECKPOINT=""
SMOKE_ROWS="${MELLOW_REASONAQA_SMOKE_ROWS:-5120}"
DIAGNOSTICS_STEPS="${MELLOW_RUNTIME_DIAGNOSTICS_STEPS:-0}"
[[ "$DIAGNOSTICS_STEPS" =~ ^(0|[1-9][0-9]{0,2}|1000)$ ]] || { echo "MELLOW_RUNTIME_DIAGNOSTICS_STEPS must be an integer in [0, 1000]" >&2; exit 2; }
[[ "$STAGE2_PROFILE" == stage2_gbs32_lr5e-4 || "$STAGE2_PROFILE" == stage2_gbs256_lr1e-4 || "$STAGE2_PROFILE" == stage2_gbs256_lr1e-4_min1e-5 ]] || { echo "unsupported MELLOW_STAGE2_PROFILE: $STAGE2_PROFILE" >&2; exit 2; }
[[ $# -ge 3 ]] && AUDIT_REPORT="$3"
[[ $# -ge 4 ]] && MAPPING_JSONL="$4"
[[ $# -ge 5 ]] && OUTPUT_DIR="$5"
if [[ $# -ge 6 ]]; then
  if [[ "$6" =~ ^[0-9]+$ ]] && [[ "$MODE" != resume && "$STAGE" != stage2 ]]; then
    SMOKE_ROWS="$6"
  else
    SOURCE_CHECKPOINT="$6"
    [[ $# -ge 7 ]] && SMOKE_ROWS="$7"
  fi
fi

if [[ "$MODE" == resume || "$STAGE" == stage2 ]]; then
  [[ -n "$SOURCE_CHECKPOINT" && -f "$SOURCE_CHECKPOINT" ]] || { echo "source checkpoint is required for $STAGE/$MODE" >&2; exit 2; }
fi
if [[ "$MODE" != formal && "$SMOKE_ROWS" -ne 5120 ]]; then
  echo "smoke and resume contracts require exactly 5120 rows" >&2
  exit 2
fi

mkdir -p "$SCRIPT_DIR/log"
ARGS=("$STAGE" "$MODE" "$AUDIT_REPORT" "$MAPPING_JSONL" "$OUTPUT_DIR")
if [[ -n "$SOURCE_CHECKPOINT" ]]; then ARGS+=("$SOURCE_CHECKPOINT"); fi
if [[ "$MODE" != formal ]]; then ARGS+=("$SMOKE_ROWS"); fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "$JOB_NAME" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mellow_v0_two_stage_${STAGE}_${MODE}${STAGE2_OUTPUT_SUFFIX}_3090.$RUN_TAG.JOB.log" \
  --cmd "MELLOW_RUNTIME_DIAGNOSTICS_STEPS=$DIAGNOSTICS_STEPS MELLOW_STAGE2_PROFILE=$STAGE2_PROFILE bash scripts/rsmol/run_reasonaqa_8gpu_stage.sh $CMD_ARGS"

echo "Submitted pdgpu-3090 8-GPU Mellow-v0 $STAGE/$MODE; output directory: $OUTPUT_DIR"
