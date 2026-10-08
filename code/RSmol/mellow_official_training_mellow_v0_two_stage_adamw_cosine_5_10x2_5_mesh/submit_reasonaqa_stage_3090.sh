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
OUTPUT_DIR="${MELLOW_V0_TWO_STAGE_OUTPUT_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0_two_stage_reasonaqa}/${STAGE}_${MODE}_$RUN_TAG"
SOURCE_CHECKPOINT=""
SMOKE_ROWS="${MELLOW_REASONAQA_SMOKE_ROWS:-5120}"
DIAGNOSTICS_STEPS="${MELLOW_RUNTIME_DIAGNOSTICS_STEPS:-0}"
[[ "$DIAGNOSTICS_STEPS" =~ ^(0|[1-9][0-9]{0,2}|1000)$ ]] || { echo "MELLOW_RUNTIME_DIAGNOSTICS_STEPS must be an integer in [0, 1000]" >&2; exit 2; }
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
  -j mellow-v0-two-stage-${STAGE}-${MODE}-3090-$RUN_TAG \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mellow_v0_two_stage_${STAGE}_${MODE}_3090.$RUN_TAG.JOB.log" \
  --cmd "MELLOW_RUNTIME_DIAGNOSTICS_STEPS=$DIAGNOSTICS_STEPS bash scripts/rsmol/run_reasonaqa_8gpu_stage.sh $CMD_ARGS"

echo "Submitted pdgpu-3090 8-GPU Mellow-v0 $STAGE/$MODE; output directory: $OUTPUT_DIR"
