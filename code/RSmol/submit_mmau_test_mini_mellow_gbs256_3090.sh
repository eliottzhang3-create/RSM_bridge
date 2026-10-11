#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "[gbs256-submit] ERROR: $*" >&2
  exit 2
}

on_exit() {
  local rc=$?
  if ((rc != 0)); then
    echo "[gbs256-submit] FAILED (exit $rc)" >&2
  fi
}
trap on_exit EXIT

if [[ $# -lt 1 || $# -gt 6 ]]; then
  echo "usage: $0 <checkpoint-file> [runtime-config] [output-dir] [full|smoke] [dataset-dir] [route-root]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
command -v readlink >/dev/null 2>&1 || die "readlink is required"
command -v vc >/dev/null 2>&1 || die "vc command is not available in this shell"
CHECKPOINT=$(readlink -f -- "$1") || die "cannot canonicalize checkpoint path: $1"
CHECKPOINT_JOB_DIR=$(dirname "$CHECKPOINT")
CHECKPOINT_DIR=$(dirname "$CHECKPOINT_JOB_DIR")
RUN_ROOT=$(dirname "$CHECKPOINT_DIR")
RUNTIME_CONFIG="${2:-$RUN_ROOT/runtime_5epochs.yaml}"
MODE="${4:-full}"
CHECKPOINT_JOB_NAME=$(basename "$CHECKPOINT_JOB_DIR")
CHECKPOINT_NAME=$(basename "$CHECKPOINT" .ckpt)
OUTPUT_DIR="${3:-$RUN_ROOT/eval_mmau_test_mini/$CHECKPOINT_JOB_NAME/$CHECKPOINT_NAME/$MODE}"
DATASET_DIR="${5:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini}"
ROUTE_ROOT="${6:-}"

[[ "$MODE" == full || "$MODE" == smoke ]] || die "mode must be full or smoke: $MODE"
[[ -f "$CHECKPOINT" ]] || die "checkpoint not found: $CHECKPOINT"
[[ -f "$RUNTIME_CONFIG" ]] || die "runtime config not found: $RUNTIME_CONFIG"
[[ -d "$DATASET_DIR" ]] || die "MMAU dataset directory not found: $DATASET_DIR"

mkdir -p "$SCRIPT_DIR/log" || die "cannot create log directory: $SCRIPT_DIR/log"
RUN_TAG=$(date +%Y%m%d_%H%M%S%N)
JOB_NAME="mellow-gbs256-mmau-${MODE}-3090-${RUN_TAG}"
JOB_LOG="$SCRIPT_DIR/log/${JOB_NAME}.JOB.log"
(( ${#JOB_NAME} <= 60 )) || die "generated vc job name is longer than 60 characters: $JOB_NAME"
ARGS=(--mode "$MODE" --checkpoint-file "$CHECKPOINT" --runtime-config "$RUNTIME_CONFIG" --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_DIR" --parquet-batch-size 8 --max-prompt-tokens 129 --max-new-tokens 300 --dtype fp32)
if [[ -n "$ROUTE_ROOT" ]]; then ARGS+=(--route-root "$ROUTE_ROOT"); fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"

echo "[gbs256-submit] preflight PASS" >&2
echo "[gbs256-submit] checkpoint=$CHECKPOINT" >&2
echo "[gbs256-submit] runtime_config=$RUNTIME_CONFIG" >&2
echo "[gbs256-submit] dataset_dir=$DATASET_DIR" >&2
echo "[gbs256-submit] output_dir=$OUTPUT_DIR" >&2
echo "[gbs256-submit] job_name=$JOB_NAME" >&2
echo "[gbs256-submit] job_log=$JOB_LOG" >&2

if VC_OUTPUT=$(vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 16 -m 64G -g 1 -n 1 \
  -j "$JOB_NAME" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/evaluate_mmau_test_mini_mellow_gbs256.sh $CMD_ARGS" 2>&1); then
  printf '%s\n' "$VC_OUTPUT"
else
  rc=$?
  printf '%s\n' "$VC_OUTPUT" >&2
  echo "[gbs256-submit] vc submit failed; inspect $JOB_LOG if it was created" >&2
  exit "$rc"
fi

echo "Submitted Mellow gbs256 MMAU $MODE evaluation on pdgpu-3090; checkpoint: $CHECKPOINT; output directory: $OUTPUT_DIR"
