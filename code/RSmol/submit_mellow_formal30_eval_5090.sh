#!/usr/bin/env bash
set -euo pipefail

die() { echo "[formal30-submit] ERROR: $*" >&2; exit 2; }
if [[ $# -lt 2 ]]; then
  echo "usage: $0 <mmau_test_mini|mmar|both> <smollm2|smollm2_epo25|smollm2_epo15|mesh_x2|mesh_x4|checkpoint-file> [--runtime-config PATH] [--output-dir PATH] [--dataset-dir PATH]" >&2
  echo "For both, output-dir is a parent directory containing mmau_test_mini/ and mmar/." >&2
  exit 2
fi
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TASK=$1
CHECKPOINT=$2
CHECKPOINT_ARGUMENT=$2
shift 2
[[ "$TASK" == mmau_test_mini || "$TASK" == mmar || "$TASK" == both ]] || die "unsupported task: $TASK"
RUNTIME_CONFIG=""
OUTPUT_DIR=""
DATASET_DIR=""
while (($#)); do
  case "$1" in
    --runtime-config|--output-dir|--dataset-dir)
      (($# >= 2)) || die "$1 requires a path"
      case "$1" in
        --runtime-config) RUNTIME_CONFIG=$2 ;;
        --output-dir) OUTPUT_DIR=$2 ;;
        --dataset-dir) DATASET_DIR=$2 ;;
      esac
      shift 2
      ;;
    *) die "unknown argument: $1" ;;
  esac
done
if [[ "$TASK" == both ]]; then
  [[ -z "$DATASET_DIR" ]] || die "--dataset-dir requires one task; MMAU and MMAR have different dataset roots"
  failures=0
  for subtask in mmau_test_mini mmar; do
    extras=()
    [[ -z "$RUNTIME_CONFIG" ]] || extras+=(--runtime-config "$RUNTIME_CONFIG")
    [[ -z "$OUTPUT_DIR" ]] || extras+=(--output-dir "$OUTPUT_DIR/$subtask")
    if bash "$SCRIPT_DIR/submit_mellow_formal30_eval_5090.sh" "$subtask" "$CHECKPOINT" "${extras[@]}"; then
      :
    else
      failures=$((failures + 1))
    fi
  done
  ((failures == 0)) || die "$failures task submission(s) failed"
  exit 0
fi
command -v vc >/dev/null 2>&1 || die "vc command is unavailable"
command -v readlink >/dev/null 2>&1 || die "readlink is required"
ALIAS="custom"
while IFS=$'\t' read -r name path; do
  if [[ "$CHECKPOINT" == "$name" ]]; then
    ALIAS=$name
    CHECKPOINT=$path
    break
  fi
done < "$SCRIPT_DIR/mellow_formal30_eval/checkpoints.tsv"
[[ -f "$CHECKPOINT" ]] || die "checkpoint does not exist yet: $CHECKPOINT"
CHECKPOINT=$(readlink -f -- "$CHECKPOINT")
CHECKPOINT_JOB_DIR=$(dirname "$CHECKPOINT")
RUN_ROOT=$(dirname "$(dirname "$CHECKPOINT_JOB_DIR")")
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$RUN_ROOT/runtime_30epochs.yaml}"
[[ -f "$RUNTIME_CONFIG" ]] || die "runtime config missing: $RUNTIME_CONFIG"
RUNTIME_CONFIG=$(readlink -f -- "$RUNTIME_CONFIG")
CHECKPOINT_JOB_NAME=$(basename "$CHECKPOINT_JOB_DIR")
CHECKPOINT_NAME=$(basename "$CHECKPOINT" .ckpt)
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/eval_formal30_unified_v3/$TASK/$CHECKPOINT_JOB_NAME/$CHECKPOINT_NAME/full}"
OUTPUT_DIR=$(readlink -m -- "$OUTPUT_DIR")
if [[ -z "$DATASET_DIR" ]]; then
  if [[ "$TASK" == mmau_test_mini ]]; then
    DATASET_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini
  else
    DATASET_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR
  fi
fi
[[ -d "$DATASET_DIR" ]] || die "dataset directory missing: $DATASET_DIR"
DATASET_DIR=$(readlink -f -- "$DATASET_DIR")
ARGS=(--task "$TASK" --checkpoint-file "$CHECKPOINT" --runtime-config "$RUNTIME_CONFIG" --output-dir "$OUTPUT_DIR" --dataset-dir "$DATASET_DIR")
printf -v CMD_ARGS '%q ' "${ARGS[@]}"
RUN_TAG=$(date +%Y%m%d_%H%M%S%N)
JOB_NAME="f30-${ALIAS}-${TASK}-${RUN_TAG}"
((${#JOB_NAME} <= 60)) || die "job name exceeds 60 characters: $JOB_NAME"
mkdir -p "$SCRIPT_DIR/log"
JOB_LOG="$SCRIPT_DIR/log/${JOB_NAME}.JOB.log"
echo "[formal30-submit] task=$TASK checkpoint_argument=$CHECKPOINT_ARGUMENT checkpoint=$CHECKPOINT" >&2
echo "[formal30-submit] FP32, audio2=zero_waveform_encoded_independently, prompt EOS, max_new_tokens=300" >&2
echo "[formal30-submit] output_dir=$OUTPUT_DIR" >&2
echo "[formal30-submit] job_log=$JOB_LOG" >&2
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "$JOB_NAME" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash mellow_formal30_eval/run.sh $CMD_ARGS"
echo "[formal30-submit] submitted $TASK on pdgpu-5090; report=$OUTPUT_DIR/evaluation_report.json"
