#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 ]]; then
  echo "usage: $0 <checkpoint-file> [runtime-config] [output-dir] [full|smoke] [dataset-dir]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHECKPOINT=$(readlink -f "$1")
CHECKPOINT_JOB_DIR=$(dirname "$CHECKPOINT")
CHECKPOINT_DIR=$(dirname "$CHECKPOINT_JOB_DIR")
DEFAULT_OUTPUT_ROOT=$(dirname "$CHECKPOINT_DIR")
RUNTIME_CONFIG="${2:-$DEFAULT_OUTPUT_ROOT/runtime_stage1_formal.yaml}"
OUTPUT_DIR="${3:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0_two_stage_reasonaqa/eval_stage1_formal_epoch5_mmau_test_mini}"
MODE="${4:-full}"
DATASET_DIR="${5:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini}"

[[ "$MODE" == full || "$MODE" == smoke ]] || { echo "mode must be full or smoke" >&2; exit 2; }
[[ -f "$CHECKPOINT" && -f "$RUNTIME_CONFIG" ]] || {
  echo "checkpoint or runtime config missing" >&2
  exit 2
}
mkdir -p "$SCRIPT_DIR/log"
RUN_TAG=$(date +%Y%m%d_%H%M%S)

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 16 -m 64G -g 1 -n 1 \
  -j mellow-v0-two-stage-mmau-${MODE}-3090-$RUN_TAG \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mellow_v0_two_stage_mmau_${MODE}_3090.$RUN_TAG.JOB.log" \
  --cmd "bash scripts/rsmol/evaluate_mmau_test_mini_mellow_v0_two_stage.sh --mode $MODE --checkpoint-file $CHECKPOINT --runtime-config $RUNTIME_CONFIG --route-root $SCRIPT_DIR --output-dir $OUTPUT_DIR --dataset-dir $DATASET_DIR --run-official-evaluation"

echo "Submitted independent MMAU $MODE evaluation on pdgpu-3090; output directory: $OUTPUT_DIR"
