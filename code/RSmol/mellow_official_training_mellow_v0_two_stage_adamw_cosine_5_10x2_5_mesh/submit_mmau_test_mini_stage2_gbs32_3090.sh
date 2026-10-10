#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHECKPOINT="${MELLOW_V0_TWO_STAGE_STAGE2_MMAU_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0_two_stage_reasonaqa/stage2_formal_5epochs_gbs32_20261009_111507/checkpoints/mellow_v0_two_stage_stage2_formal_20_20261009_031515074161572/model--epo-5.ckpt}"
RUNTIME_CONFIG="${MELLOW_V0_TWO_STAGE_STAGE2_MMAU_RUNTIME:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0_two_stage_reasonaqa/stage2_formal_5epochs_gbs32_20261009_111507/runtime_stage2_formal.yaml}"
OUTPUT_DIR="${MELLOW_V0_TWO_STAGE_STAGE2_MMAU_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0_two_stage_reasonaqa/eval_stage2_formal_epoch5_mmau_test_mini_20261009_gbs32}"
DATASET_DIR="${MELLOW_MMAU_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini}"

if [[ $# -gt 4 ]]; then
  echo "usage: $0 [checkpoint-file] [runtime-config] [output-dir] [dataset-dir]" >&2
  exit 2
fi
CHECKPOINT="${1:-$CHECKPOINT}"
RUNTIME_CONFIG="${2:-$RUNTIME_CONFIG}"
OUTPUT_DIR="${3:-$OUTPUT_DIR}"
DATASET_DIR="${4:-$DATASET_DIR}"

[[ -f "$CHECKPOINT" ]] || { echo "Stage 2 GBS32 checkpoint does not exist: $CHECKPOINT" >&2; exit 2; }
[[ -f "$RUNTIME_CONFIG" ]] || { echo "Stage 2 runtime config does not exist: $RUNTIME_CONFIG" >&2; exit 2; }
[[ -d "$DATASET_DIR" ]] || { echo "MMAU dataset directory does not exist: $DATASET_DIR" >&2; exit 2; }
mkdir -p "$SCRIPT_DIR/log"

RUN_TAG=$(date +%Y%m%d_%H%M%S)
JOB_NAME="mellow-v0-stage2-gbs32-mmau-$RUN_TAG"
JOB_LOG="$SCRIPT_DIR/log/$JOB_NAME.JOB.log"
ARGS=(
  --mode full
  --expected-training-stage stage2
  --checkpoint-file "$CHECKPOINT"
  --runtime-config "$RUNTIME_CONFIG"
  --route-root "$SCRIPT_DIR"
  --dataset-dir "$DATASET_DIR"
  --output-dir "$OUTPUT_DIR"
  --parquet-batch-size 8
  --max-prompt-tokens 129
  --max-new-tokens 300
  --dtype fp32
  --run-official-evaluation
)
printf -v CMD_ARGS '%q ' "${ARGS[@]}"

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 16 -m 64G -g 1 -n 1 \
  -j "$JOB_NAME" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/rsmol/evaluate_mmau_test_mini_mellow_v0_two_stage.sh $CMD_ARGS"

echo "Submitted full Stage 2 GBS32 MMAU evaluation; output directory: $OUTPUT_DIR"
