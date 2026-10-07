#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
CHECKPOINT="${RSMOL_X4_ABLATION_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/formal_3epochs_20261002_configfix_v3/checkpoint-011343}"
TEST_JSON="${RSMOL_REASONAQA_TEST_JSON:-/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test.json}"
OUTPUT_DIR="${RSMOL_X4_ABLATION_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/reasonaqa_x4_two_loop_ablation/samples_generation_${RUN_TAG}}"
if [[ ! -d "$CHECKPOINT" || ! -f "$TEST_JSON" ]]; then
  echo "checkpoint or ReasonAQA test JSON is missing: $CHECKPOINT $TEST_JSON" >&2
  exit 2
fi
ARGS=(--checkpoint "$CHECKPOINT" --test-json "$TEST_JSON" --output-dir "$OUTPUT_DIR" --num-samples 5 --seed 0 --max-prompt-tokens 129 --max-new-tokens 64)
if (($#)); then ARGS+=("$@"); fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"
JOB_NAME="reasonaqa-x4-two-loop-${RUN_TAG}"
JOB_LOG="$SCRIPT_DIR/log/$JOB_NAME.JOB.log"
vc submit -p pdgpu-3090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 -j "$JOB_NAME" -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/generate_reasonaqa_x4_two_loop_ablation.sh $CMD_ARGS"
echo "x4 two-loop ReasonAQA output: $OUTPUT_DIR"
