#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG=$(date +%Y%m%d_%H%M%S)
CHECKPOINT="${RSMOL_MELLOW_MCQ_CHECKPOINT_FILE:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5090/mcq_formal_global_token_5epochs_v2/checkpoints/mellow_adamw_cosine_reasonaqa_mcq_formal_20_20261004_173020762176060_26018/model--epo-3.ckpt}"
RUNTIME_CONFIG="${RSMOL_MELLOW_MCQ_RUNTIME_CONFIG:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5090/mcq_formal_global_token_5epochs_v2/runtime_mcq_3epochs.yaml}"
TEST_JSON="${RSMOL_REASONAQA_TEST_JSON:-/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test.json}"
ROUTE_ROOT="${RSMOL_MELLOW_MCQ_ROUTE_ROOT:-$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine}"
OUTPUT_DIR="${RSMOL_REASONAQA_MELLOW_GENERATION_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5090/mcq_formal_global_token_5epochs_v2/samples_generation_${RUN_TAG}}"

if [[ ! -f "$CHECKPOINT" || ! -f "$RUNTIME_CONFIG" || ! -f "$TEST_JSON" ]]; then
  echo "checkpoint, runtime config, or test JSON is missing" >&2
  echo "checkpoint=$CHECKPOINT" >&2
  echo "runtime_config=$RUNTIME_CONFIG" >&2
  echo "test_json=$TEST_JSON" >&2
  exit 2
fi
if [[ ! -d "$ROUTE_ROOT" ]]; then
  echo "official Mellow route root does not exist: $ROUTE_ROOT" >&2
  exit 2
fi

ARGS=(
  --checkpoint-file "$CHECKPOINT"
  --runtime-config "$RUNTIME_CONFIG"
  --route-root "$ROUTE_ROOT"
  --test-json "$TEST_JSON"
  --output-dir "$OUTPUT_DIR"
  --num-samples 5
  --seed 0
  --max-prompt-tokens 129
  --max-new-tokens 64
  --top-p 0.8
  --temperature 1.0
  --prompt-mode official_eval
  --audio-crop-policy random
  --missing-filepath2-policy deterministic_pool
)
if (($#)); then
  ARGS+=("$@")
fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"

JOB_NAME="mellow-reasonaqa-generation-$RUN_TAG"
if ((${#JOB_NAME} > 60)); then
  echo "generated vc job name exceeds 60 characters: $JOB_NAME (${#JOB_NAME})" >&2
  exit 2
fi
JOB_LOG="$SCRIPT_DIR/log/$JOB_NAME.JOB.log"
vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "$JOB_NAME" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/generate_reasonaqa_mellow_official_training.sh $CMD_ARGS"

echo "submitted official Mellow ReasonAQA generation: output=$OUTPUT_DIR"

