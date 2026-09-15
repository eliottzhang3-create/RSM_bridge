#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
CHECKPOINT="${RSMOL_SMOLLM2_GENERATION_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/formal_20260911_v1/checkpoint-011343}"
TEST_MANIFEST="${RSMOL_REASONAQA_TEST_MANIFEST:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_test.jsonl}"
HTSAT_CHECKPOINT="${RSMOL_HTSAT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt}"
MELLOW_ROOT="${RSMOL_MELLOW_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main}"
OUTPUT_DIR="${RSMOL_SMOLLM2_GENERATION_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/reasonaqa_samples_checkpoint011343_${RUN_TAG}}"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --test-manifest "$TEST_MANIFEST"
  --htsat-checkpoint "$HTSAT_CHECKPOINT"
  --mellow-root "$MELLOW_ROOT"
  --output-dir "$OUTPUT_DIR"
  --num-samples 5
  --seed 0
  --max-new-tokens 16
  --dtype bf16
)
if (($#)); then
  ARGS+=("$@")
fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 64G -g 1 -n 1 \
  -j audio-smollm2-reasonaqa-gen-3090-$(date +%m%d%H%M%S) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/audio_smollm2_reasonaqa_generation_3090.JOB.log" \
  --cmd "bash scripts/generate_audio_smollm2_checkpoint_reasonaqa.sh $CMD_ARGS"

echo "SmolLM2 ReasonAQA generation output directory: ${OUTPUT_DIR}"
