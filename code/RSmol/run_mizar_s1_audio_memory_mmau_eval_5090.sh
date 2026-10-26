#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RSMOL_MIZAR_AUDIO_MEMORY_RUN_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_audio_memory_gbs256_5090/formal_20261026_014032838692936/formal}"
EXPECTED_STEP="${RSMOL_MIZAR_AUDIO_MEMORY_EXPECTED_STEP:-5771}"
EXPECTED_GEOMETRY="${RSMOL_MIZAR_AUDIO_MEMORY_EXPECTED_GEOMETRY:-8,8,4}"
CHECKPOINT="${RSMOL_MIZAR_AUDIO_MEMORY_CHECKPOINT:-$RUN_ROOT/checkpoints/model--step-${EXPECTED_STEP}.ckpt}"
RUNTIME_CONFIG="${RSMOL_MIZAR_AUDIO_MEMORY_RUNTIME_CONFIG:-$RUN_ROOT/runtime.json}"
ROUTE_ROOT="${RSMOL_MIZAR_AUDIO_MEMORY_ROUTE_ROOT:-$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_1epoch_audio_memory_recursive_mesh}"
DATASET_DIR="${RSMOL_MMAU_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini}"
OUTPUT_DIR="${RSMOL_MIZAR_AUDIO_MEMORY_MMAU_OUTPUT_DIR:-$(dirname "$RUN_ROOT")/eval/mmau_test_mini_step${EXPECTED_STEP}}"
JOB_LOG="$SCRIPT_DIR/log/mizar_s1_audio_memory_mmau_5090.${RUN_TAG}.JOB.log"

ARGS=(--benchmark mmau --checkpoint "$CHECKPOINT" --runtime-config "$RUNTIME_CONFIG"
      --route-root "$ROUTE_ROOT" --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_DIR"
      --parquet-batch-size 8 --expected-step "$EXPECTED_STEP"
      --expected-geometry "$EXPECTED_GEOMETRY")
if (($#)); then ARGS+=("$@"); fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "mizar-s1-audio-memory-mmau-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash mizar_s1_audio_memory_eval/evaluate_mizar_s1_benchmark.sh $CMD_ARGS"

echo "Submitted full audio-memory MMAU test-mini evaluation; output=$OUTPUT_DIR"
