#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EVAL_VARIANT="${RSMOL_MIZAR_USAD2_EVAL_VARIANT:-base}"
case "$EVAL_VARIANT" in
  base)
    DEFAULT_RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_usad2_small_gbs256_5090/formal_20261027_115510113068168
    DEFAULT_ROUTE_ROOT="$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_usad2_small"
    EVAL_SCRIPT=scripts/evaluate_mizar_s1_usad2_benchmark.sh
    JOB_NAME_PREFIX=mizar-s1-usad2-mmar
    JOB_LOG_PREFIX=mizar_s1_usad2_small_mmar_5090
    ;;
  recursive_mesh)
    DEFAULT_RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_usad2_small_gbs256_recursive_mesh_5090/formal_20261027_160151646726596
    DEFAULT_ROUTE_ROOT="$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_usad2_small"
    EVAL_SCRIPT=scripts/evaluate_mizar_s1_usad2_recursive_mesh_benchmark.sh
    JOB_NAME_PREFIX=mizar-s1-usad2-mesh-mmar
    JOB_LOG_PREFIX=mizar_s1_usad2_small_recursive_mesh_mmar_5090
    ;;
  *)
    echo "unsupported RSMOL_MIZAR_USAD2_EVAL_VARIANT: $EVAL_VARIANT (expected base or recursive_mesh)" >&2
    exit 2
    ;;
esac
RUN_ROOT="${RSMOL_MIZAR_USAD2_RUN_ROOT:-$DEFAULT_RUN_ROOT}"
EXPECTED_STEP="${RSMOL_MIZAR_USAD2_EXPECTED_STEP:-17313}"
EXPECTED_GEOMETRY="${RSMOL_MIZAR_USAD2_EXPECTED_GEOMETRY:-8,8,4}"
CHECKPOINT="${RSMOL_MIZAR_USAD2_CHECKPOINT:-$RUN_ROOT/formal/checkpoints/model--step-$EXPECTED_STEP.ckpt}"
RUNTIME_CONFIG="${RSMOL_MIZAR_USAD2_RUNTIME_CONFIG:-$RUN_ROOT/formal/runtime.json}"
ROUTE_ROOT="${RSMOL_MIZAR_USAD2_ROUTE_ROOT:-$DEFAULT_ROUTE_ROOT}"
DATASET_DIR="${RSMOL_MMAR_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR}"
OUTPUT_DIR="${RSMOL_MIZAR_USAD2_MMAR_OUTPUT_DIR:-$RUN_ROOT/eval/mmar_step$EXPECTED_STEP}"
JOB_LOG="$SCRIPT_DIR/log/${JOB_LOG_PREFIX}.${RUN_TAG}.JOB.log"

ARGS=(--benchmark mmar --checkpoint "$CHECKPOINT" --runtime-config "$RUNTIME_CONFIG"
      --route-root "$ROUTE_ROOT" --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_DIR"
      --expected-step "$EXPECTED_STEP" --expected-geometry "$EXPECTED_GEOMETRY")
if (($#)); then ARGS+=("$@"); fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "${JOB_NAME_PREFIX}-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash $EVAL_SCRIPT $CMD_ARGS"

echo "Submitted full USAD2-Small MMAR evaluation; output=$OUTPUT_DIR"
echo "Evaluation variant=$EVAL_VARIANT; checkpoint=$CHECKPOINT; runtime=$RUNTIME_CONFIG"
