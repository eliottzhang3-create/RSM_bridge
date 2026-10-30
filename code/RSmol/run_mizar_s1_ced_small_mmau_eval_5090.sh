#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EVAL_VARIANT="${RSMOL_MIZAR_CED_EVAL_VARIANT:-base}"
case "$EVAL_VARIANT" in
  base)
    DEFAULT_RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_5090/formal_20261026_193830617147305
    DEFAULT_ROUTE_ROOT="$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_ced_small"
    EVAL_SCRIPT=scripts/evaluate_mizar_s1_ced_benchmark.sh
    JOB_NAME_PREFIX=mizar-s1-ced-mmau
    JOB_LOG_PREFIX=mizar_s1_ced_small_mmau_5090
    ;;
  recursive_mesh)
    DEFAULT_RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_5090/formal_20261027_081123150934256
    DEFAULT_ROUTE_ROOT="$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small"
    EVAL_SCRIPT=scripts/evaluate_mizar_s1_ced_recursive_mesh_benchmark.sh
    JOB_NAME_PREFIX=mizar-s1-ced-mesh-mmau
    JOB_LOG_PREFIX=mizar_s1_ced_small_recursive_mesh_mmau_5090
    ;;
  recursive_5_10_5|recursive_5_10_5_layer8_4_inject|recursive_5_10_5_layer4_8_inject|recursive_5_10_5_audio_reinject)
    case "$EVAL_VARIANT" in
      recursive_5_10_5) ROUTE_SUFFIX=; RUN_STAMP=20261028_160823559722647; JOB_TAG=plain ;;
      recursive_5_10_5_layer8_4_inject) ROUTE_SUFFIX=_layer8_4_inject; RUN_STAMP=20261030_000807991327585; JOB_TAG=layer84 ;;
      recursive_5_10_5_layer4_8_inject) ROUTE_SUFFIX=_layer4_8_inject; RUN_STAMP=20261029_210116592758429; JOB_TAG=layer48 ;;
      recursive_5_10_5_audio_reinject) ROUTE_SUFFIX=_audio_reinject; RUN_STAMP=20261029_110929631468450; JOB_TAG=reinject ;;
    esac
    DEFAULT_RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5${ROUTE_SUFFIX}_5090/formal_${RUN_STAMP}"
    DEFAULT_ROUTE_ROOT="$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small${ROUTE_SUFFIX}"
    EVAL_SCRIPT=scripts/evaluate_mizar_s1_ced_recursive_benchmark.sh
    JOB_NAME_PREFIX="mizar-ced-r515-${JOB_TAG}-mmau"
    JOB_LOG_PREFIX="mizar_s1_ced_small_${EVAL_VARIANT}_mmau_5090"
    ;;
  *)
    echo "unsupported RSMOL_MIZAR_CED_EVAL_VARIANT: $EVAL_VARIANT (expected base, recursive_mesh, or a supported recursive_5_10_5 variant)" >&2
    exit 2
    ;;
esac
RUN_ROOT="${RSMOL_MIZAR_CED_RUN_ROOT:-$DEFAULT_RUN_ROOT}"
EXPECTED_STEP="${RSMOL_MIZAR_CED_EXPECTED_STEP:-17313}"
EXPECTED_GEOMETRY="${RSMOL_MIZAR_CED_EXPECTED_GEOMETRY:-8,8,4}"
CHECKPOINT="${RSMOL_MIZAR_CED_CHECKPOINT:-$RUN_ROOT/formal/checkpoints/model--step-$EXPECTED_STEP.ckpt}"
RUNTIME_CONFIG="${RSMOL_MIZAR_CED_RUNTIME_CONFIG:-$RUN_ROOT/formal/runtime.json}"
ROUTE_ROOT="${RSMOL_MIZAR_CED_ROUTE_ROOT:-$DEFAULT_ROUTE_ROOT}"
DATASET_DIR="${RSMOL_MMAU_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini}"
OUTPUT_DIR="${RSMOL_MIZAR_CED_MMAU_OUTPUT_DIR:-$RUN_ROOT/eval/mmau_test_mini_step$EXPECTED_STEP}"
JOB_LOG="$SCRIPT_DIR/log/${JOB_LOG_PREFIX}.${RUN_TAG}.JOB.log"

ARGS=(--benchmark mmau --checkpoint "$CHECKPOINT" --runtime-config "$RUNTIME_CONFIG"
      --route-root "$ROUTE_ROOT" --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_DIR"
      --parquet-batch-size 8 --expected-step "$EXPECTED_STEP"
      --expected-geometry "$EXPECTED_GEOMETRY")
if [[ "$EVAL_VARIANT" == recursive_5_10_5* ]]; then ARGS+=(--eval-variant "$EVAL_VARIANT"); fi
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

echo "Submitted full CED-Small MMAU test-mini evaluation; output=$OUTPUT_DIR"
echo "Evaluation variant=$EVAL_VARIANT; checkpoint=$CHECKPOINT; runtime=$RUNTIME_CONFIG"
