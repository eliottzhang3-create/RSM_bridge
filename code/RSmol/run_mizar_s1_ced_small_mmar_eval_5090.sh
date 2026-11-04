#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EVAL_VARIANT="${RSMOL_MIZAR_CED_EVAL_VARIANT:-base}"
DEFAULT_EXPECTED_STEP=17313
DEFAULT_EXPECTED_GEOMETRY=8,8,4
case "$EVAL_VARIANT" in
  official_mizar)
    exec bash "$SCRIPT_DIR/mizar_official_eval/submit_5090.sh" mmar "$@"
    ;;
  base)
    DEFAULT_RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_5090/formal_20261026_193830617147305
    DEFAULT_ROUTE_ROOT="$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_ced_small"
    EVAL_SCRIPT=scripts/evaluate_mizar_s1_ced_benchmark.sh
    JOB_NAME_PREFIX=mizar-s1-ced-mmar
    JOB_LOG_PREFIX=mizar_s1_ced_small_mmar_5090
    ;;
  recursive_mesh)
    DEFAULT_RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_5090/formal_20261027_081123150934256
    DEFAULT_ROUTE_ROOT="$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small"
    EVAL_SCRIPT=scripts/evaluate_mizar_s1_ced_recursive_mesh_benchmark.sh
    JOB_NAME_PREFIX=mizar-s1-ced-mesh-mmar
    JOB_LOG_PREFIX=mizar_s1_ced_small_recursive_mesh_mmar_5090
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
    JOB_NAME_PREFIX="mizar-ced-r515-${JOB_TAG}-mmar"
    JOB_LOG_PREFIX="mizar_s1_ced_small_${EVAL_VARIANT}_mmar_5090"
    ;;
  recursive_5_10_5_layer36912_crossattn|recursive_5_10_5_layer36912_query32_crossattn|recursive_mesh_layer4_8_inject|recursive_mesh_layer4_8_listen|recursive_mesh_layer36912_listen|recursive_mesh_layer4_8_audio7slot_split3|recursive_mesh_layer4_8_audio7slot_split3_inject|recursive_mesh_layer8_4_inject|recursive_mesh_layer4_8_audio7slot_split3_prompt_mix|recursive_5_10_5_layer8_4_inject_audio_reinject|recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8|recursive_mesh_5_10x4_5)
    RUN_STAMP=
    case "$EVAL_VARIANT" in
      recursive_5_10_5_layer36912_crossattn) FAMILY=recursive_5_10_5; ROUTE_SUFFIX=layer36912_crossattn; RUN_STAMP=20261030_140459065784533; JOB_TAG=ca4 ;;
      recursive_5_10_5_layer36912_query32_crossattn) FAMILY=recursive_5_10_5; ROUTE_SUFFIX=layer36912_query32_crossattn; RUN_STAMP=20261030_162028139130855; JOB_TAG=q32 ;;
      recursive_mesh_layer4_8_inject) FAMILY=recursive_mesh; ROUTE_SUFFIX=layer4_8_inject; JOB_TAG=m48i ;;
      recursive_mesh_layer4_8_listen) FAMILY=recursive_mesh; ROUTE_SUFFIX=layer4_8_listen; RUN_STAMP=20261030_172450546025072; JOB_TAG=m48l ;;
      recursive_mesh_layer36912_listen) FAMILY=recursive_mesh; ROUTE_SUFFIX=layer36912_listen; RUN_STAMP=20261101_113133327944502; JOB_TAG=m4l ;;
      recursive_mesh_layer4_8_audio7slot_split3) FAMILY=recursive_mesh; ROUTE_SUFFIX=layer4_8_audio7slot_split3; RUN_STAMP=20261101_170211458358358; JOB_TAG=m7s3 ;;
      recursive_mesh_layer4_8_audio7slot_split3_inject) FAMILY=recursive_mesh; ROUTE_SUFFIX=layer4_8_audio7slot_split3_inject; RUN_STAMP=20261102_002211827036990; JOB_TAG=m7s3i ;;
      recursive_mesh_layer8_4_inject) FAMILY=recursive_mesh; ROUTE_SUFFIX=layer8_4_inject; RUN_STAMP=20261102_112040402999384; JOB_TAG=m84i ;;
      recursive_mesh_layer4_8_audio7slot_split3_prompt_mix) FAMILY=recursive_mesh; ROUTE_SUFFIX=layer4_8_audio7slot_split3_prompt_mix; RUN_STAMP=20261102_132232677352730; JOB_TAG=m7pm ;;
      recursive_5_10_5_layer8_4_inject_audio_reinject) FAMILY=recursive_5_10_5; ROUTE_SUFFIX=layer8_4_inject_audio_reinject; RUN_STAMP=20261102_112121702891801; JOB_TAG=r84ri ;;
      recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8) FAMILY=recursive_5_10_5; ROUTE_SUFFIX=layer8_4_inject_audio_reinject_loop_lora_r8; RUN_STAMP=20261103_165410438279511; JOB_TAG=r84l8 ;;
      recursive_mesh_5_10x4_5) FAMILY=recursive_mesh; ROUTE_SUFFIX=; RUN_STAMP=20261103_093224843011222; JOB_TAG=mx4; DEFAULT_EXPECTED_STEP=34626; DEFAULT_EXPECTED_GEOMETRY=8,4,4 ;;
    esac
    DEFAULT_RUN_ROOT=
    if [[ -n "$RUN_STAMP" ]]; then
      DEFAULT_RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_${EVAL_VARIANT}_5090/formal_${RUN_STAMP}"
    fi
    DEFAULT_ROUTE_ROOT="$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_${FAMILY}_ced_small_${ROUTE_SUFFIX}"
    if [[ "$EVAL_VARIANT" == recursive_mesh_5_10x4_5 ]]; then
      DEFAULT_RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_recursive_mesh_5_10x4_5_5090/formal_${RUN_STAMP}"
      DEFAULT_ROUTE_ROOT="$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs128_3epochs_recursive_mesh_5_10x4_5_ced_small"
    fi
    EVAL_SCRIPT=scripts/evaluate_mizar_s1_ced_multilayer_benchmark.sh
    JOB_NAME_PREFIX="mizar-ced-multi-${JOB_TAG}-mmar"
    JOB_LOG_PREFIX="mizar_s1_ced_small_${EVAL_VARIANT}_mmar_5090"
    ;;
  *)
    echo "unsupported RSMOL_MIZAR_CED_EVAL_VARIANT: $EVAL_VARIANT (expected base or a supported recursive CED variant)" >&2
    exit 2
    ;;
esac
RUN_ROOT="${RSMOL_MIZAR_CED_RUN_ROOT:-$DEFAULT_RUN_ROOT}"
# Pending training routes need an actual run or an explicit checkpoint/runtime/output triple.
# Inspect path overrides before submission, while still forwarding every CLI argument unchanged.
if [[ "$EVAL_SCRIPT" == scripts/evaluate_mizar_s1_ced_multilayer_benchmark.sh ]]; then
  RUN_ROOT="${RSMOL_MIZAR_CED_RUN_ROOT:-}"
  EXTRA_ARGS=("$@")
  for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
    case "${EXTRA_ARGS[i]}" in
      --run-root|--checkpoint|--runtime-config|--output-dir|--expected-step|--expected-geometry)
        ((i+1 < ${#EXTRA_ARGS[@]})) || { echo "missing value for ${EXTRA_ARGS[i]}" >&2; exit 2; }
        OPTION="${EXTRA_ARGS[i]}"
        VALUE="${EXTRA_ARGS[i+1]}"
        i=$((i+1))
        ;;
      --run-root=*|--checkpoint=*|--runtime-config=*|--output-dir=*|--expected-step=*|--expected-geometry=*)
        OPTION="${EXTRA_ARGS[i]%%=*}"
        VALUE="${EXTRA_ARGS[i]#*=}"
        ;;
      *) continue ;;
    esac
    [[ -n "$VALUE" && "$VALUE" != --* ]] || { echo "missing value for $OPTION" >&2; exit 2; }
    case "$OPTION" in
      --run-root) RUN_ROOT="$VALUE" ;;
      --checkpoint) RSMOL_MIZAR_CED_CHECKPOINT="$VALUE" ;;
      --runtime-config) RSMOL_MIZAR_CED_RUNTIME_CONFIG="$VALUE" ;;
      --output-dir) RSMOL_MIZAR_CED_MMAR_OUTPUT_DIR="$VALUE" ;;
      --expected-step) RSMOL_MIZAR_CED_EXPECTED_STEP="$VALUE" ;;
      --expected-geometry) RSMOL_MIZAR_CED_EXPECTED_GEOMETRY="$VALUE" ;;
    esac
  done
  if [[ -z "$RUN_ROOT" && "${RSMOL_MIZAR_CED_CHECKPOINT:-}" == */formal/checkpoints/* ]]; then
    RUN_ROOT="${RSMOL_MIZAR_CED_CHECKPOINT%/formal/checkpoints/*}"
  fi
  RUN_ROOT="${RUN_ROOT:-$DEFAULT_RUN_ROOT}"
  if [[ -z "$RUN_ROOT" ]] && { [[ -z "${RSMOL_MIZAR_CED_CHECKPOINT:-}" ]] || [[ -z "${RSMOL_MIZAR_CED_RUNTIME_CONFIG:-}" ]] || [[ -z "${RSMOL_MIZAR_CED_MMAR_OUTPUT_DIR:-}" ]]; }; then
    echo "No completed default run for $EVAL_VARIANT; set RSMOL_MIZAR_CED_RUN_ROOT / --run-root or provide checkpoint, runtime-config and output-dir explicitly." >&2
    exit 2
  fi
fi
EXPECTED_STEP="${RSMOL_MIZAR_CED_EXPECTED_STEP:-$DEFAULT_EXPECTED_STEP}"
EXPECTED_GEOMETRY="${RSMOL_MIZAR_CED_EXPECTED_GEOMETRY:-$DEFAULT_EXPECTED_GEOMETRY}"
CHECKPOINT="${RSMOL_MIZAR_CED_CHECKPOINT:-$RUN_ROOT/formal/checkpoints/model--step-$EXPECTED_STEP.ckpt}"
RUNTIME_CONFIG="${RSMOL_MIZAR_CED_RUNTIME_CONFIG:-$RUN_ROOT/formal/runtime.json}"
ROUTE_ROOT="${RSMOL_MIZAR_CED_ROUTE_ROOT:-$DEFAULT_ROUTE_ROOT}"
DATASET_DIR="${RSMOL_MMAR_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR}"
OUTPUT_DIR="${RSMOL_MIZAR_CED_MMAR_OUTPUT_DIR:-$RUN_ROOT/eval/mmar_step$EXPECTED_STEP}"
JOB_LOG="$SCRIPT_DIR/log/${JOB_LOG_PREFIX}.${RUN_TAG}.JOB.log"

ARGS=(--benchmark mmar --checkpoint "$CHECKPOINT" --runtime-config "$RUNTIME_CONFIG"
      --route-root "$ROUTE_ROOT" --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_DIR"
      --expected-step "$EXPECTED_STEP" --expected-geometry "$EXPECTED_GEOMETRY")
if [[ "$EVAL_VARIANT" == recursive_5_10_5* || "$EVAL_SCRIPT" == scripts/evaluate_mizar_s1_ced_multilayer_benchmark.sh ]]; then ARGS+=(--eval-variant "$EVAL_VARIANT"); fi
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

echo "Submitted full CED-Small MMAR evaluation; output=$OUTPUT_DIR"
echo "Evaluation variant=$EVAL_VARIANT; checkpoint=$CHECKPOINT; runtime=$RUNTIME_CONFIG"
