#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage: bash run_mizar_mmsu_selected5_5090.sh TARGET [OPTIONS]

Submit one full 5000-row MMSU evaluation for the targets selected on 2026-11-13.
TARGET:
  official_mizar        Official Mizar-159M Stage3/Q4 final release
  base                  CED-Small base, step17313
  recursive_5_10_5       Plain non-MeSH 5-10-5 recursive, step17313
  serial84_lora_r8       Serial 8->4 injection + audio reinjection + Loop-LoRA r8
  recursive_mesh        Original recursive MeSH route, step17313

OPTIONS:
  --check-args                     Parse the effective target; do not submit
  --mmsu-root PATH                 Parent of "MMSU data" and "MMSU code"
  --dataset-dir PATH               HF directory containing data/*.parquet
  --evaluation-script PATH         Pinned official mmsu_evaluation.py
  --output-dir PATH                Separate output directory for this target
  --parquet-batch-size N           Parquet I/O batch size (inference stays batch1)

Model/checkpoint/runtime/step/geometry are fixed by TARGET. Use the generic
run_mizar_s1_ced_small_mmsu_eval_5090.sh for other targets.
USAGE
}

TARGET="${1:-}"
case "$TARGET" in
  -h|--help) usage; exit 0 ;;
  '') usage >&2; exit 2 ;;
esac
shift

# Forward only I/O options. A preset must keep the selected model identity.
EXTRA_ARGS=("$@")
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --check-args) shift ;;
    --mmsu-root|--dataset-dir|--evaluation-script|--output-dir|--parquet-batch-size)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || {
        echo "Missing value for $1" >&2; exit 2;
      }
      shift 2
      ;;
    --mmsu-root=*|--dataset-dir=*|--evaluation-script=*|--output-dir=*|--parquet-batch-size=*)
      [[ -n "${1#*=}" ]] || { echo "Missing value for ${1%%=*}" >&2; exit 2; }
      shift
      ;;
    *) echo "Unsupported preset option: $1; model identity is fixed by TARGET." >&2; exit 2 ;;
  esac
done

OUTPUT_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol
TRAINING_ROOT="$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine"
case "$TARGET" in
  official_mizar)
    MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M
    ARGS=(--eval-variant official_mizar
          --model-dir "$MODEL_DIR"
          --checkpoint "$MODEL_DIR/checkpoints/final/stage3_q4_seed20260905.ckpt"
          --output-dir "$OUTPUT_ROOT/mizar_official_stage3_q4_seed20260905_matched_ced_v1/mmsu")
    ;;
  base)
    VARIANT=base
    RUN_ROOT="$OUTPUT_ROOT/mizar_s1_ced_small_gbs256_5090/formal_20261026_193830617147305"
    ROUTE_ROOT="$TRAINING_ROOT/mizar_stage1_fixed20_gbs256_3epochs_ced_small"
    ;;
  recursive_5_10_5)
    VARIANT=recursive_5_10_5
    RUN_ROOT="$OUTPUT_ROOT/mizar_s1_ced_small_gbs256_recursive_5_10_5_5090/formal_20261028_160823559722647"
    ROUTE_ROOT="$TRAINING_ROOT/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small"
    ;;
  serial84_lora_r8)
    VARIANT=recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8
    RUN_ROOT="$OUTPUT_ROOT/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8_5090/formal_20261103_165410438279511"
    ROUTE_ROOT="$TRAINING_ROOT/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8"
    ;;
  recursive_mesh)
    VARIANT=recursive_mesh
    RUN_ROOT="$OUTPUT_ROOT/mizar_s1_ced_small_gbs256_recursive_mesh_5090/formal_20261027_081123150934256"
    ROUTE_ROOT="$TRAINING_ROOT/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small"
    ;;
  *) echo "Unknown MMSU target: $TARGET" >&2; usage >&2; exit 2 ;;
esac

if [[ "$TARGET" != official_mizar ]]; then
  ARGS=(--eval-variant "$VARIANT"
        --checkpoint "$RUN_ROOT/formal/checkpoints/model--step-17313.ckpt"
        --runtime-config "$RUN_ROOT/formal/runtime.json"
        --route-root "$ROUTE_ROOT"
        --expected-step 17313 --expected-geometry 8,8,4
        --output-dir "$RUN_ROOT/eval/mmsu_step17313")
fi

exec bash "$SCRIPT_DIR/run_mizar_s1_ced_small_mmsu_eval_5090.sh" "${ARGS[@]}" "${EXTRA_ARGS[@]}"
