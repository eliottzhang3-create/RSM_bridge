#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage: bash run_mizar_mmsu_mesh48_lora_r8_5090.sh [OPTIONS]

Submit full 5000-row MMSU evaluation of the fixed MeSH 4->8 injection +
Loop-LoRA r8 checkpoint: formal_20261104_161727489909082, step17313.
Uses the existing MMAU/MMAR strict adapter and MMSU answer-text scoring v2.

OPTIONS:
  --check-args                     Parse the fixed target; do not submit
  --mmsu-root PATH                 Parent of "MMSU data" and "MMSU code"
  --dataset-dir PATH               HF directory containing data/*.parquet
  --evaluation-script PATH         Pinned official mmsu_evaluation.py
  --output-dir PATH                Separate MMSU output directory
  --parquet-batch-size N           I/O batch size only; inference stays batch1

Checkpoint, variant, runtime, training route, step and geometry are fixed.
USAGE
}

EXTRA_ARGS=("$@")
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --check-args) shift ;;
    --mmsu-root|--dataset-dir|--evaluation-script|--output-dir|--parquet-batch-size)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || {
        echo "Missing value for $1" >&2; exit 2;
      }
      shift 2 ;;
    --mmsu-root=*|--dataset-dir=*|--evaluation-script=*|--output-dir=*|--parquet-batch-size=*)
      [[ -n "${1#*=}" ]] || { echo "Missing value for ${1%%=*}" >&2; exit 2; }
      shift ;;
    *) echo "Unsupported preset option: $1; model identity is fixed." >&2; exit 2 ;;
  esac
done

RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_loop_lora_r8_5090/formal_20261104_161727489909082
ROUTE_ROOT="$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8"

exec bash "$SCRIPT_DIR/run_mizar_s1_ced_small_mmsu_eval_5090.sh" \
  --eval-variant recursive_mesh_layer4_8_inject_loop_lora_r8 \
  --checkpoint "$RUN_ROOT/formal/checkpoints/model--step-17313.ckpt" \
  --runtime-config "$RUN_ROOT/formal/runtime.json" \
  --route-root "$ROUTE_ROOT" \
  --expected-step 17313 --expected-geometry 8,8,4 \
  --output-dir "$RUN_ROOT/eval/mmsu_step17313" \
  "${EXTRA_ARGS[@]}"
