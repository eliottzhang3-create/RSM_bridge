#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage: bash run_mizar_selected4_benchmarks_5090.sh TARGET BENCHMARK [OPTIONS]

TARGET (fixed step17313 checkpoint, runtime and training route):
  init0904                         Plain recursive, 0904 text initialization
  text_epoch1                      Plain recursive, text epoch1 initialization
  text_epoch1_serial48_lora_r8      CED4->8 + reinjection, fresh LoRA at audio step0
  text_lora_epoch1_serial48_r8      CED4->8 + reinjection, inherited text LoRA
BENCHMARK: mmau | mmar | mmsu | all
  all submits three independent full jobs and attempts every submission.

OPTIONS:
  --check-args                     Print shell-quoted commands; do not submit
  --parquet-batch-size N           I/O batch size only; inference stays batch1
  --dataset-dir PATH               Single benchmark only
  --evaluation-script PATH         Single benchmark only, official scorer
  --output-dir PATH                Single benchmark only, isolated output
  --metadata-json PATH             MMAU/MMAR only
  --audio-root PATH                MMAU/MMAR only
  --parquet PATH                   MMAU only
  --mmsu-root PATH                 MMSU only (parent of MMSU data / MMSU code)

The all mode accepts only --check-args and --parquet-batch-size. Run benchmarks
separately for path overrides. --check-args displays bindings without reading
weights/data or validating their existence. Model identity cannot be overridden.
USAGE
}

TARGET="${1:-}"
case "$TARGET" in -h|--help) usage; exit 0 ;; '') usage >&2; exit 2 ;; esac
BENCHMARK="${2:-}"
case "$BENCHMARK" in mmau|mmar|mmsu|all) ;; *) usage >&2; exit 2 ;; esac
shift 2

OUTPUT_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol
TRAINING_ROOT="$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine"
case "$TARGET" in
  init0904)
    VARIANT=recursive_5_10_5_init0904
    RUN_ROOT="$OUTPUT_ROOT/mizar_s1_ced_small_gbs256_recursive_5_10_5_init0904_5090/formal_20261111_130037935014521"
    ROUTE_ROOT="$TRAINING_ROOT/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_init0904"
    ;;
  text_epoch1)
    VARIANT=recursive_5_10_5_text_epoch1
    RUN_ROOT="$OUTPUT_ROOT/mizar_s1_ced_small_text_epoch1_gbs256_5090/formal_20261112_103104688616376"
    ROUTE_ROOT="$TRAINING_ROOT/mizar_s1_ced_small_text_epoch1"
    ;;
  text_epoch1_serial48_lora_r8)
    VARIANT=recursive_5_10_5_text_epoch1_layer4_8_audio_reinject_loop_lora_r8
    RUN_ROOT="$OUTPUT_ROOT/mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_gbs256_5090/formal_20261112_103717262469368"
    ROUTE_ROOT="$TRAINING_ROOT/mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8"
    ;;
  text_lora_epoch1_serial48_r8)
    VARIANT=recursive_5_10_5_ced48_text_epoch1_loop_lora_r8
    RUN_ROOT="$OUTPUT_ROOT/mizar_s1_ced48_text_epoch1_loop_lora_r8_5090/formal_20261112_111357380584730"
    ROUTE_ROOT="$TRAINING_ROOT/mizar_s1_ced48_text_epoch1_loop_lora_r8"
    ;;
  *) echo "Unknown TARGET: $TARGET" >&2; exit 2 ;;
esac

CHECK_ARGS=0
EXTRA_ARGS=()
ALL_ARGS=("$@")
while (($#)); do
  OPTION="${1%%=*}"
  case "$OPTION" in
    -h|--help) usage; exit 0 ;;
    --check-args)
      [[ "$1" == --check-args ]] || { echo "--check-args takes no value" >&2; exit 2; }
      CHECK_ARGS=1; shift; continue ;;
    --parquet-batch-size) ;;
    --dataset-dir|--evaluation-script|--output-dir)
      [[ "$BENCHMARK" != all ]] || { echo "$OPTION requires one benchmark" >&2; exit 2; } ;;
    --metadata-json|--audio-root)
      [[ "$BENCHMARK" == mmau || "$BENCHMARK" == mmar ]] || { echo "$OPTION requires MMAU/MMAR" >&2; exit 2; } ;;
    --parquet)
      [[ "$BENCHMARK" == mmau ]] || { echo "$OPTION requires MMAU" >&2; exit 2; } ;;
    --mmsu-root)
      [[ "$BENCHMARK" == mmsu ]] || { echo "$OPTION requires MMSU" >&2; exit 2; } ;;
    *) echo "Unsupported preset option: $1; model identity is fixed by TARGET." >&2; exit 2 ;;
  esac
  if [[ "$1" == *=* ]]; then
    VALUE="${1#*=}"
    shift
  else
    (($# >= 2)) || { echo "Missing value for $OPTION" >&2; exit 2; }
    VALUE="$2"
    shift 2
  fi
  [[ -n "$VALUE" && "$VALUE" != --* ]] || { echo "Missing value for $OPTION" >&2; exit 2; }
  if [[ "$OPTION" == --parquet-batch-size && ! "$VALUE" =~ ^[1-9][0-9]*$ ]]; then
    echo "--parquet-batch-size must be a positive integer" >&2; exit 2
  fi
  EXTRA_ARGS+=("$OPTION" "$VALUE")
done

if [[ "$BENCHMARK" == all ]]; then
  STATUS=0
  for BENCH in mmau mmar mmsu; do
    if bash "$SCRIPT_DIR/run_mizar_selected4_benchmarks_5090.sh" "$TARGET" "$BENCH" "${ALL_ARGS[@]}"; then
      :
    else
      echo "Submission failed: target=$TARGET benchmark=$BENCH; continuing other benchmarks." >&2
      STATUS=1
    fi
  done
  exit "$STATUS"
fi

CHECKPOINT="$RUN_ROOT/formal/checkpoints/model--step-17313.ckpt"
RUNTIME_CONFIG="$RUN_ROOT/formal/runtime.json"
OUTPUT_DIR="$RUN_ROOT/eval/${BENCHMARK}_step17313"
if [[ "$BENCHMARK" == mmau ]]; then OUTPUT_DIR="$RUN_ROOT/eval/mmau_test_mini_step17313"; fi

if [[ "$BENCHMARK" == mmsu ]]; then
  COMMAND=(bash "$SCRIPT_DIR/run_mizar_s1_ced_small_mmsu_eval_5090.sh"
           --eval-variant "$VARIANT" --checkpoint "$CHECKPOINT"
           --runtime-config "$RUNTIME_CONFIG" --route-root "$ROUTE_ROOT"
           --expected-step 17313 --expected-geometry 8,8,4 --output-dir "$OUTPUT_DIR"
           "${EXTRA_ARGS[@]}")
else
  # Bind every model field explicitly, including over stale exported settings.
  COMMAND=(env "RSMOL_MIZAR_CED_EVAL_VARIANT=$VARIANT"
           "RSMOL_MIZAR_CED_RUN_ROOT=$RUN_ROOT" "RSMOL_MIZAR_CED_ROUTE_ROOT=$ROUTE_ROOT"
           "RSMOL_MIZAR_CED_CHECKPOINT=$CHECKPOINT" "RSMOL_MIZAR_CED_RUNTIME_CONFIG=$RUNTIME_CONFIG"
           RSMOL_MIZAR_CED_EXPECTED_STEP=17313 RSMOL_MIZAR_CED_EXPECTED_GEOMETRY=8,8,4
           "RSMOL_MIZAR_CED_MMAU_OUTPUT_DIR=$OUTPUT_DIR" "RSMOL_MIZAR_CED_MMAR_OUTPUT_DIR=$OUTPUT_DIR"
           bash "$SCRIPT_DIR/run_mizar_s1_ced_small_${BENCHMARK}_eval_5090.sh"
           "${EXTRA_ARGS[@]}")
fi
if ((CHECK_ARGS)); then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
exec "${COMMAND[@]}"
