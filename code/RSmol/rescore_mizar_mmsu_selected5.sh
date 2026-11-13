#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() {
  cat <<'USAGE'
Usage: bash rescore_mizar_mmsu_selected5.sh TARGET [OPTIONS]
TARGET: official_mizar | base | recursive_5_10_5 | serial84_lora_r8 | recursive_mesh | all
CPU-only offline scoring of existing 5000-row outputs; no vc, weights or inference.
OPTIONS:
  --evaluation-script PATH   Pinned official MMSU scorer
  --output-dir PATH          Separate scoring directory (one TARGET only)
  --check-args               Print parsed paths only; no reads or scoring
Default output: <existing MMSU output>/rescore_mmsu_choice_v2/
USAGE
}
TARGET="${1:-}"
case "$TARGET" in -h|--help) usage; exit 0 ;; '') usage >&2; exit 2 ;; esac
shift
case "$TARGET" in official_mizar|base|recursive_5_10_5|serial84_lora_r8|recursive_mesh|all) ;;
  *) echo "Unknown TARGET: $TARGET" >&2; exit 2 ;; esac
EXTRA_ARGS=("$@")
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --check-args) shift ;;
    --evaluation-script|--output-dir)
      [[ "$1" != --output-dir || "$TARGET" != all ]] || { echo "--output-dir requires one TARGET" >&2; exit 2; }
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
      shift 2 ;;
    --evaluation-script=*|--output-dir=*)
      [[ "$1" != --output-dir=* || "$TARGET" != all ]] || { echo "--output-dir requires one TARGET" >&2; exit 2; }
      [[ -n "${1#*=}" ]] || { echo "Missing value for ${1%%=*}" >&2; exit 2; }
      shift ;;
    *) echo "Unsupported option: $1" >&2; exit 2 ;;
  esac
done
if [[ "$TARGET" == all ]]; then
  STATUS=0
  for ITEM in official_mizar base recursive_5_10_5 serial84_lora_r8 recursive_mesh; do
    if bash "$SCRIPT_DIR/rescore_mizar_mmsu_selected5.sh" "$ITEM" "${EXTRA_ARGS[@]}"; then :
    else echo "Offline MMSU scoring failed: $ITEM; continuing other targets." >&2; STATUS=1; fi
  done
  exit "$STATUS"
fi
OUTPUT_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol
case "$TARGET" in
  official_mizar) SOURCE="$OUTPUT_ROOT/mizar_official_stage3_q4_seed20260905_matched_ced_v1/mmsu" ;;
  base) SOURCE="$OUTPUT_ROOT/mizar_s1_ced_small_gbs256_5090/formal_20261026_193830617147305/eval/mmsu_step17313" ;;
  recursive_5_10_5) SOURCE="$OUTPUT_ROOT/mizar_s1_ced_small_gbs256_recursive_5_10_5_5090/formal_20261028_160823559722647/eval/mmsu_step17313" ;;
  serial84_lora_r8) SOURCE="$OUTPUT_ROOT/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8_5090/formal_20261103_165410438279511/eval/mmsu_step17313" ;;
  recursive_mesh) SOURCE="$OUTPUT_ROOT/mizar_s1_ced_small_gbs256_recursive_mesh_5090/formal_20261027_081123150934256/eval/mmsu_step17313" ;;
esac
exec bash "$SCRIPT_DIR/mizar_s1_ced_mmsu_eval/rescore.sh" --source-output-dir "$SOURCE" "${EXTRA_ARGS[@]}"
