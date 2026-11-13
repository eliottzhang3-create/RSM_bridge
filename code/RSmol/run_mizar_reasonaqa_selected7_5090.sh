#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() {
  cat <<'USAGE'
Usage: bash run_mizar_reasonaqa_selected7_5090.sh TARGET [OPTIONS]
TARGET: base | recursive_5_10_5 | serial84 | serial48_lora_r8 |
        serial84_lora_r8 | recursive_mesh | mesh48_lora_r8 | all
OPTIONS: --test-json PATH, --audiocaps-root PATH, --clotho-root PATH,
         --clotho-aqa-root PATH, --output-dir PATH, --check-args
Each target fixes the exact user-selected step17313 checkpoint and 8,8,4 geometry.
all submits seven independent full evaluations; --output-dir is single-target only.
--check-args parses without submission, weights, data reads or inference.
USAGE
}
TARGET="${1:-}"
case "$TARGET" in
  -h|--help) usage; exit 0 ;;
  base|recursive_5_10_5|serial84|serial48_lora_r8|serial84_lora_r8|recursive_mesh|mesh48_lora_r8|all) ;;
  *) usage >&2; exit 2 ;;
esac
shift
EXTRA_ARGS=("$@")
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --check-args) shift ;;
    --test-json|--audiocaps-root|--clotho-root|--clotho-aqa-root|--output-dir)
      [[ "$TARGET" != all || "$1" != --output-dir ]] || { echo 'all requires separate default outputs' >&2; exit 2; }
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
      shift 2 ;;
    --test-json=*|--audiocaps-root=*|--clotho-root=*|--clotho-aqa-root=*|--output-dir=*)
      [[ "$TARGET" != all || "${1%%=*}" != --output-dir ]] || { echo 'all requires separate default outputs' >&2; exit 2; }
      [[ -n "${1#*=}" ]] || { echo "Missing value for ${1%%=*}" >&2; exit 2; }
      shift ;;
    *) echo "Unsupported preset option: $1; target identity is fixed" >&2; exit 2 ;;
  esac
done
if [[ "$TARGET" == all ]]; then
  for selected in base recursive_5_10_5 serial84 serial48_lora_r8 serial84_lora_r8 recursive_mesh mesh48_lora_r8; do
    bash "$SCRIPT_DIR/run_mizar_s1_ced_small_reasonaqa_eval_5090.sh" --target "$selected" "${EXTRA_ARGS[@]}"
  done
else
  exec bash "$SCRIPT_DIR/run_mizar_s1_ced_small_reasonaqa_eval_5090.sh" --target "$TARGET" "${EXTRA_ARGS[@]}"
fi
