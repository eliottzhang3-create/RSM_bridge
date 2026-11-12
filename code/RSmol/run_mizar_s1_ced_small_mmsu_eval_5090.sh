#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# Validate explicit targets before submission. This only parses CPU arguments.
for argument in "$@"; do
  case "$argument" in
    --help|-h|--check-args)
      exec bash "$SCRIPT_DIR/mizar_s1_ced_mmsu_eval/run.sh" "$@"
      ;;
  esac
done
bash "$SCRIPT_DIR/mizar_s1_ced_mmsu_eval/run.sh" --check-args "$@"
mkdir -p log
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
printf -v CMD_ARGS '%q ' "$@"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "mizar-ced-mmsu-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mizar_s1_ced_small_mmsu_5090.${RUN_TAG}.JOB.log" \
  --cmd "bash mizar_s1_ced_mmsu_eval/run.sh $CMD_ARGS"
echo "Submitted full MMSU evaluation; explicit target arguments forwarded unchanged."
