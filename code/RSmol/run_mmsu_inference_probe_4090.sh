#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REVISION=""
FORWARDED=()
while (($#)); do
  case "$1" in
    --source-revision)
      [[ $# -ge 2 && -n "$2" ]] || { echo '--source-revision needs a Git revision' >&2; exit 2; }
      SOURCE_REVISION="$2"; shift 2 ;;
    *) FORWARDED+=("$1"); shift ;;
  esac
done
if [[ -n "$SOURCE_REVISION" ]]; then
  # Shared storage is required: login-node /tmp cannot be read by GPU workers.
  mkdir -p "$SCRIPT_DIR/log/mmsu_inference_probe_sources"
  SOURCE_DIR="$(mktemp -d "$SCRIPT_DIR/log/mmsu_inference_probe_sources/source_XXXXXXXX")"
  REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
  git -C "$REPO_ROOT" archive --format=tar "$SOURCE_REVISION" code/RSmol |
    tar -x -C "$SOURCE_DIR"
  FORWARDED+=(--implementation-root "$SOURCE_DIR/code/RSmol")
  printf 'Historical source snapshot: %s (Git %s)\n' "$SOURCE_DIR" "$SOURCE_REVISION"
fi
for argument in "${FORWARDED[@]}"; do
  case "$argument" in
    --help|-h|--check-args) exec bash "$SCRIPT_DIR/mizar_mmsu_inference_probe/run.sh" "${FORWARDED[@]}" ;;
  esac
done
bash "$SCRIPT_DIR/mizar_mmsu_inference_probe/run.sh" "${FORWARDED[@]}" --check-args
mkdir -p "$SCRIPT_DIR/log"
RUN_TAG="$(date +%m%d%H%M%S)_$$"
printf -v CMD_ARGS '%q ' "${FORWARDED[@]}"
vc submit -p pdgpu-4090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 -j "mmsu-probe-${RUN_TAG}" \
  -d "$SCRIPT_DIR" JOB=1:1 "$SCRIPT_DIR/log/mmsu-probe.${RUN_TAG}.JOB.log" \
  --cmd "bash mizar_mmsu_inference_probe/run.sh $CMD_ARGS"
