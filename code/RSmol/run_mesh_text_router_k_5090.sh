#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mesh_text_router_k_5090
TEXT_MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244
PROMPTS_JSONL=""
OUTPUT_DIR=""
CAP=0.005
K_MARGIN=1.0
SEED=1234
usage() {
  echo "usage: $0 --prompts-jsonl FILE [--text-model-dir DIR] [--output-dir DIR] [--max-new-tokens 100] [--new-slot-mass-cap 0.005] [--k-margin 1.0] [--seed 1234]" >&2
  exit 2
}
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || usage
  case "$1" in
    --prompts-jsonl) PROMPTS_JSONL=$2 ;;
    --text-model-dir) TEXT_MODEL_DIR=$2 ;;
    --output-dir) OUTPUT_DIR=$2 ;;
    --max-new-tokens) [[ "$2" == 100 ]] || { echo "max-new-tokens is fixed at 100" >&2; exit 2; } ;;
    --new-slot-mass-cap) CAP=$2 ;;
    --k-margin) K_MARGIN=$2 ;;
    --seed) SEED=$2 ;;
    *) usage ;;
  esac
  shift 2
done
[[ -n "$PROMPTS_JSONL" ]] || usage
RUN_TAG=$(date +%Y%m%d_%H%M%S%N)
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/run_${RUN_TAG}}
for path in "$PROMPTS_JSONL" "$TEXT_MODEL_DIR" "$OUTPUT_DIR"; do
  case "$path" in /*) ;; *) echo "paths must be absolute: $path" >&2; exit 2 ;; esac
done
ARGS=(--prompts-jsonl "$PROMPTS_JSONL" --text-model-dir "$TEXT_MODEL_DIR"
      --output-dir "$OUTPUT_DIR" --max-new-tokens 100 --new-slot-mass-cap "$CAP"
      --k-margin "$K_MARGIN" --seed "$SEED")
# Login-node standard-library validation; no torch/CUDA, inference or output claim.
python -B "$SCRIPT_DIR/scripts/calibrate_mesh_text_router_k.py" "${ARGS[@]}" --check-inputs
mkdir -p "$ROOT/logs"
printf -v CMD_ARGS '%q ' "${ARGS[@]}"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "mesh-text-router-k-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$ROOT/logs/mesh_text_router_k.${RUN_TAG}.JOB.log" \
  --cmd "bash scripts/calibrate_mesh_text_router_k.sh $CMD_ARGS"
echo "Submitted original five-slot text generation: 10 prompts, max 100 new tokens, FP32, one GPU."
echo "Combined two-new-slot mass cap=$CAP; extra K logit margin=$K_MARGIN; output=$OUTPUT_DIR"
echo "Job log: $ROOT/logs/mesh_text_router_k.${RUN_TAG}.JOB.log"
