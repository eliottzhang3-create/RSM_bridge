#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RSMOL_MIZAR_RUN_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_5090/formal_async_v3/formal}"
CHECKPOINT="${RSMOL_MIZAR_CHECKPOINT:-$RUN_ROOT/checkpoints/model--step-122937.ckpt}"
RUNTIME_CONFIG="${RSMOL_MIZAR_RUNTIME_CONFIG:-$RUN_ROOT/runtime.json}"
ROUTE_ROOT="${RSMOL_MIZAR_ROUTE_ROOT:-$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_3epochs}"
DATASET_DIR="${RSMOL_MMAR_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR}"
OUTPUT_DIR="${RSMOL_MMAR_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_5090/formal_async_v3/eval/mmar_step122937}"
JOB_LOG="$SCRIPT_DIR/log/mizar_s1_mmar_5090.${RUN_TAG}.JOB.log"

ARGS=(--benchmark mmar --checkpoint "$CHECKPOINT" --runtime-config "$RUNTIME_CONFIG"
      --route-root "$ROUTE_ROOT" --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_DIR")
if (($#)); then ARGS+=("$@"); fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 1 -n 1 \
  -j "mizar-s1-mmar-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/evaluate_mizar_s1_benchmark.sh $CMD_ARGS"

echo "Submitted full MMAR evaluation; output=$OUTPUT_DIR"
