#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${RSMOL_MELLOW_V0_MMAU_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/mmau_test_mini_mellow_author_reply_protocol_v1}"
PREFLIGHT_REPORT="${RSMOL_MELLOW_V0_PREFLIGHT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/preflight/mellow_v0_artifact_preflight.json}"
JOB_LOG="$SCRIPT_DIR/log/mmau_mellow_v0_smoke_4090.${RUN_TAG}.JOB.log"

ARGS=(
  --mode smoke
  --preflight-report "$PREFLIGHT_REPORT"
  --output-dir "$OUTPUT_DIR"
  --parquet-batch-size 8
  --max-prompt-tokens 129
  --max-new-tokens 300
  --dtype fp32
)
if (($#)); then
  ARGS+=("$@")
fi
printf -v CMD_ARGS '%q ' "${ARGS[@]}"

vc submit \
  -p pdgpu-4090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "mmau-mellow-v0-smoke-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash scripts/evaluate_mmau_test_mini_mellow_v0.sh $CMD_ARGS"

echo "Mellow-v0 MMAU smoke output directory: ${OUTPUT_DIR}"
