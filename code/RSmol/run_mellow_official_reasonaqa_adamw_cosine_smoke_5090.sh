#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
AUDIT_REPORT="${MELLOW_REASONAQA_AUDIT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/path_audit.json}"
MAPPING_JSONL="${MELLOW_REASONAQA_MAPPING_JSONL:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/train_audio_mapping.jsonl}"
OUTPUT_DIR="${MELLOW_REASONAQA_ADAMW_SMOKE_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5090/smoke_$RUN_TAG}"
SMOKE_ROWS="${MELLOW_REASONAQA_SMOKE_ROWS:-2048}"

if [[ $# -gt 4 ]]; then
  echo "usage: $0 [audit-report] [mapping-jsonl] [output-dir] [smoke-rows]" >&2
  exit 2
fi
if [[ $# -ge 1 ]]; then AUDIT_REPORT="$1"; fi
if [[ $# -ge 2 ]]; then MAPPING_JSONL="$2"; fi
if [[ $# -ge 3 ]]; then OUTPUT_DIR="$3"; fi
if [[ $# -ge 4 ]]; then SMOKE_ROWS="$4"; fi

printf -v CMD_ARGS '%q ' "$AUDIT_REPORT" "$MAPPING_JSONL" "$OUTPUT_DIR" "$SMOKE_ROWS"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j mellow-official-reasonaqa-adamw-cosine-smoke-5090-$RUN_TAG \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mellow_official_reasonaqa_adamw_cosine_smoke_5090.$RUN_TAG.JOB.log" \
  --cmd "bash mellow_official_training_c8204d8_adamw_cosine/scripts/rsmol/run_reasonaqa_8gpu_smoke.sh $CMD_ARGS"

echo "Submitted 8-GPU AdamW/cosine Mellow smoke; output directory: $OUTPUT_DIR"
