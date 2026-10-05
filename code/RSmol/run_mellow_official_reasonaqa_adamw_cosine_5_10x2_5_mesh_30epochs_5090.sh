#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
AUDIT_REPORT="${MELLOW_REASONAQA_AUDIT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/path_audit.json}"
MAPPING_JSONL="${MELLOW_REASONAQA_MAPPING_JSONL:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/train_audio_mapping.jsonl}"
OUTPUT_DIR="${MELLOW_REASONAQA_MESH_FORMAL_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5_10x2_5_mesh_5090/formal_30epochs_$RUN_TAG}"

if [[ $# -gt 3 ]]; then
  echo "usage: $0 [audit-report] [mapping-jsonl] [output-dir]" >&2
  exit 2
fi
[[ $# -ge 1 ]] && AUDIT_REPORT="$1"
[[ $# -ge 2 ]] && MAPPING_JSONL="$2"
[[ $# -ge 3 ]] && OUTPUT_DIR="$3"

printf -v CMD_ARGS '%q ' "$AUDIT_REPORT" "$MAPPING_JSONL" "$OUTPUT_DIR"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j mellow-awc-mesh-formal-5090-$RUN_TAG \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mellow_official_reasonaqa_adamw_cosine_5_10x2_5_mesh_30epochs_5090.$RUN_TAG.JOB.log" \
  --cmd "bash mellow_official_training_adamw_cosine_5_10x2_5_mesh/scripts/rsmol/run_reasonaqa_8gpu_formal.sh $CMD_ARGS"

echo "Submitted 8-GPU 30-epoch MeSH formal training; output directory: $OUTPUT_DIR"
