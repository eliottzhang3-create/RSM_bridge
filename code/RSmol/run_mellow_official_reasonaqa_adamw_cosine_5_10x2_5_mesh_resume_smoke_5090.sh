#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
AUDIT_REPORT="${MELLOW_REASONAQA_AUDIT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/path_audit.json}"
MAPPING_JSONL="${MELLOW_REASONAQA_MAPPING_JSONL:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/train_audio_mapping.jsonl}"
RESUME_CHECKPOINT="${MELLOW_REASONAQA_MESH_RESUME_CHECKPOINT:-}"
OUTPUT_DIR="${MELLOW_REASONAQA_MESH_RESUME_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5_10x2_5_mesh_5090/resume_smoke_$RUN_TAG}"
SMOKE_ROWS="${MELLOW_REASONAQA_MESH_SMOKE_ROWS:-5120}"

if [[ $# -gt 5 ]]; then
  echo "usage: $0 [audit-report] [mapping-jsonl] [step-20-checkpoint] [output-dir] [smoke-rows=5120]" >&2
  exit 2
fi
[[ $# -ge 1 ]] && AUDIT_REPORT="$1"
[[ $# -ge 2 ]] && MAPPING_JSONL="$2"
[[ $# -ge 3 ]] && RESUME_CHECKPOINT="$3"
[[ $# -ge 4 ]] && OUTPUT_DIR="$4"
[[ $# -ge 5 ]] && SMOKE_ROWS="$5"
if [[ -z "$RESUME_CHECKPOINT" || ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "an audited step-20 checkpoint is required" >&2
  exit 2
fi
[[ "$SMOKE_ROWS" -eq 5120 ]] || { echo "resume smoke rows must be exactly 5120" >&2; exit 2; }

printf -v CMD_ARGS '%q ' "$AUDIT_REPORT" "$MAPPING_JSONL" "$RESUME_CHECKPOINT" "$OUTPUT_DIR" "$SMOKE_ROWS"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j mellow-awc-mesh-resume-5090-$RUN_TAG \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mellow_official_reasonaqa_adamw_cosine_5_10x2_5_mesh_resume_smoke_5090.$RUN_TAG.JOB.log" \
  --cmd "bash mellow_official_training_adamw_cosine_5_10x2_5_mesh/scripts/rsmol/run_reasonaqa_8gpu_resume_smoke.sh $CMD_ARGS"

echo "Submitted 8-GPU MeSH resume smoke (step 20 -> 22); output directory: $OUTPUT_DIR"
