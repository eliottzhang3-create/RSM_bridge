#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
AUDIT_REPORT="${MELLOW_REASONAQA_AUDIT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/path_audit.json}"
MAPPING_JSONL="${MELLOW_REASONAQA_MAPPING_JSONL:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/train_audio_mapping.jsonl}"
RESUME_CHECKPOINT="${MELLOW_REASONAQA_RESUME_CHECKPOINT:-}"
OUTPUT_DIR="${MELLOW_REASONAQA_ADAMW_RESUME_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5090/resume_smoke_$RUN_TAG}"
SMOKE_ROWS="${MELLOW_REASONAQA_SMOKE_ROWS:-2048}"

if [[ $# -gt 5 ]]; then
  echo "usage: $0 [audit-report] [mapping-jsonl] [resume-checkpoint] [output-dir] [smoke-rows]" >&2
  exit 2
fi
[[ $# -ge 1 ]] && AUDIT_REPORT="$1"
[[ $# -ge 2 ]] && MAPPING_JSONL="$2"
[[ $# -ge 3 ]] && RESUME_CHECKPOINT="$3"
[[ $# -ge 4 ]] && OUTPUT_DIR="$4"
[[ $# -ge 5 ]] && SMOKE_ROWS="$5"
if [[ -z "$RESUME_CHECKPOINT" ]]; then
  echo "resume checkpoint is required" >&2
  exit 2
fi

printf -v CMD_ARGS '%q ' "$AUDIT_REPORT" "$MAPPING_JSONL" "$RESUME_CHECKPOINT" "$OUTPUT_DIR" "$SMOKE_ROWS"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j mellow-official-reasonaqa-adamw-cosine-resume-5090-$RUN_TAG \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mellow_official_reasonaqa_adamw_cosine_resume_5090.$RUN_TAG.JOB.log" \
  --cmd "bash mellow_official_training_c8204d8_adamw_cosine/scripts/rsmol/run_reasonaqa_8gpu_resume_smoke.sh $CMD_ARGS"

echo "Submitted 8-GPU AdamW/cosine Mellow resume smoke; output directory: $OUTPUT_DIR"
