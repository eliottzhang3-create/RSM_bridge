#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
AUDIT_REPORT="${MELLOW_REASONAQA_AUDIT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/path_audit.json}"
MAPPING_JSONL="${MELLOW_REASONAQA_MAPPING_JSONL:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/train_audio_mapping.jsonl}"
OUTPUT_DIR="${MELLOW_REASONAQA_GBS32_FORMAL_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs32_5090/formal_5epochs_$RUN_TAG}"
RESUME_CHECKPOINT="${MELLOW_REASONAQA_RESUME_CHECKPOINT:-}"

if [[ $# -gt 4 ]]; then
  echo "usage: $0 [audit-report] [mapping-jsonl] [output-dir] [resume-checkpoint]" >&2
  exit 2
fi
[[ $# -ge 1 ]] && AUDIT_REPORT="$1"
[[ $# -ge 2 ]] && MAPPING_JSONL="$2"
[[ $# -ge 3 ]] && OUTPUT_DIR="$3"
[[ $# -ge 4 ]] && RESUME_CHECKPOINT="$4"
if [[ -n "$RESUME_CHECKPOINT" && ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "resume checkpoint not found: $RESUME_CHECKPOINT" >&2
  exit 2
fi

INNER_ARGS=("$AUDIT_REPORT" "$MAPPING_JSONL" "$OUTPUT_DIR" 5)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  INNER_ARGS+=("$RESUME_CHECKPOINT")
fi
printf -v CMD_ARGS '%q ' "${INNER_ARGS[@]}"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j mellow-awc-gbs32-5090-$RUN_TAG \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mellow_official_reasonaqa_gbs32_formal_5090.$RUN_TAG.JOB.log" \
  --cmd "bash mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch32_10epochs/scripts/rsmol/run_reasonaqa_8gpu_formal.sh $CMD_ARGS"

echo "Submitted 8-GPU 5-epoch full ReasonAQA AdamW/cosine Mellow training (global batch 32); output directory: $OUTPUT_DIR"
