#!/usr/bin/env bash
set -euo pipefail
usage() { echo "usage: $0 [--audit-report PATH] [--mapping-jsonl PATH] [--output-dir EMPTY_PATH]" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AUDIT_REPORT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/path_audit.json"
MAPPING_JSONL="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/train_audio_mapping.jsonl"
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --audit-report) [[ $# -ge 2 ]] || usage; AUDIT_REPORT=$2; shift 2 ;;
    --mapping-jsonl) [[ $# -ge 2 ]] || usage; MAPPING_JSONL=$2; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || usage; OUTPUT_DIR=$2; shift 2 ;;
    *) usage ;;
  esac
done
RUN_TAG="$(date +%Y%m%d_%H%M%S%N)"
if [[ -z "$OUTPUT_DIR" ]]; then OUTPUT_DIR="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_formal30_smoke8_5090/smoke8_$RUN_TAG"; fi
[[ -f "$AUDIT_REPORT" && -f "$MAPPING_JSONL" ]] || { echo "audit or mapping missing" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" || ( -d "$OUTPUT_DIR" && -z "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ) ]] || { echo "output must be empty" >&2; exit 2; }
mkdir -p "$SCRIPT_DIR/log"
printf -v CMD_ARGS '%q ' "$AUDIT_REPORT" "$MAPPING_JSONL" "$OUTPUT_DIR"
vc submit -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 -j "mellow-smoke8-$RUN_TAG" \
  -d "$SCRIPT_DIR" JOB=1:1 "$SCRIPT_DIR/log/mellow_formal30_smoke8.$RUN_TAG.JOB.log" \
  --cmd "bash mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch256_formal30_smoke8/scripts/rsmol/run_reasonaqa_8gpu_smoke.sh $CMD_ARGS"
echo "Submitted eight-step ReasonAQA smoke: $OUTPUT_DIR"
