#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <PASS path-audit report.json> <train-audio mapping.jsonl> [persistent output dir]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p log

AUDIT_REPORT=$(readlink -f "$1")
MAPPING_JSONL=$(readlink -f "$2")
OUTPUT_DIR="${3:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/staging_probe_$(date +%Y%m%d_%H%M%S%N)}"
if [[ ! -f "$AUDIT_REPORT" ]]; then
  echo "PASS path-audit report is missing: $AUDIT_REPORT" >&2
  exit 2
fi
if [[ ! -f "$MAPPING_JSONL" ]]; then
  echo "train-audio mapping is missing: $MAPPING_JSONL" >&2
  exit 2
fi

printf -v CMD_ARGS '%q ' "$AUDIT_REPORT" "$MAPPING_JSONL" "$OUTPUT_DIR"
JOB_TAG="mellow-raw-stage-probe-3090-$(date +%m%d%H%M%S%N)"

# The probe is CPU/I/O work, but it uses the normal submitted compute-node
# path. One GPU is reserved so the allocation matches the later 3090 route;
# this staging-only job starts neither a CUDA model nor torchrun.
vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 16 -m 128G -g 1 -n 1 \
  -j "$JOB_TAG" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/${JOB_TAG}.JOB.log" \
  --cmd "bash mellow_official_training_c8204d8/scripts/rsmol/run_reasonaqa_staging_probe.sh $CMD_ARGS"
