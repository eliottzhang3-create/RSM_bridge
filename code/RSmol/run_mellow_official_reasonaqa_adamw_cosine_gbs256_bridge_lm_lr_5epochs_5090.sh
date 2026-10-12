#!/usr/bin/env bash
set -euo pipefail

usage(){ echo "usage: $0 --bridge-max-lr VALUE --lm-max-lr VALUE [--audit-report PATH] [--mapping-jsonl PATH] [--output-dir PATH] [--resume-checkpoint PATH]" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_REPORT="${MELLOW_REASONAQA_AUDIT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/path_audit.json}"
MAPPING_JSONL="${MELLOW_REASONAQA_MAPPING_JSONL:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/train_audio_mapping.jsonl}"
OUTPUT_DIR=""; RESUME_CHECKPOINT=""; BRIDGE_MAX_LR=""; LM_MAX_LR=""
while [[ $# -gt 0 ]]; do case "$1" in --bridge-max-lr) [[ $# -ge 2 ]] || usage; BRIDGE_MAX_LR=$2; shift 2;; --lm-max-lr) [[ $# -ge 2 ]] || usage; LM_MAX_LR=$2; shift 2;; --audit-report) [[ $# -ge 2 ]] || usage; AUDIT_REPORT=$2; shift 2;; --mapping-jsonl) [[ $# -ge 2 ]] || usage; MAPPING_JSONL=$2; shift 2;; --output-dir) [[ $# -ge 2 ]] || usage; OUTPUT_DIR=$2; shift 2;; --resume-checkpoint) [[ $# -ge 2 ]] || usage; RESUME_CHECKPOINT=$2; shift 2;; *) usage;; esac; done
[[ -n "$BRIDGE_MAX_LR" && -n "$LM_MAX_LR" ]] || usage
TAG=$(python3 - "$BRIDGE_MAX_LR" "$LM_MAX_LR" <<'PY'
import math,sys
for value in sys.argv[1:]:
 parsed=float(value)
 if not math.isfinite(parsed) or parsed<=0: raise SystemExit("both max LRs must be positive and finite")
print(f"bridge{float(sys.argv[1]):.0e}_lm{float(sys.argv[2]):.0e}")
PY
)
RUN_TAG=$(date +%Y%m%d_%H%M%S%N)
OUTPUT_DIR="${OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_bridge_lm_lr_5090/formal_5epochs_${TAG}_${RUN_TAG}}"
[[ -f "$AUDIT_REPORT" && -f "$MAPPING_JSONL" ]] || { echo "audit report or mapping is missing" >&2; exit 2; }
[[ -z "$RESUME_CHECKPOINT" || -f "$RESUME_CHECKPOINT" ]] || { echo "resume checkpoint not found" >&2; exit 2; }
if [[ -e "$OUTPUT_DIR" && ! -d "$OUTPUT_DIR" ]]; then echo "output path is not a directory: $OUTPUT_DIR" >&2; exit 2; fi
if [[ -z "$RESUME_CHECKPOINT" && -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then echo "fresh run requires an empty output directory: $OUTPUT_DIR" >&2; exit 2; fi
mkdir -p "$SCRIPT_DIR/log"
args=("$AUDIT_REPORT" "$MAPPING_JSONL" "$OUTPUT_DIR" "$BRIDGE_MAX_LR" "$LM_MAX_LR"); [[ -z "$RESUME_CHECKPOINT" ]] || args+=("$RESUME_CHECKPOINT")
printf -v cmd_args '%q ' "${args[@]}"
vc submit -p pdgpu-5090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 -c 32 -m 256G -g 8 -n 1 -j "mbl-${RUN_TAG}" -d "$SCRIPT_DIR" JOB=1:1 "$SCRIPT_DIR/log/mellow_bridge_lm_${TAG}.${RUN_TAG}.JOB.log" --cmd "bash mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch256_bridge_lm_lr_5epochs/scripts/rsmol/run_reasonaqa_8gpu_formal.sh $cmd_args"
echo "Submitted grouped-LR training: bridge=$BRIDGE_MAX_LR, lm=$LM_MAX_LR; output=$OUTPUT_DIR"
