#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --max-lr 1e-3 [--audit-report PATH] [--mapping-jsonl PATH] [--output-dir PATH] [--resume-checkpoint PATH]" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_REPORT="${MELLOW_REASONAQA_AUDIT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/path_audit.json}"
MAPPING_JSONL="${MELLOW_REASONAQA_MAPPING_JSONL:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/train_audio_mapping.jsonl}"
OUTPUT_DIR=""
RESUME_CHECKPOINT=""
MAX_LR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-lr)
      [[ $# -ge 2 ]] || usage
      MAX_LR=$2
      shift 2
      ;;
    --audit-report)
      [[ $# -ge 2 ]] || usage
      AUDIT_REPORT=$2
      shift 2
      ;;
    --mapping-jsonl)
      [[ $# -ge 2 ]] || usage
      MAPPING_JSONL=$2
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || usage
      OUTPUT_DIR=$2
      shift 2
      ;;
    --resume-checkpoint)
      [[ $# -ge 2 ]] || usage
      RESUME_CHECKPOINT=$2
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ -n "$MAX_LR" ]] || usage
LR_TAG=$(python3 - "$MAX_LR" <<'PY'
import math
import sys

try:
    max_lr = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f"invalid max LR: {sys.argv[1]!r}") from exc
if not math.isfinite(max_lr) or max_lr <= 0 or max_lr * 0.1 <= 0:
    raise SystemExit("max LR must be positive, finite, and produce a positive min LR")
if not math.isclose(max_lr, 1e-3, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("this isolated loss ablation requires max LR 1e-3")
mantissa, exponent = f"{max_lr:.12e}".split("e")
print(f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}")
PY
)

RUN_TAG="$(date +%Y%m%d_%H%M%S%N)"
OUTPUT_DIR="${OUTPUT_DIR:-${MELLOW_REASONAQA_OLDMEAN_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_old_microbatch_mean_3090/formal_5epochs_maxlr_${LR_TAG}_${RUN_TAG}}}"
[[ -f "$AUDIT_REPORT" && -f "$MAPPING_JSONL" ]] || {
  echo "audit report or mapping is missing" >&2
  exit 2
}
if [[ -n "$RESUME_CHECKPOINT" && ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "resume checkpoint not found: $RESUME_CHECKPOINT" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" && ! -d "$OUTPUT_DIR" ]]; then
  echo "output path is not a directory: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ -z "$RESUME_CHECKPOINT" && -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "fresh run requires an empty output directory: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$SCRIPT_DIR/log"
INNER_ARGS=("$AUDIT_REPORT" "$MAPPING_JSONL" "$OUTPUT_DIR" "$MAX_LR")
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  INNER_ARGS+=("$RESUME_CHECKPOINT")
fi
printf -v CMD_ARGS '%q ' "${INNER_ARGS[@]}"
vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "mellow-oldmean256-1e-3-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/mellow_official_reasonaqa_oldmean256_1e-3.${RUN_TAG}.JOB.log" \
  --cmd "bash mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch256_old_microbatch_mean_5epochs/scripts/rsmol/run_reasonaqa_8gpu_formal.sh $CMD_ARGS"

echo "Submitted 8-GPU full ReasonAQA old-microbatch-mean loss ablation: batch=256, epochs=5, max_lr=$MAX_LR, min_lr=1e-4; output directory: $OUTPUT_DIR"
