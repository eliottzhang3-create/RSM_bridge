#!/usr/bin/env bash
set -euo pipefail
[[ $# == 7 ]] || { echo "run via code/RSmol/run_mizar_s1_5090.sh" >&2; exit 2; }
MODE=$1
RESOLVED_PATHS=$(readlink -f "$2")
OUTPUT_ROOT=$(readlink -m "$3")
SMOKE_REPORT=$4
RESUME_CHECKPOINT=$5
TEXT_MODEL_DIR=$(readlink -f "$6")
HTSAT_ROOT=$(readlink -f "$7")
ROUTE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_filtered.json
[[ "$MODE" == smoke || "$MODE" == formal || "$MODE" == smoke-then-formal ]] || exit 2
CONDA_BASE=${MELLOW_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mellow_c8204d8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cd "$ROUTE_ROOT"
mkdir -p "$OUTPUT_ROOT"
RUN_ID="${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S%N)"
STAGE_ROOT=""
STAGE_TOKEN=""
cleanup() {
  if [[ -n "$STAGE_ROOT" ]]; then
    python "$ROUTE_ROOT/stage_data.py" --cleanup --stage-root "$STAGE_ROOT" --token "$STAGE_TOKEN"
  fi
}
trap cleanup EXIT
# These eight checks passed remotely on 2026-10-21 (torch 2.11.0+cu128).
# Keep them available for manual diagnostics without rerunning at each launch.
# python runtime_checks.py

run_one() {
  local run_mode=$1
  local run_dir="$OUTPUT_ROOT/$run_mode"
  mkdir -p "$run_dir"
  if [[ -z "$RESUME_CHECKPOINT" && -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "fresh $run_mode directory is not empty: $run_dir" >&2
    return 2
  fi
  if [[ "$run_mode" == formal ]]; then
    python - "$SMOKE_REPORT" <<'PY'
import json
import sys
from pathlib import Path
from contract import code_digest
from write_config import validate_smoke
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
validate_smoke(report, report["contract_sha256"])
if report.get("code_sha256") != code_digest():
    raise SystemExit("code changed since smoke; rerun the eight-update smoke on this code")
PY
  fi
  STAGE_TOKEN="${run_mode}_${RUN_ID}"
  STAGE_ROOT="/dev/shm/mizar_s1_${STAGE_TOKEN}"
  python stage_data.py --manifest "$MANIFEST" --resolved-paths "$RESOLVED_PATHS" \
    --stage-root "$STAGE_ROOT" --token "$STAGE_TOKEN" --mode "$run_mode" --output "$run_dir/staging_report.json"
  local config_args=(--prepared "$run_dir/staging_report.json" --output "$run_dir/runtime.json" \
    --run-dir "$run_dir" --mode "$run_mode" --text-model-dir "$TEXT_MODEL_DIR" --htsat-root "$HTSAT_ROOT")
  if [[ "$run_mode" == formal ]]; then
    config_args+=(--smoke-report "$SMOKE_REPORT")
  fi
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    config_args+=(--resume-checkpoint "$RESUME_CHECKPOINT")
  fi
  python write_config.py "${config_args[@]}"
  torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --config "$run_dir/runtime.json"
  python - "$run_dir/training_report.json" "$run_mode" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = 8 if sys.argv[2] == "smoke" else 122937
if report.get("status") != "PASS" or report.get("optimizer_steps") != expected:
    raise SystemExit("training report did not confirm all required optimizer updates")
print(f"[{sys.argv[2]}] PASS: {expected} optimizer updates; report={sys.argv[1]}")
PY
  cleanup
  STAGE_ROOT=""
  STAGE_TOKEN=""
}

if [[ "$MODE" == smoke-then-formal ]]; then
  [[ -z "$RESUME_CHECKPOINT" ]] || { echo "pipeline always starts fresh" >&2; exit 2; }
  run_one smoke
  SMOKE_REPORT="$OUTPUT_ROOT/smoke/training_report.json"
  run_one formal
else
  run_one "$MODE"
fi
