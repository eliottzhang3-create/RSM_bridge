#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(dirname "$HERE")"
cd "$SCRIPT_DIR"
BENCHMARK="${1:?expected mmau or mmar}"
shift
MODEL_DIR="${RSMOL_MIZAR_OFFICIAL_MODEL_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M}"
OUTPUT_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_official_stage3_q4_seed20260905_matched_ced_v1
case "$BENCHMARK" in
  mmau)
    DATASET_DIR="${RSMOL_MMAU_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini}"
    OUTPUT_DIR="${RSMOL_MIZAR_OFFICIAL_MMAU_OUTPUT_DIR:-$OUTPUT_ROOT/mmau_test_mini}"
    ;;
  mmar)
    DATASET_DIR="${RSMOL_MMAR_DATASET_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR}"
    OUTPUT_DIR="${RSMOL_MIZAR_OFFICIAL_MMAR_OUTPUT_DIR:-$OUTPUT_ROOT/mmar}"
    ;;
  *) echo "unsupported benchmark: $BENCHMARK" >&2; exit 2 ;;
esac
ARGS=(--benchmark "$BENCHMARK" --model-dir "$MODEL_DIR"
      --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_DIR" --parquet-batch-size 8)
# Python derives the checkpoint from the effective --model-dir, after all CLI overrides.
if [[ -n "${RSMOL_MIZAR_OFFICIAL_CHECKPOINT:-}" ]]; then
  ARGS+=(--checkpoint "$RSMOL_MIZAR_OFFICIAL_CHECKPOINT")
fi
ARGS+=("$@")
printf -v CMD_ARGS '%q ' "${ARGS[@]}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
mkdir -p log
JOB_LOG="$SCRIPT_DIR/log/mizar_official_${BENCHMARK}_5090.${RUN_TAG}.JOB.log"
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "mizar-official-${BENCHMARK}-${RUN_TAG}" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$JOB_LOG" \
  --cmd "bash mizar_official_eval/run.sh $CMD_ARGS"

echo "Submitted official Mizar full $BENCHMARK evaluation with the existing CED protocol."
echo "Defaults: model=$MODEL_DIR; output=$OUTPUT_DIR (explicit CLI options take precedence)."
