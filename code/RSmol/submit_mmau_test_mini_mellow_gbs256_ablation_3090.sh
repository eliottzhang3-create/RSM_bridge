#!/usr/bin/env bash

# Submit gbs256 ReasonAQA checkpoints for the same MMAU test-mini evaluation.
# With no checkpoint arguments this keeps the original six-checkpoint batch;
# custom checkpoint paths can be supplied after the optional mode.
# Usage: $0 [full|smoke] [checkpoint-file ...]
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SINGLE_SUBMIT="$SCRIPT_DIR/submit_mmau_test_mini_mellow_gbs256_3090.sh"
MODE="${1:-full}"

if [[ "$MODE" == full || "$MODE" == smoke ]]; then
  shift
else
  MODE=full
fi
if [[ "$MODE" != full && "$MODE" != smoke ]]; then
  echo "mode must be full or smoke: $MODE" >&2
  exit 2
fi
if [[ ! -f "$SINGLE_SUBMIT" ]]; then
  echo "single-checkpoint submitter not found: $SINGLE_SUBMIT" >&2
  exit 2
fi

CHECKPOINTS=(
  "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_old_microbatch_mean_3090/formal_5epochs_maxlr_1e-3_20261009_234938223175399/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_oldmean_formal_20_20261009_213956781718666/model--epo-5.ckpt"
  "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5090/formal_5epochs_maxlr_1e-3_20261009_151832898359917/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr_formal_20_20261009_074133259050800/model--epo-5.ckpt"
  "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5090/formal_5epochs_maxlr_2e-3_20261009_234954643835601/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr_formal_20_20261009_233443558075953/model--epo-5.ckpt"
  "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5090/formal_5epochs_maxlr_2e-4_20261010_210025620938234/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr_formal_20_20261010_154021644553389/model--epo-5.ckpt"
  "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5090/formal_5epochs_maxlr_4e-4_20261010_210023713300705/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr_formal_20_20261010_132413418875990/model--epo-5.ckpt"
  "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5090/formal_5epochs_maxlr_8e-4_20261010_210000727826895/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr_formal_20_20261010_150900245347077/model--epo-5.ckpt"
)
if (( $# > 0 )); then
  CHECKPOINTS=("$@")
fi

submitted=0
failed=0
for checkpoint in "${CHECKPOINTS[@]}"; do
  checkpoint_name="${checkpoint##*/}"
  checkpoint_name="${checkpoint_name%.ckpt}"
  echo "[gbs256-ablation] submitting $checkpoint_name ($MODE)" >&2
  # Empty runtime/output arguments intentionally select the single wrapper's
  # per-checkpoint defaults. The fourth argument carries the mode.
  if bash "$SINGLE_SUBMIT" "$checkpoint" "" "" "$MODE"; then
    submitted=$((submitted + 1))
  else
    rc=$?
    failed=$((failed + 1))
    echo "[gbs256-ablation] submission failed (exit $rc): $checkpoint" >&2
  fi
done

echo "[gbs256-ablation] submitted=$submitted failed=$failed mode=$MODE" >&2
if (( failed != 0 )); then
  exit 1
fi

