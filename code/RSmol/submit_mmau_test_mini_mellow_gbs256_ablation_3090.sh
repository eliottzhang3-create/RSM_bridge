#!/usr/bin/env bash

# Submit LR10 gbs256 ReasonAQA checkpoints for the same MMAU test-mini evaluation.
# With no checkpoint arguments this evaluates the two documented LR10 runs;
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
  "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090/formal_10epochs_lr5_frozen_clip_maxlr_2e-5_20261018_181826682119766/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr10_frozen_clip_20_20261018_101831978638346/model--epo-10.ckpt"
  "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090/formal_10epochs_lr5_frozen_clip_maxlr_3e-5_20261018_181649446900432/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr10_frozen_clip_20_20261018_101654725998277/model--epo-10.ckpt"
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

