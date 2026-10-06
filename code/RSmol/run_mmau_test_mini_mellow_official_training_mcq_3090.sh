#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

CHECKPOINT="${RSMOL_MELLOW_MCQ_CHECKPOINT_FILE:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_3090/mcq_formal_10epochs_20261006_013046/checkpoints/mellow_adamw_cosine_reasonaqa_mcq_formal_10epochs_20_20261005_220825904758999_6766/model--epo-10.ckpt}"
RUNTIME_CONFIG="${RSMOL_MELLOW_MCQ_RUNTIME_CONFIG:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_3090/mcq_formal_10epochs_20261006_013046/runtime_mcq_10epochs.yaml}"
ROUTE_ROOT="${RSMOL_MELLOW_MCQ_ROUTE_ROOT:-$SCRIPT_DIR/mellow_official_training_c8204d8_adamw_cosine}"
MODE="${RSMOL_MELLOW_MCQ_EVAL_MODE:-smoke}"
OUTPUT_DIR="${RSMOL_MELLOW_MCQ_MMAU_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_3090/mcq_formal_10epochs_20261006_013046/eval/mmau_test_mini_official_training_v1}"

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "RSMOL_MELLOW_MCQ_EVAL_MODE must be smoke or full: $MODE" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "MCQ checkpoint does not exist: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$RUNTIME_CONFIG" ]]; then
  echo "MCQ runtime config does not exist: $RUNTIME_CONFIG" >&2
  exit 2
fi
if [[ ! -d "$ROUTE_ROOT" ]]; then
  echo "official Mellow route root does not exist: $ROUTE_ROOT" >&2
  exit 2
fi

RSMOL_MELLOW_TRAINING_ROUTE=adamw_cosine \
RSMOL_MELLOW_EVAL_MODE="$MODE" \
RSMOL_MELLOW_CHECKPOINT_FILE="$CHECKPOINT" \
RSMOL_MELLOW_RUNTIME_CONFIG="$RUNTIME_CONFIG" \
RSMOL_MELLOW_ROUTE_ROOT="$ROUTE_ROOT" \
RSMOL_MELLOW_EVAL_OUTPUT_DIR="$OUTPUT_DIR" \
  bash "$SCRIPT_DIR/run_mmau_test_mini_mellow_official_training_3090.sh" "$@"
