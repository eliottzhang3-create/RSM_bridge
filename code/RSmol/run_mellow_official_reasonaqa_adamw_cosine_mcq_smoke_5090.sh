#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"; mkdir -p log
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
MCQ_MANIFEST="${MELLOW_REASONAQA_MCQ_MANIFEST:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5090/mcq_manifest/reasonaqa_mcq_train.json}"
INIT_CKPT="${MELLOW_REASONAQA_MCQ_INIT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5090/formal_30epochs_20261001_003838/checkpoints/mellow_adamw_cosine_reasonaqa_formal_20_20260930_163845508069938/model--epo-30.ckpt}"
OUTPUT_DIR="${MELLOW_REASONAQA_MCQ_SMOKE_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5090/mcq_smoke_$RUN_TAG}"
printf -v CMD_ARGS '%q ' "$MCQ_MANIFEST" "$INIT_CKPT" "$OUTPUT_DIR"
vc submit -p pdgpu-5090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 -c 32 -m 256G -g 8 -n 1 -j mellow-mcq-smoke-5090-$RUN_TAG -d "$SCRIPT_DIR" JOB=1:1 "$SCRIPT_DIR/log/mellow_mq_smoke.$RUN_TAG.JOB.log" --cmd "bash mellow_official_training_c8204d8_adamw_cosine/scripts/rsmol/run_reasonaqa_mcq_8gpu_smoke.sh $CMD_ARGS"
