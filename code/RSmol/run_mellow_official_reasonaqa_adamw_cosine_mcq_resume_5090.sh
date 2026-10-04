#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"; mkdir -p log; RUN_TAG="$(date +%Y%m%d_%H%M%S)"
if [[ $# -ne 1 ]]; then echo "usage: $0 <smoke-full-checkpoint.ckpt>" >&2; exit 2; fi
MCQ_MANIFEST="${MELLOW_REASONAQA_MCQ_MANIFEST:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5090/mcq_manifest/reasonaqa_mcq_train.json}"; OUTPUT_DIR="${MELLOW_REASONAQA_MCQ_RESUME_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5090/mcq_resume_$RUN_TAG}"; AUDIT_REPORT="${MELLOW_REASONAQA_AUDIT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/path_audit.json}"; MAPPING_JSONL="${MELLOW_REASONAQA_MAPPING_JSONL:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_c8204d8/preflight/path_audit_20260929_190423/train_audio_mapping.jsonl}"
printf -v CMD_ARGS '%q ' "$MCQ_MANIFEST" "$1" "$OUTPUT_DIR" "$AUDIT_REPORT" "$MAPPING_JSONL"
vc submit -p pdgpu-5090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 -c 32 -m 256G -g 8 -n 1 -j mellow-mcq-resume-5090-$RUN_TAG -d "$SCRIPT_DIR" JOB=1:1 "$SCRIPT_DIR/log/mellow_mq_resume.$RUN_TAG.JOB.log" --cmd "bash mellow_official_training_c8204d8_adamw_cosine/scripts/rsmol/run_reasonaqa_mcq_8gpu_resume.sh $CMD_ARGS"
