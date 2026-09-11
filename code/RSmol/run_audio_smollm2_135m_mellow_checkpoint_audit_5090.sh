#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
CMD_ARGS=""
if (($#)); then
  printf -v CMD_ARGS '%q ' "$@"
fi
vc submit -p pdgpu-5090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 16 -m 128G -g 1 -n 1 \
  -j audio-smollm2-baseline-audit-5090-$(date +%m%d%H%M%S) \
  -d "$SCRIPT_DIR" JOB=1:1 "$SCRIPT_DIR/log/audio_smollm2_baseline_audit_5090.JOB.log" \
  --cmd "bash scripts/audit_audio_smollm2_135m_mellow_checkpoint.sh $CMD_ARGS"
