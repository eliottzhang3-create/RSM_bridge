#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log
vc submit -p pdgpu-3090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 -c 8 -m 32G -g 1 -n 1 -j "audit-stage1-5-10x2-5-mesh-$(date +%m%d%H%M)" -d "$SCRIPT_DIR" JOB=1:1 "$SCRIPT_DIR/log/audit_stage1_5_10x2_5_mesh.JOB.log" --cmd "bash scripts/audit_stage1_5_10x2_5_mesh.sh"
