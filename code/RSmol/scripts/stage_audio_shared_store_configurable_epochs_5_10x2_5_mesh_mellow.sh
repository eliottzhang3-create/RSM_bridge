#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <smoke|formal> [trainer arguments...]" >&2
  exit 2
fi
MODE="$1"
shift
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "invalid configurable shared-store training mode: $MODE" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate rsmol

SOURCE_STORE="${RSMOL_SHARED_STORE_SOURCE:-/hpc_stor03/sjtu_home/jinwei.zhang/data/rsmol_reasonaqa_train_unique_waveforms_32k_10s_f32_v3}"
SOURCE_MANIFEST="${RSMOL_SHARED_MANIFEST_SOURCE:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl}"
RUN_ID="${RSMOL_SHARED_STORE_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}"
STAGED_STORE="/dev/shm/rsmol_shared_train_${RUN_ID}"
STAGED_MANIFEST="$STAGED_STORE/reasonaqa_train.jsonl"

if [[ ! -d "$SOURCE_STORE" || ! -f "$SOURCE_STORE/metadata.json" || ! -f "$SOURCE_STORE/index.jsonl" || ! -f "$SOURCE_STORE/waveforms.f32" ]]; then
  echo "persistent unique store is incomplete: $SOURCE_STORE" >&2
  exit 2
fi
if [[ -e "$SOURCE_STORE/BUILDING" ]]; then
  echo "persistent unique store is still BUILDING: $SOURCE_STORE" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_MANIFEST" ]]; then
  echo "persistent manifest is missing: $SOURCE_MANIFEST" >&2
  exit 2
fi

python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); m=json.loads((p/"metadata.json").read_text()); assert m.get("status")=="PASS", m; assert m.get("format")=="manifest_unique_fixed_waveform_store_v1", m' "$SOURCE_STORE"

STORE_KIB=$(du -sk "$SOURCE_STORE" | awk '{print $1}')
MANIFEST_KIB=$((($(stat -c '%s' "$SOURCE_MANIFEST") + 1023) / 1024))
SHM_FREE_KIB=$(df -Pk /dev/shm | awk 'NR == 2 {print $4}')
SHM_MARGIN_KIB=$((10 * 1024 * 1024))
REQUIRED_KIB=$((STORE_KIB + MANIFEST_KIB + SHM_MARGIN_KIB))
if [[ -z "$SHM_FREE_KIB" || "$SHM_FREE_KIB" -lt "$REQUIRED_KIB" ]]; then
  echo "insufficient /dev/shm space: free_kib=${SHM_FREE_KIB:-unknown} required_kib=$REQUIRED_KIB" >&2
  exit 2
fi
if [[ -e "$STAGED_STORE" ]]; then
  echo "refusing existing staging directory: $STAGED_STORE" >&2
  exit 2
fi

cleanup_shared_store() {
  if [[ -n "${STAGED_STORE:-}" && "$STAGED_STORE" == /dev/shm/rsmol_shared_train_* && -d "$STAGED_STORE" ]]; then
    rm -rf -- "$STAGED_STORE"
  fi
}
trap cleanup_shared_store EXIT INT TERM

TOTAL_START_NS=$(date +%s%N)
mkdir "$STAGED_STORE"
STORE_START_NS=$(date +%s%N)
echo "[shared-store-stage] copying $SOURCE_STORE to $STAGED_STORE" >&2
cp -a "$SOURCE_STORE"/. "$STAGED_STORE"/
STORE_END_NS=$(date +%s%N)
MANIFEST_START_NS=$(date +%s%N)
cp "$SOURCE_MANIFEST" "$STAGED_MANIFEST"
MANIFEST_END_NS=$(date +%s%N)

SOURCE_WAVEFORM_BYTES=$(stat -c '%s' "$SOURCE_STORE/waveforms.f32")
STAGED_WAVEFORM_BYTES=$(stat -c '%s' "$STAGED_STORE/waveforms.f32")
if [[ "$SOURCE_WAVEFORM_BYTES" -ne "$STAGED_WAVEFORM_BYTES" ]]; then
  echo "staged waveform byte-size mismatch: source=$SOURCE_WAVEFORM_BYTES staged=$STAGED_WAVEFORM_BYTES" >&2
  exit 2
fi
SOURCE_MANIFEST_SHA=$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')
STAGED_MANIFEST_SHA=$(sha256sum "$STAGED_MANIFEST" | awk '{print $1}')
SOURCE_INDEX_SHA=$(sha256sum "$SOURCE_STORE/index.jsonl" | awk '{print $1}')
STAGED_INDEX_SHA=$(sha256sum "$STAGED_STORE/index.jsonl" | awk '{print $1}')
if [[ "$SOURCE_MANIFEST_SHA" != "$STAGED_MANIFEST_SHA" || "$SOURCE_INDEX_SHA" != "$STAGED_INDEX_SHA" ]]; then
  echo "staged manifest/index SHA256 mismatch" >&2
  exit 2
fi

STORE_COPY_SECONDS=$(awk -v start="$STORE_START_NS" -v end="$STORE_END_NS" 'BEGIN {printf "%.9f", (end-start)/1000000000}')
MANIFEST_COPY_SECONDS=$(awk -v start="$MANIFEST_START_NS" -v end="$MANIFEST_END_NS" 'BEGIN {printf "%.9f", (end-start)/1000000000}')
STAGING_END_NS=$(date +%s%N)
STAGING_TOTAL_SECONDS=$(awk -v start="$TOTAL_START_NS" -v end="$STAGING_END_NS" 'BEGIN {printf "%.9f", (end-start)/1000000000}')
echo "[shared-store-stage] PASS store_seconds=$STORE_COPY_SECONDS manifest_seconds=$MANIFEST_COPY_SECONDS total_seconds=$STAGING_TOTAL_SECONDS" >&2

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
torchrun --standalone --nproc_per_node=8 "$SCRIPT_DIR/train_audio_shared_store_configurable_epochs_5_10x2_5_mesh_mellow_ddp.py" \
  --mode "$MODE" \
  --train-manifest "$STAGED_MANIFEST" \
  --unique-waveform-store-dir "$STAGED_STORE" \
  --persistent-manifest-source "$SOURCE_MANIFEST" \
  --persistent-store-source "$SOURCE_STORE" \
  --store-copy-seconds "$STORE_COPY_SECONDS" \
  --manifest-copy-seconds "$MANIFEST_COPY_SECONDS" \
  --staging-total-seconds "$STAGING_TOTAL_SECONDS" \
  "$@"
