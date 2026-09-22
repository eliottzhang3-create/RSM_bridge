#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate rsmol

# PERF20 is intentionally isolated from STAGE5/STAGE7/FORMAL.  The default
# output name is unique per launch; the Python gate refuses to reuse it.
# The shared trainer uses torch.autocast(device_type="cuda", dtype=torch.bfloat16).
DEFAULT_MESH="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244"
DEFAULT_HTSAT="/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt"
DEFAULT_MELLOW="/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main"
DEFAULT_MANIFEST="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl"
DEFAULT_SHARED_WAVEFORM_STORE="/hpc_stor03/sjtu_home/jinwei.zhang/data/rsmol_reasonaqa_train_unique_waveforms_32k_10s_f32_v3"
DEFAULT_COMPONENT_PARTITION_STORE_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/data/rsmol_reasonaqa_train_component_partitions6_32k_10s_f32_v2"
PERF20_RUN_ID="${PERF20_RUN_ID:-$(date +%Y%m%d_%H%M%S%N)-$$}"
PERF20_INPUT_MODE="online"
PERF20_PARTITION_ID="0"
SHARED_STORE_SOURCE="$DEFAULT_SHARED_WAVEFORM_STORE"
PERF20_ARGS=("$@")
for ((argument_index=0; argument_index<${#PERF20_ARGS[@]}; argument_index++)); do
  argument="${PERF20_ARGS[$argument_index]}"
  case "$argument" in
    --preload-data)
      PERF20_INPUT_MODE="full_preload"
      ;;
    --perf20-input-mode=*)
      PERF20_INPUT_MODE="${argument#*=}"
      ;;
    --perf20-input-mode)
      argument_index=$((argument_index + 1))
      PERF20_INPUT_MODE="${PERF20_ARGS[$argument_index]:-}"
      ;;
    --perf20-partition-id=*)
      PERF20_PARTITION_ID="${argument#*=}"
      ;;
    --perf20-partition-id)
      argument_index=$((argument_index + 1))
      PERF20_PARTITION_ID="${PERF20_ARGS[$argument_index]:-}"
      ;;
    --shared-waveform-store-dir=*)
      SHARED_STORE_SOURCE="${argument#*=}"
      ;;
    --shared-waveform-store-dir)
      argument_index=$((argument_index + 1))
      SHARED_STORE_SOURCE="${PERF20_ARGS[$argument_index]:-}"
      ;;
  esac
done
case "$PERF20_INPUT_MODE" in
  online|warm_online|waveform_preload|full_preload|shared_waveform_store|store_rank_ram_preload|store_rank_ram_prefetch)
    PERF20_OUTPUT_PREFIX="perf20_${PERF20_INPUT_MODE}"
    ;;
  shared_waveform_store_tmpfs)
    PERF20_OUTPUT_PREFIX="perf20_${PERF20_INPUT_MODE}"
    ;;
  partition_rank_ram_preload)
    if [[ ! "$PERF20_PARTITION_ID" =~ ^[0-9]+$ ]]; then
      echo "invalid PERF20 partition ID: $PERF20_PARTITION_ID" >&2
      exit 2
    fi
    PERF20_OUTPUT_PREFIX="perf20_${PERF20_INPUT_MODE}_p${PERF20_PARTITION_ID}"
    ;;
  *)
    echo "invalid PERF20 input mode: $PERF20_INPUT_MODE" >&2
    exit 2
    ;;
esac
DEFAULT_OUTPUT_DIR="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/${PERF20_OUTPUT_PREFIX}_${PERF20_RUN_ID}"

# The tmpfs control copies the complete v3 store into this node's shared RAM
# before torchrun.  It is deliberately opt-in: ordinary shared_waveform_store
# continues to measure the persistent mmap store and is not silently changed.
SHARED_STORE_ARG=("$SHARED_STORE_SOURCE")
SHARED_STORE_OVERRIDE_ARGS=()
SHM_STORE_DIR=""
if [[ "$PERF20_INPUT_MODE" == "shared_waveform_store_tmpfs" ]]; then
  if [[ ! -d "$SHARED_STORE_SOURCE" || ! -f "$SHARED_STORE_SOURCE/metadata.json" || ! -f "$SHARED_STORE_SOURCE/index.jsonl" || ! -f "$SHARED_STORE_SOURCE/waveforms.f32" ]]; then
    echo "shared waveform store is incomplete: $SHARED_STORE_SOURCE" >&2
    exit 2
  fi
  if [[ -e "$SHARED_STORE_SOURCE/BUILDING" ]]; then
    echo "shared waveform store is still being built: $SHARED_STORE_SOURCE" >&2
    exit 2
  fi
  if ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"PASS"' "$SHARED_STORE_SOURCE/metadata.json"; then
    echo "shared waveform store metadata is not PASS: $SHARED_STORE_SOURCE/metadata.json" >&2
    exit 2
  fi
  STORE_BYTES=$(stat -c '%s' "$SHARED_STORE_SOURCE/waveforms.f32")
  SHM_FREE_KIB=$(df -Pk /dev/shm | awk 'NR == 2 {print $4}')
  SHM_MARGIN_KIB=$((5 * 1024 * 1024))
  STORE_REQUIRED_KIB=$(((STORE_BYTES + 1023) / 1024 + SHM_MARGIN_KIB))
  if [[ -z "$SHM_FREE_KIB" || "$SHM_FREE_KIB" -lt "$STORE_REQUIRED_KIB" ]]; then
    echo "insufficient /dev/shm space: free_kib=${SHM_FREE_KIB:-unknown} required_kib=$STORE_REQUIRED_KIB" >&2
    exit 2
  fi
  SHM_STORE_DIR="/dev/shm/rsmol_perf20_${PERF20_RUN_ID}"
  mkdir "$SHM_STORE_DIR"
  trap 'if [[ -n "${SHM_STORE_DIR:-}" && "$SHM_STORE_DIR" == /dev/shm/rsmol_perf20_* ]]; then rm -rf -- "$SHM_STORE_DIR"; fi' EXIT
  echo "[audio-perf20] staging full unique waveform store into $SHM_STORE_DIR" >&2
  cp -a "$SHARED_STORE_SOURCE"/. "$SHM_STORE_DIR"/
  [[ -f "$SHM_STORE_DIR/metadata.json" && -f "$SHM_STORE_DIR/index.jsonl" && -f "$SHM_STORE_DIR/waveforms.f32" ]]
  STAGED_STORE_BYTES=$(stat -c '%s' "$SHM_STORE_DIR/waveforms.f32")
  if [[ "$STAGED_STORE_BYTES" -ne "$STORE_BYTES" ]]; then
    echo "staged waveform store size mismatch: source=$STORE_BYTES staged=$STAGED_STORE_BYTES" >&2
    exit 2
  fi
  SHARED_STORE_ARG=("$SHM_STORE_DIR")
  SHARED_STORE_OVERRIDE_ARGS=(--shared-waveform-store-dir "$SHM_STORE_DIR")
fi

PYTHON_ARGS=("${PERF20_ARGS[@]}")

# Keep the baseline explicit, but do not pass both sides of the mutually
# exclusive argparse profiler group when the outer submission wrapper adds
# ``--profiler``.  The caller's explicit profiler flag always wins.
PROFILER_FLAG_SEEN=0
for argument in "$@"; do
  case "$argument" in
    --profiler|--enable-profiler|--no-profiler|--disable-profiler)
      PROFILER_FLAG_SEEN=1
      ;;
  esac
done
PROFILER_DEFAULT=()
if [[ "$PROFILER_FLAG_SEEN" -eq 0 ]]; then
  PROFILER_DEFAULT=(--no-profiler)
fi

torchrun --standalone --nproc_per_node=8 "$SCRIPT_DIR/train_audio_5_10x2_5_mesh_mellow_ddp.py" \
  --gate PERF20 \
  --model-path "$DEFAULT_MESH" \
  --htsat-checkpoint "$DEFAULT_HTSAT" \
  --mellow-root "$DEFAULT_MELLOW" \
  --train-manifest "$DEFAULT_MANIFEST" \
  --shared-waveform-store-dir "${SHARED_STORE_ARG[0]}" \
  --perf20-partition-store-root "$DEFAULT_COMPONENT_PARTITION_STORE_ROOT" \
  --output-dir "$DEFAULT_OUTPUT_DIR" \
  --world-size 8 \
  --micro-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --num-workers 0 \
  --epochs 1 \
  --max-steps 20 \
  --max-lr 1e-3 \
  --min-lr 0 \
  --seed 0 \
  --steady-state-start-step 6 \
  "${PROFILER_DEFAULT[@]}" \
  "${PYTHON_ARGS[@]}" \
  "${SHARED_STORE_OVERRIDE_ARGS[@]}"
