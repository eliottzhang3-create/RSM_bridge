# Mellow-faithful SmolLM2 shared-store route

This directory and its configurable-epoch entrypoints are an isolated reproduction route. The legacy fixed-260 SmolLM2 shared-store package and all MeSH routes remain unchanged.

## Fixed contract

- Official Mellow training reference: commit c8204d8eb99b4384fd7a76ad57995731e0c0c2bf.
- Standard 30-layer SmolLM2-135M, frozen HTSAT backbone, trainable c2l and Mellow projection, and fully trainable text model.
- Missing audio slots sample uniformly from sorted unique non-empty filepath1 values. Self-selection is allowed.
- The Stage-1 manifest stores an originally empty filepath2 as a duplicate audio2_path for legacy routes. This route restores the missing-slot meaning from filepath2_raw, audio2_reused, and audio2_source before applying Mellow's random-audio2 process.
- Both audio slots are cropped independently. Clips longer than 320000 samples use an inclusive random start; shorter clips are right-zero-padded.
- Fixed layout: audio1 129 + separator 1 + audio2 129 + separator 1 + prompt 129 + answer 250 = 639 tokens.
- Adam, LR 1e-3, weight decay 1e-4, gradient clipping 0.5, FP32, no warmup, epoch-level CosineAnnealingLR with T_max 30.
- Qualification smoke/reference/resume geometry: 8 GPUs, microbatch 4 per rank, gradient accumulation 1, effective global batch 32, num_workers 0.
- Formal geometry: 8 GPUs, microbatch 8 per rank, gradient accumulation 4, effective global batch 256, num_workers 0. This restores the previous production batching geometry while retaining the Mellow-faithful data, model, FP32, optimizer, and epoch-level scheduler contracts.
- Thirty epochs. Save every 5000 optimizer steps and at the final step, retaining only the newest four complete checkpoints.
- Dataset randomness is stateful Python random. Checkpoints include every rank's Python, Torch, and CUDA RNG state.
- Resume leaves Adam's non-capturable scalar step tensors on CPU, matching a continuous run. The DataLoader uses a private generator for its iterator seed so reconstruction at the resume cursor does not advance the training RNG stream.
- Sampler order is torch.randperm with seed equal to the epoch, followed by contiguous rank slices. Resume starts directly at the saved optimizer-step cursor and does not replay prior batches.

## Text-contract audit and training behavior

The failed formal run on 2026-09-27 reached step 9970 and then encountered a
Stage-1 row whose normalized input or answer field was empty. The original
record is retained under metadata, but the previous dataset implementation
only consulted the normalized top-level fields when that row was sampled.

GPU training no longer scans all 968059 text rows at startup. Text aliases are
recovered lazily when a sampled row is used. If a row remains unresolved, that
row is skipped and replaced by the next usable manifest row in cyclic order so
every rank keeps a complete microbatch and DDP remains synchronized. The final
training report records the requested and selected indices, distinct invalid
rows encountered, replacement counts, and representative failures.

Run the complete text inspection separately on CPU before formal training:

~~~bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl
TEXT_REPORT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow_shared_store_configurable_epochs/mellow_text_audit_20260927.json

python scripts/audit_mellow_faithful_text_manifest.py \
  --manifest "$MANIFEST" \
  --report "$TEXT_REPORT" \
  --expected-rows 968059 \
  --progress-every 50000
~~~

The audit is CPU-only, loads no model weights, consumes no training RNG, and
writes status=PASS only when all 968059 rows are valid after metadata-alias
recovery. A FAIL report still lists invalid rows; formal training can continue
because its runtime policy skips those rows.

The step-9970 run produced no checkpoint under the previous epoch-boundary
policy. It cannot be resumed and must be restarted with a fresh output
directory after pulling this fix. New formal runs save every 5000 optimizer
steps, so later interruptions can resume from the newest retained checkpoint.

With 968059 manifest rows and formal global batch 256, the shape is 3781 optimizer steps per epoch, 123 dropped rows per epoch, and 113430 total steps. The final retained checkpoints are expected at steps 100000, 105000, 110000, and 113430.

## Build the variable-length unique store on CPU

Run directly in a CPU terminal. There is no submission wrapper.
The builder uses `torchaudio.load` when its codec backend is available. With
TorchAudio 2.9 environments that do not include TorchCodec, it falls back to
SoundFile float32 decoding while retaining the same mono and 32 kHz resampling
contract.

~~~bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl
STORE=/hpc_stor03/sjtu_home/jinwei.zhang/data/rsmol_reasonaqa_mellow_faithful_full_waveforms_32k_f32_v2

python scripts/prepare_mellow_faithful_unique_audio_waveform_store.py \
  --manifest "$MANIFEST" \
  --output-dir "$STORE" \
  --workers 8 \
  --torch-threads-per-worker 1 \
  --checkpoint-every 100 \
  --verify-samples 128 \
  --free-space-margin-gib 10 \
  --dry-run

python scripts/prepare_mellow_faithful_unique_audio_waveform_store.py \
  --manifest "$MANIFEST" \
  --output-dir "$STORE" \
  --workers 8 \
  --torch-threads-per-worker 1 \
  --checkpoint-every 100 \
  --verify-samples 128 \
  --free-space-margin-gib 10
~~~

If an identical interrupted build retains BUILDING, build_config.json, progress.json, and both partial files, rerun the second command with --resume.

## Required GPU qualification and formal sequence

Use fresh output directories. Formal training accepts the existing global-batch-32 PASS smoke20 report as a qualification run when its dataset, optimizer, store, and runtime audits match. Formal itself uses global batch 256. The previous resume2 attempts reached step 22 but failed only the exact cross-run loss comparison; they are not claimed as exact-resume PASS. By request, reference22 and resume2 are optional diagnostics and do not block formal training. Progress logs include `step_seconds`.

~~~bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow_shared_store_configurable_epochs
SMOKE_REPORT="$ROOT/smoke20_30epochs_20260926_gbs32_rngboundary_v1/shared_store_training_report.json"
FORMAL_TAG=20260926_gbs32_direct_formal_v1

bash run_audio_smollm2_shared_store_formal_configurable_epochs_135m_mellow_5090.sh \
  --epochs 30 \
  --smoke20-report "$SMOKE_REPORT" \
  --output-dir "$ROOT/formal_30epochs_$FORMAL_TAG"
~~~

The formal wrapper submits one pdgpu-5090 node with 8 GPUs, 32 CPUs, and 256 GiB memory. Local checks only establish code readiness; remote PASS requires the generated report and checkpoints.
