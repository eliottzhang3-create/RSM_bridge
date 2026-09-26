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
- Mellow-matched global batch: 8 GPUs, microbatch 4 per rank, no gradient accumulation (1), effective global batch 32, num_workers 0.
- Thirty epochs. Save at every epoch boundary and retain only the newest three complete checkpoints.
- Dataset randomness is stateful Python random. Checkpoints include every rank's Python, Torch, and CUDA RNG state.
- Resume leaves Adam's non-capturable scalar step tensors on CPU, matching a continuous run. The DataLoader uses a private generator for its iterator seed so reconstruction at the resume cursor does not advance the training RNG stream.
- Sampler order is torch.randperm with seed equal to the epoch, followed by contiguous rank slices. Resume starts directly at the saved optimizer-step cursor and does not replay prior batches.

With 968059 manifest rows, the shape is 30251 optimizer steps per epoch, 27 dropped rows per epoch, and 907530 total steps. The final retained checkpoints are expected at steps 847028, 877279, and 907530.

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

Use fresh output directories. The uninterrupted reference22 run is mandatory: resume2 compares steps 21 and 22 across all ranks, including row indices, selected audio IDs, crop offsets, templates, losses, LR, and final model/optimizer/scheduler fingerprints.

~~~bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow_shared_store_configurable_epochs
TAG=20260926_gbs32_v1

bash run_audio_smollm2_shared_store_smoke20_configurable_epochs_135m_mellow_3090.sh \
  --epochs 30 \
  --output-dir "$ROOT/smoke20_30epochs_$TAG"

bash run_audio_smollm2_shared_store_reference22_configurable_epochs_135m_mellow_3090.sh \
  --epochs 30 \
  --output-dir "$ROOT/reference22_30epochs_$TAG"

bash run_audio_smollm2_shared_store_resume2_configurable_epochs_135m_mellow_3090.sh \
  --epochs 30 \
  --resume-from "$ROOT/smoke20_30epochs_$TAG/checkpoint-000020" \
  --reference22-report "$ROOT/reference22_30epochs_$TAG/shared_store_training_report.json" \
  --output-dir "$ROOT/resume2_30epochs_$TAG"

bash run_audio_smollm2_shared_store_formal_configurable_epochs_135m_mellow_3090.sh \
  --epochs 30 \
  --smoke20-report "$ROOT/smoke20_30epochs_$TAG/shared_store_training_report.json" \
  --reference22-report "$ROOT/reference22_30epochs_$TAG/shared_store_training_report.json" \
  --smoke-resume-report "$ROOT/resume2_30epochs_$TAG/shared_store_training_report.json" \
  --output-dir "$ROOT/formal_30epochs_$TAG"
~~~

All four GPU wrappers submit one pdgpu-3090 node with 8 GPUs, 32 CPUs, and 256 GiB memory. Local checks only establish code readiness; remote PASS requires the generated reports and checkpoints.
