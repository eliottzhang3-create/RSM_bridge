# Audio MeSH node-shared unique-store training

This is an isolated training route for the existing
`audio_5_10x2_5_mesh_mellow` model.  It changes only the data-residency and
sampling route.  It does not import or modify the six-partition trainer.

## Fixed contract

- Contract: `node_shared_unique_store_fullshuffle_fixed260_audio_reuse_answer_eos_v2`.
- Initial weights: the current MeSH text checkpoint `checkpoint-009244`.
- Audio/model semantics: the audited Mellow HTSAT + c2l + bridge path, fixed
  260-token single/dual prefix, and answer-only loss including terminal
  `<|endoftext|>`. A structural single-audio row reuses audio1's HTSAT
  embedding for slot two, then runs both embeddings through the bridge
  independently with the same weights and separate dropout draws, matching
  the historical 3-epoch route. Dual rows use both real audio embeddings.
- Topology: one node, 8 GPUs, micro batch 8 per rank, gradient accumulation 4,
  effective global batch 256, and `num_workers=0`.
- Sampling: one `DistributedSampler` over the complete canonical manifest,
  `shuffle=True`, a new deterministic permutation at every epoch, disjoint
  rank slices, and `drop_last=True`.
- Storage: the complete immutable v3 unique waveform store is copied once from
  persistent storage to a run-unique directory below `/dev/shm`.  All eight
  ranks mmap the same `waveforms.f32` inode.  There is no rank-local copy of
  the full store. No additional decode/resample/crop/pad is applied before
  HTSAT; its internal feature extraction remains unchanged. Store staging
  time is reported separately from training.
- Checkpoints do not contain the dataset/store.  They contain model, tokenizer,
  bridge/c2l, optimizer, scheduler, exact epoch/batch/global-step cursor, and
  every rank's RNG state.

Persistent inputs:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/rsmol_reasonaqa_train_unique_waveforms_32k_10s_f32_v3
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl
```

The staging wrapper requires store metadata status `PASS`, rejects `BUILDING`,
checks `/dev/shm` capacity with a 10 GiB margin, copies the store and manifest,
and verifies waveform size plus manifest/index SHA256 before launching DDP.
The Python trainer repeats the identity and format audits and proves that all
ranks see the same staged file device/inode/size.

## Required release sequence

The previous compact shared-store smoke/resume passed according to the user.
This fixed260 contract requires new smoke/resume reports and rejects old
compact checkpoints. Use fresh output directories for the commands below.
Formal training initializes from the text checkpoint, not the smoke weights.

Run from the remote repository root:

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

bash run_audio_shared_store_smoke20_5_10x2_5_mesh_mellow_5090.sh \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/smoke20_20260922

bash run_audio_shared_store_resume2_5_10x2_5_mesh_mellow_5090.sh \
  --resume-from /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/smoke20_20260922/checkpoint-000020 \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/resume2_20260922

bash run_audio_shared_store_formal_5_10x2_5_mesh_mellow_5090.sh \
  --smoke20-report /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/smoke20_20260922/shared_store_training_report.json \
  --smoke-resume-report /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/resume2_20260922/shared_store_training_report.json \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/formal_3epochs_20260922
```

The first smoke stops at optimizer step 20 and publishes
`checkpoint-000020`.  Resume must start at the exact cursor
`epoch=0,batch_in_epoch=80,global_step=20`, restores all training state, runs
exactly two steps, and publishes `checkpoint-000022`.  Formal training refuses
to start unless both reports are `PASS`, use this exact contract and store
identity, and pass the answer-mask and MeSH 5-10-10-5 gradient/trace audits.

The fixed formal schedule is 3 epochs, maximum LR `1e-3`, minimum LR `1e-4`,
5% warmup rounded up to whole optimizer steps, save every 500 steps, and retain
the newest four complete checkpoints. Smoke and its resume use the same
3-epoch scheduler horizon and resume-validated settings. For 968,059 manifest
rows this is 3,781 steps/epoch, 11,343 total steps and 568 warmup steps. Smoke
only stops early; it does not compress this schedule into 22 steps. The final
formal checkpoint is `checkpoint-011343`.

Every run writes `shared_store_training_report.json` to its output directory.
Output directories must be new or empty.  `/dev/shm` staging is removed on job
exit; the persistent v3 store is never modified.

## Isolated configurable-epoch copy

The original three-epoch entrypoints above remain unchanged so an existing job
is not affected.  A separate copied entrypoint family accepts an explicit
positive `--epochs` value while preserving this package's fixed-260 model,
full-manifest sampler, shared-store staging, optimizer, LR endpoints,
checkpoint, exact-resume, and formal-gate contracts.

For every run, the scheduler horizon is calculated from the real complete
optimizer-step budget:

```text
steps_per_epoch = floor(dataset_rows / (world_size * micro_batch * GA))
total_steps     = steps_per_epoch * epochs
warmup_steps    = ceil(total_steps * 0.05)
```

With 968,059 rows, 8 ranks, microbatch 8 and GA 4, `--epochs 10` means 3,781
steps/epoch, 37,810 total optimizer steps, and 1,891 warmup steps.  Smoke20,
resume2, and formal must all use `--epochs 10`; reports or checkpoints created
with the old three-epoch horizon are rejected by shape, epoch, warmup, and
checkpoint configuration checks.

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_shared_store_configurable_epochs

bash run_audio_shared_store_smoke20_configurable_epochs_5_10x2_5_mesh_mellow_5090.sh \
  --epochs 10 \
  --output-dir "$ROOT/smoke20_10epochs_YYYYMMDD"

bash run_audio_shared_store_resume2_configurable_epochs_5_10x2_5_mesh_mellow_5090.sh \
  --epochs 10 \
  --resume-from "$ROOT/smoke20_10epochs_YYYYMMDD/checkpoint-000020" \
  --output-dir "$ROOT/resume2_10epochs_YYYYMMDD"

bash run_audio_shared_store_formal_configurable_epochs_5_10x2_5_mesh_mellow_5090.sh \
  --epochs 10 \
  --smoke20-report "$ROOT/smoke20_10epochs_YYYYMMDD/shared_store_training_report.json" \
  --smoke-resume-report "$ROOT/resume2_10epochs_YYYYMMDD/shared_store_training_report.json" \
  --output-dir "$ROOT/formal_10epochs_YYYYMMDD"
```

The formal run starts fresh from the configured text MeSH checkpoint.  The
smoke checkpoint is used only to prove exact step-20 to step-22 resume and to
release the formal gate.
