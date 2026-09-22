# Audio MeSH node-shared unique-store training

This is an isolated training route for the existing
`audio_5_10x2_5_mesh_mellow` model.  It changes only the data-residency and
sampling route.  It does not import or modify the six-partition trainer.

## Fixed contract

- Contract: `node_shared_unique_store_fullshuffle_compact_audio_answer_eos_v2`.
- Initial weights: the current MeSH text checkpoint `checkpoint-009244`.
- Audio/model semantics: the audited Mellow HTSAT + c2l + bridge path, compact
  single/dual prefix (130/260 tokens), and answer-only loss including terminal
  `<|endoftext|>`.
- Topology: one node, 8 GPUs, micro batch 8 per rank, gradient accumulation 4,
  effective global batch 256, and `num_workers=0`.
- Sampling: one `DistributedSampler` over the complete canonical manifest,
  `shuffle=True`, a new deterministic permutation at every epoch, disjoint
  rank slices, and `drop_last=True`.
- Storage: the complete immutable v3 unique waveform store is copied once from
  persistent storage to a run-unique directory below `/dev/shm`.  All eight
  ranks mmap the same `waveforms.f32` inode.  There is no rank-local copy of
  the full store.  Store staging time is reported separately from training.
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
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/formal_10epochs_20260922
```

The first smoke stops at optimizer step 20 and publishes
`checkpoint-000020`.  Resume must start at the exact cursor
`epoch=0,batch_in_epoch=80,global_step=20`, restores all training state, runs
exactly two steps, and publishes `checkpoint-000022`.  Formal training refuses
to start unless both reports are `PASS`, use this exact contract and store
identity, and pass the answer-mask and MeSH 5-10-10-5 gradient/trace audits.

Defaults for formal training are 10 epochs, maximum LR `1e-3`, minimum LR `0`,
5% warmup rounded up to whole optimizer steps, save every 500 steps, and retain
the newest four complete checkpoints.  CLI arguments placed after the wrapper
defaults may override these values, but smoke and its resume must use identical
resume-validated settings.

Every run writes `shared_store_training_report.json` to its output directory.
Output directories must be new or empty.  `/dev/shm` staging is removed on job
exit; the persistent v3 store is never modified.
