# Audio 5-10x5-5 MeSH 7-slot shared-store training

This is an isolated formal-only audio route initialized from:

    /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x5_5_mesh/formal_20261001_175242_gbs1024_pdgpu3090/checkpoint-003081

The route follows the validated shared-store audio implementation. Its text
backbone is the 5-10x5-5 MeSH model, while the mapper, data, staging, loss,
and optimizer behavior remain the same as the validated x2/7-slot route.

## Locked contracts

- Text model: 20 physical layers, 60 logical executions (5 + 10x5 + 5),
  seven transient memory slots, six independent write/read router groups,
  and twelve router modules.
- Text checkpoint: the complete x5 checkpoint above. Its metadata stores the
  canonical `logical_to_physical` schedule and proves router optimization.
- Mapper: Mellow c2l(527,768), followed by the existing 768-to-576-to-576,
  dropout 0.5, residual LayerNorm, CLS-preserving avg-pool-8 bridge.
- Frozen: the HTSAT AudioSet backbone remains in eval mode without gradients.
- Trainable: c2l, the bridge, and the complete x5 text model including all
  twelve router modules.
- Prefix: two 129-token audio slots plus two separators, always 260 tokens.
- Structurally single-audio row: slot one uses the real waveform. Slot two is
  one exact all-zero waveform created on the current GPU, encoded once per
  microbatch, then expanded before the trainable bridge.
- Dual-audio row: both real inputs are used. Explicit identical-path dual rows
  may reuse the first HTSAT embedding, while bridge calls remain independent.
- Loss: answer tokens only, including the terminal end-of-text token.
- Data: the complete audited unique waveform store is copied once to a unique
  `/dev/shm` directory and all eight ranks mmap the same waveform inode.
- Sampler: full-manifest DistributedSampler with shuffle and drop_last enabled.
- Geometry: 8 GPUs x microbatch 8 x gradient accumulation 4 = global batch 256.
- Three epochs on 968,059 rows: 3,781 steps per epoch and 11,343 total steps.
- Warmup: ceil(11,343 x 5%) = 568 steps.
- Optimizer schedule: AdamW, max LR 1e-3, cosine decay to min LR 1e-4.
- Checkpoints: every 500 steps and the final step; retain four complete copies.
- Queue: formal submission uses `pdgpu-3090`, 8 GPUs, 32 CPU cores, 256G.

This deliverable intentionally has no smoke/resume submission gate. The formal
run performs source-checkpoint, shared-store, label/EOS, runtime-zero, trainable
module, and x5/7-slot gradient audits before optimizer step one.

## Formal submission

Run from `code/RSmol` after pushing this route to the remote checkout:

    bash run_audio_shared_store_formal_configurable_epochs_5_10x5_5_mesh_7slot_mellow_3090.sh \
      --epochs 3

The wrapper creates an isolated output directory under:

    /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x5_5_mesh_7slot_mellow_shared_store_configurable_epochs

The source store and manifest can still be overridden with
`RSMOL_SHARED_STORE_SOURCE` and `RSMOL_SHARED_MANIFEST_SOURCE` when needed.
