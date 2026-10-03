# Audio 5-10x3-5 MeSH 7-slot shared-store training

This is an isolated formal-only audio route initialized from:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x3_5_mesh_7slot/formal_third_epoch_3081steps_3090_v1/checkpoint-003081
```

The mapper, data, staging, loss, optimizer, and checkpoint behavior follow
the validated x2/7-slot shared-store implementation. Only the text backbone,
architecture contract, and runtime MeSH gradient audit are x3-specific.

## Locked contracts

- Text model: 20 physical layers and 40 logical executions (`5 + 10x3 + 5`).
- Seven transient memory slots.
- Four independent write/read router groups, eight router modules total.
- Prefix: two 129-token audio slots plus two separators, always 260 tokens.
- Structurally single-audio rows use one exact all-zero waveform created on the
  current GPU for the second slot.
- Dual-audio rows use both real waveforms; identical real paths may reuse the
  first HTSAT embedding.
- Loss supervises answer tokens only and includes the terminal EOS token.
- HTSAT stays frozen; c2l, bridge, x3 text model, and all eight routers train.
- Complete unique waveform store is staged per job into one node-shared
  `/dev/shm` directory, and all ranks read the same staged waveform inode.
- Geometry: 8 GPUs x microbatch 8 x gradient accumulation 4 = global batch 256.
- Three epochs on 968,059 rows: 3,781 steps per epoch and 11,343 total steps.
- Warmup: `ceil(11,343 * 5%) = 568` optimizer steps.
- AdamW, betas `(0.9, 0.95)`, weight decay `0.1`, max LR `1e-3`, cosine decay
  to min LR `1e-4`.
- Checkpoints every 500 steps and at the final step; retain four complete
  checkpoints.
- Formal queue: `pdgpu-3090`, 8 GPUs, 32 CPU cores, 256G.

This route intentionally has no smoke/resume submission gate. The formal job
performs the source-checkpoint, shared-store, answer/EOS-label, runtime-zero,
and x3/7-slot trace/gradient audits before optimizer step one.

## Formal submission

From `code/RSmol` after pushing and pulling the route:

```bash
bash run_audio_shared_store_formal_configurable_epochs_5_10x3_5_mesh_7slot_mellow_3090.sh \
  --epochs 3
```

The isolated output root is:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x3_5_mesh_7slot_mellow_shared_store_configurable_epochs
```

The source unique store and manifest can be overridden with
`RSMOL_SHARED_STORE_SOURCE` and `RSMOL_SHARED_MANIFEST_SOURCE`.
