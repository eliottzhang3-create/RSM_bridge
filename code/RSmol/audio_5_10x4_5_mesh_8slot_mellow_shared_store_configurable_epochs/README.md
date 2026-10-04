# Audio 5-10x4-5 MeSH 8-slot shared-store training

This isolated route initializes the text backbone from:

`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh_8slot/formal_third_epoch_3081steps_20261003_221944_3090_v1/checkpoint-003081`

It keeps the validated fixed-260 shared-store audio protocol: eight GPUs,
per-rank microbatch 8, gradient accumulation 4, full-manifest shuffle, one
node-shared `/dev/shm` waveform store, runtime-zero second audio slot for
structurally single-audio rows, answer-only loss with terminal EOS, and AdamW
with LR `1e-3` to `1e-4`. The text backbone is the independent 5-10x4-5
8-slot model: 50 logical executions, 20 physical layers, eight memory slots,
five router groups, and ten router modules.

The route is formal-only. It performs its source checkpoint, store, label/EOS,
runtime trace, gradient, and trainable-module audits inside the formal job; it
has no smoke or resume gate.

Run from `code/RSmol` after syncing the repository:

```bash
bash run_audio_shared_store_formal_configurable_epochs_5_10x4_5_mesh_8slot_mellow_3090.sh \
  --epochs 3
```

The wrapper submits `pdgpu-3090` with 8 GPUs, 32 CPU cores, and 256G memory,
and writes output below
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x4_5_mesh_8slot_mellow_shared_store_configurable_epochs`.
