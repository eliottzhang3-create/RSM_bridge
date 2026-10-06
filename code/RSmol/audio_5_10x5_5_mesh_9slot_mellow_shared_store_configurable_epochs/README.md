# Audio 5-10x5-5 MeSH 9-slot shared-store training

This isolated route initializes the text backbone from:

`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x5_5_mesh_9slot/formal_third_epoch_3081steps_20261005_212437_3090_v1/checkpoint-003081`

It keeps the validated fixed-260 shared-store audio protocol: eight GPUs,
per-rank microbatch 8, gradient accumulation 4, full-manifest shuffle, one
node-shared `/dev/shm` waveform store, runtime-zero second audio slot for
structurally single-audio rows, answer-only loss with terminal EOS, and AdamW
with LR `1e-3` to `1e-4`. The text backbone is the independent 5-10x5-5
9-slot model: 60 logical executions, 20 physical layers, nine memory slots,
six router groups, and twelve router modules.

The route is formal-only. It performs its source checkpoint, store, label/EOS,
runtime trace, gradient, and trainable-module audits inside the formal job; it
has no smoke or resume gate.

Run from `code/RSmol` after syncing the repository:

```bash
bash run_audio_shared_store_formal_configurable_epochs_5_10x5_5_mesh_9slot_mellow_3090.sh \
  --epochs 3
```

The wrapper submits `pdgpu-3090` with 8 GPUs, 32 CPU cores, and 256G memory,
and writes output below
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x5_5_mesh_9slot_mellow_shared_store_configurable_epochs`.

## MMAU test-mini full evaluation

The fixed-260 runtime-zero MMAU evaluator uses the same author-reply and
official scoring contract as the existing 7/8-slot MeSH routes.  It loads the
completed `checkpoint-011343` above, uses one `pdgpu-3090` GPU, and writes the
comparison artifacts below `mmau_audio_mesh_zero_slot/x5_9slot`.

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mmau_test_mini_audio_mesh_9slot_3090.sh
```

To choose a unique output directory explicitly:

```bash
RSMOL_MMAU_9SLOT_OUTPUT_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mmau_audio_mesh_zero_slot/x5_9slot/full_20261006_9slot \
  bash run_mmau_test_mini_audio_mesh_9slot_3090.sh
```
