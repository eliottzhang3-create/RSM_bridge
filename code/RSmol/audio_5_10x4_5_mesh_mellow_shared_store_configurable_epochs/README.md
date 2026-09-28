# Audio 5-10x4-5 MeSH shared-store training

This isolated route initializes the text model from:

    /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/formal_third_epoch_3081steps_20260927_3090_v1/checkpoint-003081

It preserves the successful x2 shared-store data and mapper implementation,
while replacing the text backbone with x4 MeSH and applying the requested
single-audio second-slot policy.

## Locked contracts

- Text model: 20 physical layers, 50 logical executions (5 + 10x4 + 5), seven
  transient memory slots, and five independent write/read router groups.
- Mapper: Mellow c2l(527,768), followed by the existing 768-to-576-to-576,
  dropout 0.5, residual LayerNorm, CLS-preserving avg-pool-8 bridge.
- Frozen: the HTSAT AudioSet backbone remains in eval mode without gradients.
- Trainable: c2l, the bridge, and the complete x4 text model including routers.
- Prefix: two 129-token audio slots plus two separators, always 260 tokens.
- Structurally single-audio row: slot one uses the real waveform. Slot two uses
  one exact all-zero waveform created on the current GPU. The real waveform is
  not reused. The zero waveform is encoded once per microbatch and expanded
  before the trainable bridge.
- Dual-audio row: both real inputs are used. An explicit identical-path dual row
  may reuse the first HTSAT embedding, while bridge calls remain independent.
- Loss: answer tokens only, including the terminal end-of-text token.
- Data: the complete audited unique waveform store is copied once to a unique
  /dev/shm directory and all eight ranks mmap the same waveform inode.
- Sampler: full-manifest DistributedSampler with shuffle and drop_last enabled.
- Geometry: 8 GPUs x microbatch 8 x gradient accumulation 4 = global batch 256.
- Three epochs on 968,059 rows: 3,781 steps per epoch and 11,343 total steps.
- Warmup: ceil(11,343 x 5%) = 568 steps.
- Optimizer schedule: AdamW, max LR 1e-3, cosine decay to min LR 1e-4.
- Checkpoints: every 500 steps and the final step; retain four complete copies.
- Queue: all submitted GPU jobs use pdgpu-3090.

The smoke job is the authoritative CUDA memory check for microbatch eight on
the allocated 3090 nodes. Formal training starts fresh from the text checkpoint
and requires matching PASS smoke20 and resume2 reports.

## Submission commands

Run from code/RSmol after pushing the code to the remote checkout.

    SMOKE=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/smoke20_3epochs_20260928_v1
    bash run_audio_shared_store_smoke20_configurable_epochs_5_10x4_5_mesh_mellow_3090.sh \
      --epochs 3 \
      --output-dir "$SMOKE"

    SMOKE=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/smoke20_3epochs_20260928_v1
    RESUME=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/resume2_3epochs_20260928_v1
    bash run_audio_shared_store_resume2_configurable_epochs_5_10x4_5_mesh_mellow_3090.sh \
      --epochs 3 \
      --resume-from "$SMOKE/checkpoint-000020" \
      --output-dir "$RESUME"

    SMOKE=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/smoke20_3epochs_20260928_v1
    RESUME=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/resume2_3epochs_20260928_v1
    FORMAL=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/formal_3epochs_20260928_v1
    bash run_audio_shared_store_formal_configurable_epochs_5_10x4_5_mesh_mellow_3090.sh \
      --epochs 3 \
      --smoke20-report "$SMOKE/shared_store_training_report.json" \
      --smoke-resume-report "$RESUME/shared_store_training_report.json" \
      --output-dir "$FORMAL"

