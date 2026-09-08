# Isolated MeSH audio route

This route combines the existing 5-10x2-5 MeSH checkpoint with Mellow's
HTSAT interface and ReasonAQA manifests.  It does not modify the legacy
5-10-5, Mellow, or text-only MeSH scripts.

The contract is 32 kHz mono audio normalized to 10 seconds by cropping the
first 10 seconds or right-padding.  Empty audio2 rows reuse audio1 and, when
possible, the same encoded prefix.  HTSAT is frozen; Mellow c2l (527 to 768),
the projection/downsampling bridge, MeSH, and routers are trainable.

The formal route is 8 GPUs, microbatch 4 per GPU, GA 4 (effective global
batch 128), 3 epochs, max LR 1e-3, cosine schedule, warmup
`ceil(total_optimizer_steps * 0.05)`, and
gradient clipping 0.5.  Each sample is tokenized as `prompt + answer` before
the batch is right-padded to its longest complete text sequence.  Only the
real answer interval has labels; all audio, separator, prompt, and trailing
batch-padding positions are `-100`.  The standard causal-LM shift therefore
predicts the first answer token from the immediately preceding real prompt
token without inserting padding between prompt and answer.

GPU execution must go through the repository-level `vc submit` wrappers; do
not run the CUDA Python/torchrun scripts directly on a login node:

```text
run_audio_stage4_5_10x2_5_mesh_mellow_5090.sh  # 1-GPU forward/backward audit
run_audio_stage5_5_10x2_5_mesh_mellow_5090.sh  # 8-GPU topology smoke
run_audio_stage7_5_10x2_5_mesh_mellow_5090.sh  # 8-GPU 10-step/reload smoke
run_audio_formal_5_10x2_5_mesh_mellow_5090.sh  # 8-GPU formal 3-epoch training
run_audio_checkpoint_audit_5_10x2_5_mesh_mellow_5090.sh  # 1-GPU standalone checkpoint reload audit
```

The wrappers accept the same arguments as their underlying runtime scripts and
submit them inside the project GPU image.
