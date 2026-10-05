# AdamW / Step-Cosine 5-10x2-5 MeSH Comparison Route

This directory is an isolated comparison route derived from
`mellow_official_training_c8204d8`. The audio encoder, dataset, staging, loss
window, DDP geometry, precision, and optimizer contract follow that route. The
text decoder is loaded from the converted 5-10x2-5 MeSH checkpoint directory
and is the only model architecture adaptation.

The only training-semantic changes are:

- `torch.optim.AdamW`
- `betas=(0.9, 0.95)`
- `weight_decay=1e-4`
- `max_lr=1e-3`
- linear warmup for `ceil(total_optimizer_steps * 0.05)` steps
- step-level cosine decay to `min_lr=5e-5`
- answer loss reduced as one global token mean across all ranks and four
  accumulated microbatches

The required text initialization directory is:

`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244`

The loader checks hidden size 576, logical/physical depth 30/20, two middle
loops, five memory slots, six routers, and the explicit 30-slot schedule before
constructing Mellow.

The schedule is defined over the full configured epoch horizon. The first
optimizer update uses the first warmup learning rate. Each subsequent
optimizer update advances the scheduler exactly once. The scheduler state,
including the total step contract and current step, is saved in schema-v2
checkpoints and validated during resume.

The outer submission wrappers are in the parent directory:

- `run_mellow_official_reasonaqa_adamw_cosine_5_10x2_5_mesh_smoke_5090.sh`
- `run_mellow_official_reasonaqa_adamw_cosine_5_10x2_5_mesh_resume_smoke_5090.sh`
- `run_mellow_official_reasonaqa_adamw_cosine_5_10x2_5_mesh_30epochs_5090.sh`

The smoke route samples exactly 5120 records, which is one complete epoch at
the fixed eight-GPU geometry (20 optimizer steps), and publishes
`model--step-20.ckpt` with `epoch_completed=1`. Resume validates that full
checkpoint and advances exactly to step 22. Every runner removes its temporary
`/dev/shm` staging tree on exit, including failures.

The smoke runner executes `scripts/rsmol/audit_global_token_mean.py` before
staging, which checks both unequal-token weighting and the DDP world-size
scaling algebra.

All jobs use `pdgpu-5090`, eight GPUs, 32 CPUs, and 256G memory. The raw
audio is restaged into a job-local `/dev/shm` directory for every submission.
