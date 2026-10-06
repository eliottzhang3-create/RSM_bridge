# AdamW / Step-Cosine Comparison Route

This directory is an isolated comparison route derived from
`mellow_official_training_c8204d8`. The model, dataset, staging, loss, DDP
geometry, precision, and checkpoint contract are intentionally unchanged.

The only training-semantic changes are:

- `torch.optim.AdamW`
- `betas=(0.9, 0.95)`
- `weight_decay=1e-4`
- `max_lr=1e-3`
- linear warmup for `ceil(total_optimizer_steps * 0.05)` steps
- step-level cosine decay to `min_lr=5e-5`

The schedule is defined over the full configured epoch horizon. The first
optimizer update uses the first warmup learning rate. Each subsequent
optimizer update advances the scheduler exactly once. The scheduler state,
including the total step contract and current step, is saved in schema-v2
checkpoints and validated during resume.

The outer submission wrappers are in the parent directory:

- `run_mellow_official_reasonaqa_adamw_cosine_smoke_5090.sh`
- `run_mellow_official_reasonaqa_adamw_cosine_resume_smoke_5090.sh`
- `run_mellow_official_reasonaqa_adamw_cosine_30epochs_5090.sh`

All jobs use `pdgpu-5090`, eight GPUs, 32 CPUs, and 256G memory. The raw
audio is restaged into a job-local `/dev/shm` directory for every submission.
