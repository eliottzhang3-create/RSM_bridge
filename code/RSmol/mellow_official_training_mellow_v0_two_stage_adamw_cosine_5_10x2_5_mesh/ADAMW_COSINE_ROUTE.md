# Mellow-v0 two-stage ReasonAQA route

This isolated route is based on the official Mellow AdamW/cosine MeSH
implementation. It is separate from every earlier `5_10x2_5_mesh` experiment.

Initialization uses:

`/hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/converted/mellow_v0_5_10x2_5_mesh_epoch17_routers`

The directory must contain `text_model/` and
`mellow_v0_5_10x2_5_mesh_init.pt`. The init checkpoint is loaded as model
weights only; optimizer, scheduler, and RNG state start fresh.
The 12 router tensors come from the audited epoch-17 checkpoint. The other
389 tensors retain their original Mellow-v0 initialization values.

## Training stages

- Stage 1: 5 ReasonAQA epochs. HTSAT is frozen; c2l, bridge, text decoder,
  and routers are trainable. Routers use `1e-3 -> 1e-4`; all other trainable
  parameters use `1e-4 -> 1e-5`.
- Stage 2: a fresh additional 5 ReasonAQA epochs initialized from the stage
  1 full checkpoint. HTSAT remains frozen and every other trainable parameter
  uses `5e-4 -> 5e-5`. Stage 2 runs on eight GPUs with per-rank batch 4
  and no gradient accumulation, giving effective global batch 32. Stage 1
  retains per-rank batch 8 and accumulation 4, giving global batch 256.
- Both stages use 5% step warmup followed by cosine decay. Stage 2 does not
  restore stage 1 optimizer or scheduler state.

The route writes schema-v2 full checkpoints with explicit route, stage,
trainability, parameter-group, scheduler, and resume contracts.

The six `submit_*_3090.sh` wrappers submit to `pdgpu-3090` with eight GPUs,
32 CPUs, and 256G memory. Smoke uses 20 optimizer steps; resume uses two
additional steps from the audited smoke checkpoint.
