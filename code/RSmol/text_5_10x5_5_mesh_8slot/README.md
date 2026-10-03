# Text 5-10x5-5 MeSH isolated 8-slot route

This route is isolated from the existing 7-slot x4/x5 routes and every audio
route.

## Fixed contract

- 20 physical decoder layers and 60 logical executions: `5 + 10 * 5 + 5`.
- Eight transient MeSH memory slots. The logical cache still has 60 entries.
- Six independent router groups (prefix plus five loop transitions). Each
  group owns one write router and one read router: twelve independent router
  parameter modules.
- The 30-to-20 source mapping is unchanged from the validated MeSH routes:
  `0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29`.
- Router initialization uses `sqrt(2 / (8 * hidden_size))`, with the 8-slot
  policy recorded in the conversion metadata.

## Text training contract

- `pdgpu-3090`, one node, 8 GPUs, 32 CPU cores, 256G memory.
- Per-rank microbatch 4, gradient accumulation 32, effective global batch
  `8 * 4 * 32 = 1024`, context length 1024.
- Direct persistent Parquet reads; no audio staging and no `/dev/shm`
  production dataset copy.
- BF16 autocast, token-weighted accumulation, and the validated x5 text loss
  path are unchanged.
- Reference epoch: 9,244 optimizer steps. Formal run: 3,081 steps (one third,
  floor division), one configured epoch.
- AdamW betas `(0.9, 0.95)`, weight decay `0.1`, LR `1e-3` to `1e-4` with
  155-step warmup and step-level cosine decay.
- Checkpoint every 500 steps and at the final step; retain the newest three
  complete checkpoints.

## Conversion

Run the CPU-only converter from the repository checkout after syncing the
code. It starts from the clean `SmolLM2-5-10-5` source, not from a 7-slot
checkpoint:

    cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
    conda activate rsmol
    bash scripts/convert_stepwise_5_10x5_5_mesh_8slot.sh

Default output:

    /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x5-5-mesh-8slot

## Formal training

After conversion succeeds, submit the isolated formal wrapper directly:

    bash run_stage4_5_10x5_5_mesh_8slot_formal_3090.sh

Default output:

    /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x5_5_mesh_8slot/formal_third_epoch_3081steps_<timestamp>_3090_v1

The trainer rejects a model whose architecture contract, memory slot count,
or router parameter count does not match this route.
