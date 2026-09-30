# Isolated Text 5-10x2-5 MeSH, 7-slot / 3-router

This route is isolated from the existing 5-10x2-5 (5-slot) and 5-10x4-5
(7-slot / 5-router) text routes.

## Model contract

- 20 physical decoder layers and 30 logical executions: `5 + 10 * 2 + 5`.
- Seven transient MeSH memory slots.
- Three independent router groups, each with one write and one read router;
  six independent linear router modules in total.
- The source mapping is the established 30-to-20 mapping:
  `0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29`.

The converter accepts the clean original 30-layer SmolLM2 checkpoint or the
conversion-only 5-10-5 directory. It intentionally does not accept an x4
MeSH model as a source.

## Training contract

- 8 ranks on `pdgpu-3090`, direct persistent parquet reads.
- Context length 1024, per-rank microbatch 4, gradient accumulation 32;
  effective global batch is `8 * 4 * 32 = 1024`.
- AdamW, betas `(0.9, 0.95)`, weight decay `0.1`, epsilon `1e-8`.
- Step-level cosine schedule from `1e-3` to `1e-4`, warmup 155 steps.
- One configured epoch with 3081 optimizer steps, one third of the 9244-step
  reference epoch.
- Checkpoint every 500 optimizer steps and at the final step; retain the
  newest three complete checkpoints.

## Remote commands

Conversion is CPU-only:

```bash
bash code/RSmol/scripts/convert_stepwise_5_10x2_5_mesh_7slot.sh
```

Then run the 8-GPU smoke, resume gate, and formal training wrappers:

```bash
bash code/RSmol/run_stage4_5_10x2_5_mesh_7slot_smoke_3090.sh
export RSMOL_5_10X2_5_MESH_7SLOT_RESUME_FROM=/path/to/smoke/checkpoint-000010
bash code/RSmol/run_stage4_5_10x2_5_mesh_7slot_resume_3090.sh
bash code/RSmol/run_stage4_5_10x2_5_mesh_7slot_formal_3090.sh
```
