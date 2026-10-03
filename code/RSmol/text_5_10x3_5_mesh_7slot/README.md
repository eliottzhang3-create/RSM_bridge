# Isolated Text 5-10x3-5 MeSH, 7-slot / 4-router

This route is isolated from the existing 5-10x2-5, 5-10x4-5, and
5-10x5-5 text routes. It is the text-only phase of a future x3/7-slot audio
route; this deliverable does not modify or train any audio code.

## Model contract

- 20 physical decoder layers and 40 logical executions: `5 + 10 * 3 + 5`.
- Seven transient MeSH memory slots.
- Four independent router groups, each with one write and one read router;
  eight independent linear router modules in total.
- The source mapping is the established 30-to-20 mapping:
  `0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29`.
- The logical schedule is five prefix layers, the ten physical middle layers
  repeated three times, and five suffix layers.

The converter accepts the clean original 30-layer SmolLM2 checkpoint or the
conversion-only 5-10-5 directory. It rejects training checkpoints by default.

## Training contract

- 8 ranks on `pdgpu-3090`, direct persistent parquet reads.
- Context length 1024, per-rank microbatch 4, gradient accumulation 32;
  effective global batch is `8 * 4 * 32 = 1024`.
- AdamW, betas `(0.9, 0.95)`, weight decay `0.1`, epsilon `1e-8`.
- Step-level cosine schedule from `1e-3` to `1e-4`, warmup 155 steps.
- One configured epoch with 3,081 optimizer steps, one third of the 9,244-step
  reference epoch.
- Checkpoint every 500 optimizer steps and at the final step; retain the
  newest three complete checkpoints.

## Remote commands

Conversion is CPU-only:

```bash
bash code/RSmol/scripts/convert_stepwise_5_10x3_5_mesh_7slot.sh
```

The default converted model is:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x3-5-mesh-7slot
```

After conversion, submit the formal 8-GPU job directly. This isolated route
does not require smoke or resume gates:

```bash
bash code/RSmol/run_stage4_5_10x3_5_mesh_7slot_formal_3090.sh
```

The default output is under:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x3_5_mesh_7slot
```
