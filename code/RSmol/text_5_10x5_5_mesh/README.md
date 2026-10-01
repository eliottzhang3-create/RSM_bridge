# Text 5-10x5-5 MeSH isolated route

This route is isolated from every audio route and from the existing text
5-10x2-5, 5-10x4-5, and 5-10x6-5 implementations.

## Fixed contract

- 20 physical decoder layers and 60 logical executions: `5 + 10 * 5 + 5`.
- Seven transient MeSH memory slots.
- Six independent router groups. Each group owns one write router and one
  read router, for twelve independent linear router modules.
- The source layer mapping is identical to the prior MeSH conversions:
  `0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29`.
- Eight ranks, microbatch 4, gradient accumulation 32, effective global batch
  1024, and context length 1024.
- The historical reference epoch is 9,244 optimizer steps. Formal training
  runs exactly one third with 3,081 optimizer steps, using one configured
  epoch.
- Learning rate warms for 155 steps to `1e-3`, then cosine decays to `1e-4`
  over the 3,081-step target.
- Checkpoints are written every 500 optimizer steps and at the final step;
  only the newest three complete checkpoints are retained.
- Sorted parquet shards are assigned by shard index modulo 8. Each rank
  streams its assigned persistent parquet shards directly from the remote
  data directory.
- Production training does not copy the dataset into `/dev/shm`; this route
  has no shared-memory staging dependency.

## CPU-only conversion

Do not submit conversion as a GPU job:

    cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
    conda activate rsmol
    bash scripts/convert_stepwise_5_10x5_5_mesh.sh

The default output is:

    /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x5-5-mesh

Optional CPU parquet inventory:

    python scripts/audit_text_parquet_store_5_10x5_5_mesh.py \
      --data-dir /hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset/data \
      --report-path /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x5_5_mesh/preflight/source_inventory.json

## Optional Stage 1 CUDA audit

The isolated route retains a Stage 1 architecture audit, but it does not
create smoke or smoke-resume wrappers. The audit is diagnostic only and is
not a formal-training gate:

    bash run_audit_stage1_5_10x5_5_mesh_3090.sh

The audit checks the 60 logical trace entries, five recursive transitions,
seven memory slots, twelve router modules, router gradients, cache behavior,
generation, and save/reload consistency.

## Direct formal training

After conversion, submit formal training directly:

    bash run_stage4_5_10x5_5_mesh_formal_3090.sh

The wrapper uses the `pdgpu-3090` queue with 8 GPUs, 32 CPU cores, and 256G
memory. It does not inspect or require smoke/resume PASS reports.

The default output follows this pattern:

    /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x5_5_mesh/formal_third_epoch_3081steps_<run_timestamp>_3090_v1

A fresh formal run starts from the converted x5 model. To continue an
interrupted formal run, provide a complete x5 formal checkpoint and a new
output directory:

    export RSMOL_5_10X5_5_MESH_RESUME_FROM=/path/to/formal/checkpoint-002000
    export RSMOL_5_10X5_5_MESH_OUTPUT_DIR=/path/to/new/formal_resume_output
    bash run_stage4_5_10x5_5_mesh_formal_3090.sh

The formal wrapper validates the completion marker, checkpoint manifest, and
training state before forwarding the checkpoint into the submitted FORMAL job.

## Resume cursor and numerical policy

Formal resume restores each rank's exact shard and row offset. Rows already
consumed inside the current shard are skipped before the first resumed
microbatch. New checkpoints save each rank's CUDA RNG state; older checkpoints
remain loadable when that optional field is absent.

Gradient clipping remains fixed at global norm 1.0. If an accumulation window
produces a nonfinite gradient norm, all ranks discard that window, keep the
optimizer and scheduler step unchanged, clear gradients, and consume the next
data window for the same optimizer step. Formal training still completes
exactly 3,081 successful optimizer updates.

The default safety limits are 32 consecutive skipped windows and 256 total
skipped windows. They can be adjusted with:

    RSMOL_5_10X5_5_MESH_MAX_CONSECUTIVE_NONFINITE_WINDOWS
    RSMOL_5_10X5_5_MESH_MAX_TOTAL_NONFINITE_WINDOWS
