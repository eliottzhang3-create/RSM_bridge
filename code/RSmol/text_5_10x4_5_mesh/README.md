# Text 5-10x4-5 MeSH isolated route

This route is isolated from every audio route and from the existing text 5-10x2-5 implementation.

## Fixed contract

- 20 physical decoder layers and 50 logical executions: 5 + 10 * 4 + 5.
- Seven transient MeSH memory slots.
- Five independent router groups. Each group owns one write router and one read router, for ten independent linear router modules.
- Source layer mapping remains identical to the prior 5-10x2-5 conversion: 0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29.
- Eight ranks, microbatch 8, gradient accumulation 16, effective global batch 1024, context length 1024.
- Two epochs of 9,244 optimizer steps each, 18,488 total steps.
- LR warms for 925 steps to 1e-3, then cosine decays to 1e-4.
- Checkpoints are written every 500 optimizer steps and at the final step; only the newest three complete checkpoints are retained.
- Sorted parquet shards are assigned by shard_index modulo 8. Each epoch resets the same fixed assignment. Unconsumed shard tails are accepted; data exhaustion before 9,244 steps is a distributed hard failure.
- Every GPU job copies the full persistent parquet directory once into a run-specific /dev/shm directory. All ranks read that same staged copy. Source and staged parquet footer, row count, byte size, and SHA256 inventories must match before torchrun starts.

## CPU-only conversion in the terminal

Do not submit conversion as a GPU job:

    cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
    conda activate rsmol
    bash scripts/convert_stepwise_5_10x4_5_mesh.sh

The default output is /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x4-5-mesh.

Optional CPU parquet preflight:

    python scripts/audit_text_parquet_store_5_10x4_5_mesh.py \
      --data-dir /hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset/data \
      --report-path /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/preflight/source_inventory.json

## pdgpu-4090 submissions

    bash run_audit_stage1_5_10x4_5_mesh_4090.sh
    bash run_stage4_5_10x4_5_mesh_smoke_4090.sh

After smoke produces checkpoint-000010, run the two-step resume gate:

    export RSMOL_5_10X4_5_MESH_RESUME_FROM=/path/to/smoke/checkpoint-000010
    bash run_stage4_5_10x4_5_mesh_resume_4090.sh

Formal training:

    bash run_stage4_5_10x4_5_mesh_formal_4090.sh

Local static checks do not count as remote CUDA or training PASS. Use the Stage 1 audit JSON and each training gate report as the remote result of record.
