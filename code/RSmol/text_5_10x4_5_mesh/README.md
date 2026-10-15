# Text 5-10x4-5 MeSH isolated route

This route is isolated from every audio route and from the existing text 5-10x2-5 implementation.

## Fixed contract

- 20 physical decoder layers and 50 logical executions: 5 + 10 * 4 + 5.
- Seven transient MeSH memory slots.
- Five independent router groups. Each group owns one write router and one read router, for ten independent linear router modules.
- Source layer mapping remains identical to the prior 5-10x2-5 conversion: 0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29.
- Eight ranks, microbatch 4, gradient accumulation 32, effective global batch 1024, context length 1024. The smaller microbatch lowers per-rank activation memory while preserving the optimizer-step batch.
- Formal training runs one epoch for exactly 9,244 optimizer steps, then stops. Rank assignment, first-epoch shard order and row order within each shard stay fixed.
- LR warms for 463 steps (ceil(5% * 9244)) to 8e-4, then uses the existing cosine formula with min LR 2e-5 over the 9,244-step target.
- Training forces float32 parameters and forward/backward execution. BF16 autocast and TF32 are disabled; startup diagnostics and checkpoint metadata record the FP32 contract.
- Checkpoints are written every 500 optimizer steps and at the final step; only the newest three complete checkpoints are retained.
- Sorted parquet shards are assigned by shard_index modulo 8. Each rank streams its assigned persistent parquet shards directly from the remote login-mounted data directory. Epoch boundaries are defined by optimizer steps, not by parquet exhaustion. Checkpoints preserve the epoch-specific shard order and row cursor for resume.
- Production training does not copy the dataset into /dev/shm; the shared-memory staging script remains available only as an independent diagnostic tool.

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

## Validated pdgpu-3090 training submissions

The current production submission route uses pdgpu-3090. The historical allocation checks used this sequence; the new FP32 configuration still requires remote execution to establish its memory use and training result:

    bash run_audit_same_allocation_text_5_10x4_5_mesh_3090.sh
    bash run_stage4_5_10x4_5_mesh_smoke_3090.sh

After smoke produces checkpoint-000010, run the two-step resume gate:

    export RSMOL_5_10X4_5_MESH_RESUME_FROM=/path/to/smoke/checkpoint-000010
    bash run_stage4_5_10x4_5_mesh_resume_3090.sh

The resume gate accepts only a checkpoint produced by the current FP32 direct-read smoke contract. Formal resume also checks the one-epoch scheduler contract; historical BF16 or two-epoch checkpoints cannot resume the new formal run.

Formal training:

    bash run_stage4_5_10x4_5_mesh_formal_3090.sh

Formal output is formal_1epoch_9244steps_fp32_<timestamp>_3090_v1 by default. It initializes from /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x4-5-mesh unless an explicit model path is supplied. The formal wrapper clears any inherited resume path and explicitly sets max LR 8e-4 and min LR 2e-5.

The historical run_stage4_5_10x4_5_mesh_formal_4090.sh calls the same trainer and now uses the same one-epoch FP32 contract, with pdgpu-4090 and a separate timestamped output name. The current production entry remains the 3090 wrapper above.

Local static checks do not count as remote CUDA or training PASS. Use the Stage 1 audit JSON and each training gate report as the remote result of record.
