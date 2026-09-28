# Text 5-10x6-5 MeSH isolated route

This route is isolated from every audio route and from the existing text 5-10x2-5 and 5-10x4-5 implementations.

## Fixed contract

- 20 physical decoder layers and 70 logical executions: 5 + 10 * 6 + 5.
- Seven transient MeSH memory slots.
- Seven independent router groups. Each group owns one write router and one read router, for fourteen independent linear router modules.
- Source layer mapping remains identical to the prior 5-10x2-5 conversion: 0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29.
- Eight ranks, microbatch 4, gradient accumulation 32, effective global batch 1024, context length 1024. The smaller microbatch lowers per-rank activation memory while preserving the optimizer-step batch.
- The historical reference epoch is 9,244 optimizer steps; this route trains exactly one third with 3,081 optimizer steps (floor(9,244 / 3)), using one configured epoch.
- LR warms for 155 steps to 1e-3, then cosine decays to 1e-4 over the 3,081-step target.
- Checkpoints are written every 500 optimizer steps and at the final step; only the newest three complete checkpoints are retained.
- Sorted parquet shards are assigned by shard_index modulo 8. Each rank streams its assigned persistent parquet shards directly from the remote login-mounted data directory, matching the validated 5-10x2-5 text trainer. The one-third target stops before data exhaustion.
- Production training does not copy the dataset into /dev/shm. The x6 production route contains no shared-memory staging dependency.

## CPU-only conversion in the terminal

Do not submit conversion as a GPU job:

    cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
    conda activate rsmol
    bash scripts/convert_stepwise_5_10x6_5_mesh.sh

The default output is /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x6-5-mesh.

Optional CPU parquet preflight:

    python scripts/audit_text_parquet_store_5_10x6_5_mesh.py \
      --data-dir /hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset/data \
      --report-path /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x6_5_mesh/preflight/source_inventory.json

## pdgpu-3090 qualification and training submissions

Run the CUDA architecture audit first, then the ten-step smoke:

    bash run_audit_stage1_5_10x6_5_mesh_3090.sh
    bash run_stage4_5_10x6_5_mesh_smoke_3090.sh

After smoke produces checkpoint-000010, run the two-step resume gate:

    export RSMOL_5_10X6_5_MESH_RESUME_FROM=/path/to/smoke/checkpoint-000010
    bash run_stage4_5_10x6_5_mesh_resume_3090.sh

The resume gate accepts only a checkpoint produced by this direct-read x6 smoke contract. Checkpoints from x2, x4, audio, or staged-data contracts are rejected by the architecture and training contracts.

Formal training:

    bash run_stage4_5_10x6_5_mesh_formal_3090.sh

Formal output is formal_third_epoch_3081steps_20260928_3090_v1 by default. A fresh
formal run starts from the converted x6 model rather than continuing the smoke
checkpoint.

To continue an interrupted formal run, use the same formal wrapper with a complete
formal checkpoint and a new output directory:

    export RSMOL_5_10X6_5_MESH_RESUME_FROM=/path/to/formal/checkpoint-002000
    export RSMOL_5_10X6_5_MESH_OUTPUT_DIR=/path/to/new/formal_resume_output
    bash run_stage4_5_10x6_5_mesh_formal_3090.sh

The formal wrapper validates the completion marker, manifest, and training state,
then forwards the checkpoint into the submitted FORMAL job. The separate
run_stage4_5_10x6_5_mesh_resume_3090.sh wrapper remains restricted to the smoke
10-to-12 resume gate.

## Resume cursor and nonfinite-gradient policy

Formal resume restores each rank's exact shard and row offset. Rows already consumed
inside the current shard are skipped before the first resumed microbatch is yielded.
New checkpoints also save each rank's CUDA RNG state; older checkpoints remain
loadable when that optional field is absent.

Gradient clipping remains fixed at global norm 1.0. If an accumulated data window
produces a NaN or infinite gradient norm, all ranks synchronously discard that entire
window, keep the optimizer and scheduler step unchanged, clear the gradients, and
consume the next data window for the same optimizer step. Therefore formal training
still completes exactly 3,081 successful optimizer updates; skipped numerical windows
do not count as optimizer steps. Every skip is written to
ddp_diagnostics/rank<rank>.nonfinite_skips.jsonl and summarized in the final report.

To avoid hiding a permanently corrupted model, the default safety limits are 32
consecutive skipped windows and 256 total skipped windows. They may be adjusted with
RSMOL_5_10X6_5_MESH_MAX_CONSECUTIVE_NONFINITE_WINDOWS and
RSMOL_5_10X6_5_MESH_MAX_TOTAL_NONFINITE_WINDOWS.

Local static checks do not count as remote CUDA or training PASS. Use the Stage 1 audit JSON and each training gate report as the remote result of record.
