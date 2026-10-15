# Text 5-10x4-5 MeSH isolated route

This route is isolated from every audio route and from the existing text 5-10x2-5 implementation.

## Fixed contract

- 20 physical decoder layers and 50 logical executions: 5 + 10 * 4 + 5.
- Seven transient MeSH memory slots.
- Five independent router groups. Each group owns one write router and one read router, for ten independent linear router modules.
- Source layer mapping remains identical to the prior 5-10x2-5 conversion: 0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29.
- Eight ranks, microbatch 4, gradient accumulation 32, effective global batch 1024, context length 1024. The smaller microbatch lowers per-rank activation memory while preserving the optimizer-step batch.
- Formal training runs for one third of a fixed 9,244-step epoch: exactly 3,081 new optimizer steps (9244 // 3), then stops. Rank assignment, first-epoch shard order and row order within each shard stay fixed.
- LR warms for 155 steps (ceil(5% * 3081)) to 2e-4, then uses the existing cosine formula with min LR 1e-5 over the 3,081-step target.
- Initialize model weights in FP32 from /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/formal_2epochs_18488steps_20261007_004959_3090_v1/checkpoint-018488. Load the tokenizer from the same directory by default. Create a new AdamW and schedule from step 0; do not restore the historical optimizer, scheduler, RNG or data cursor.
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

The resume gate accepts only a checkpoint produced by the current FP32 direct-read smoke contract. Formal resume also checks the new 3,081-step scheduler contract. Historical BF16, one-epoch or two-epoch checkpoints may supply model weights through --model-path, but cannot supply full training state through --resume-from for the new formal run.

Formal training:

    bash run_stage4_5_10x4_5_mesh_formal_3090.sh

Formal output is formal_continue_3081steps_fp32_from018488_<timestamp>_3090_v1 by default. The formal wrapper explicitly selects the checkpoint-018488 initialization above, clears any inherited resume path, and sets max LR 2e-4 and min LR 1e-5. The final checkpoint is checkpoint-003081. Startup logs and the gate report record weights_only_new_training_state with starting_optimizer_step=0.

The historical run_stage4_5_10x4_5_mesh_formal_4090.sh calls the same trainer and now uses the same 3,081-step FP32 continuation contract, with pdgpu-4090 and a separate timestamped output name. The current production entry remains the 3090 wrapper above.

Local static checks do not count as remote CUDA or training PASS. Use the Stage 1 audit JSON and each training gate report as the remote result of record.

## Single-GPU Stage 3 text evaluation

From the remote code/RSmol directory, submit:

```bash
bash run_stage3_eval_5_10x4_5_mesh_5090.sh
```

This uses pdgpu-5090 with one GPU, 8 CPUs and 32G memory. The default checkpoint is /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/formal_2epochs_18488steps_20261007_004959_3090_v1/checkpoint-018488. The evaluator loads the original seven-slot x4 model and checks 20 physical layers, 50 logical executions/cache slots, and five router groups (ten modules).

Evaluation matches run_stage3_eval_5_10x2_5_mesh_5090.sh: BF16, cuda:0, batch size 1, seed 0, all samples, and HellaSwag/MMLU/GSM8K/ARC-Easy/ARC-Challenge. MMLU and GSM8K use five shots. Both routes share offline dataset overlays and official task prompt/generation/metric semantics, with lm_eval==0.4.12, transformers==4.54.1 and datasets==3.6.0. The two-token generation setting applies only to runtime preflight; GSM8K keeps the task YAML's generation settings. FP32 training settings do not change this comparison's BF16 evaluation.

The isolated implementation is scripts/evaluate_stage3_5_10x4_5_mesh.py plus its .sh runtime. Model/path/task controls use RSMOL_STAGE3_5_10X4_5_MESH_* environment variables, forwarded by the submit wrapper. Defaults write results to /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage3_eval_5_10x4_5_mesh_018488, with separate eval_logs/stage3_5_10x4_5_mesh_018488 and eval_cache/stage3_5_10x4_5_mesh_018488 directories. Read audit_report.json, summary.json/summary.csv, per-task lm_eval_results.json and task_protocol.json to assess a completed run. Local checks do not establish remote benchmark results.
