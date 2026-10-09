# Full ReasonAQA: Equal Microbatch Token Mean Loss Ablation

This is an isolated five-epoch full ReasonAQA training route. It uses the
audited full ReasonAQA train manifest and raw-audio mapping, not MCQ. The
model, optimizer, data order, batch geometry, and scheduler match the
global-token-mean route. The only training reduction change is the historical
per-microbatch answer-token mean:

```text
L = (1 / (R * G)) * sum_r sum_g [ sum_t CE[r,g,t] / N[r,g] ]
```

Each microbatch loss is divided by its own valid answer-token count, then by
the four accumulation steps for backpropagation. DDP averages the resulting
gradients across the eight ranks. The logged loss is the corresponding average
over ranks and accumulation microbatches; it is not the global valid-token
weighted mean used by the baseline route. Checkpoints identify this contract as
`equal_microbatch_token_mean` and cannot resume a `global_token_mean` run.

The route is fixed to 8 ranks, per-rank microbatch 8, accumulation 4, global
batch 256, five epochs, FP32, AdamW betas (0.9, 0.95), weight decay 1e-4,
5% warmup, step-level cosine decay, max_lr=1e-3, and min_lr=1e-4.

From the remote repository's `code/RSmol` directory, submit a fresh run with:

```bash
bash run_mellow_official_reasonaqa_adamw_cosine_gbs256_old_microbatch_mean_1e-3_5epochs_3090.sh --max-lr 1e-3
```

The default output root is
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_old_microbatch_mean_3090/`.
The wrapper stages raw audio into a job-local `/dev/shm` directory and audits
the final schema-v2 `model--epo-5.ckpt`, including the loss reduction, batch
geometry, optimizer, and scheduler contracts. Optional arguments are
`--audit-report`, `--mapping-jsonl`, `--output-dir`, and
`--resume-checkpoint`; any resume checkpoint must come from this same route.
