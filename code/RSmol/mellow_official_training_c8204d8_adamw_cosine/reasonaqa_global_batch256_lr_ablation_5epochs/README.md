# Full ReasonAQA: Global Batch 256 Learning-Rate Ablation

This is an isolated five-epoch training copy of the original full ReasonAQA
Mellow AdamW/cosine route. It uses the audited full ReasonAQA train JSON and
raw-audio mapping, not the MCQ manifest. The model initializes from the same
SmolLM2 and frozen HTSAT sources for every fresh run; no checkpoint is used for
initialization. Use the original route for evaluation.

The comparison holds the original 8 GPU ranks, per-rank microbatch 8,
gradient accumulation 4, effective global batch 256, FP32, seed 1234,
global valid-answer-token mean cross-entropy, AdamW betas (0.9, 0.95),
weight decay 1e-4, and per-epoch schema-v2 checkpoints. The schedule is a
linear 5% warmup followed by optimizer-step cosine decay across all five
epochs. `max_lr` is required at submission; `min_lr` is always 0.1 times it.

For the currently documented 968,059 training examples, the expected schedule
is 3,781 optimizer steps per epoch, 18,905 steps total, and 946 warmup steps.
The trainer computes these values from the actual staged training data.

From the remote repository's `code/RSmol` directory, submit each experiment
independently:

```bash
bash run_mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5epochs_5090.sh --max-lr 2e-3
bash run_mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5epochs_5090.sh --max-lr 1e-3
bash run_mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5epochs_5090.sh --max-lr 8e-4
bash run_mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5epochs_5090.sh --max-lr 4e-4
bash run_mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5epochs_5090.sh --max-lr 2e-4
```

The corresponding minimum learning rates are `2e-4`, `1e-4`, `8e-5`,
`4e-5`, and `2e-5`. Each submission gets its own LR-tagged directory under
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5090/`.
The job restages raw audio into a job-local `/dev/shm` directory, writes
`staging_report.json` and `runtime_5epochs.yaml`, and audits its final
`model--epo-5.ckpt` for the submitted LR, five-epoch schedule, batch geometry,
and loss contract.

Optional arguments are `--audit-report`, `--mapping-jsonl`, `--output-dir`,
and `--resume-checkpoint`. A resume must use a schema-v2 full checkpoint from
the same five-epoch schedule and learning rate; the trainer validates its
optimizer, scheduler, batch geometry, and epoch horizon.
