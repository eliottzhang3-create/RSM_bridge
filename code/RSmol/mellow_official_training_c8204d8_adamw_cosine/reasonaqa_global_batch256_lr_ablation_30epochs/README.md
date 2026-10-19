# Full ReasonAQA: LR5-based 30-epoch Run at max_lr=4e-5

Created by copying the complete sibling directory
`reasonaqa_global_batch256_lr_ablation_10epochs`. It preserves that route and
changes only the training horizon and requested LR:

1. Train for 30 epochs with the existing 5% warmup/cosine schedule calculated
   across the complete 30-epoch horizon.
2. Use `max_lr=4e-5` and `min_lr=4e-6`.

All other behavior is copied from the 10-epoch route: HTSAT remains in
eval/no_grad while c2l and projection remain trainable; global L2 gradient
clipping stays at 0.5 after accumulation and unscale. Dataset, collate,
decoder, padding, causal attention, random second-audio fallback, checkpoint
state, optimizer, loss, seed and resource settings remain the same.

The dataset, collate function and decoder are exact copies of LR5. Prompt
and answer are tokenized separately to 129 and 250 positions. Each text gets
` <|endoftext|>` appended before tokenization, then the original truncation
and fixed padding apply; EOS can be truncated for long text. The sequence is
260 audio positions, 129 prompt positions, then 250 answer positions. The LM
receives `attention_mask=None`, exactly as in LR5: causal attention remains
active, with no additional padding mask. Prompt PAD is visible to later
answer positions. Answer PAD is excluded from loss and cannot affect earlier
tokens through causal attention. No EOS replacement or compact packing is added.
Missing audio slots retain LR5's random audio fallback.

Training settings retain the 10-epoch route: 8 ranks, microbatch 8 per rank,
accumulation 4, global batch 256, FP32, seed 1234, global valid-answer-token
mean CE, AdamW betas (0.9, 0.95), weight decay 1e-4, and one schema-v2
checkpoint per epoch. Fresh runs initialize from the same SmolLM2/HTSAT
sources. The wrapper keeps the explicit `--max-lr` interface; submit this run
with `4e-5`, and the runtime writer derives `min_lr=0.1*max_lr=4e-6`. With the
documented 968,059 examples, the expected schedule is 3,781 updates/epoch,
113,430 total updates and 5,672 warmup updates; the trainer calculates from
actual data.

After syncing code through GitHub, submit from the remote `code/RSmol`:

```bash
bash run_mellow_official_reasonaqa_gbs256_lr_ablation_30epochs_5090.sh --max-lr 4e-5
```

Resources remain
`pdgpu-5090 -c 32 -m 256G -g 8 -n 1`, the same container and
`mellow_c8204d8` environment. Each submission gets an independent directory:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
  mellow_official_reasonaqa_gbs256_lr_ablation_30epochs_5090/
    formal_30epochs_lr5_frozen_clip_maxlr_4e-5_<timestamp>/
      runtime_30epochs.yaml
      staging_report.json
      checkpoints/<job-id>/model--epo-1.ckpt ... model--epo-30.ckpt
```

Optional arguments remain `--audit-report`, `--mapping-jsonl`, `--output-dir`
and `--resume-checkpoint`. Resume only from this isolated route with the same
30-epoch horizon, LR and batch geometry; its route identifier is
`lr_ablation_30epochs_htsat_frozen_global_clip`. Cross-resume from the 10-epoch
route is rejected. The job stages audio in its own `/dev/shm` directory and
cleans it on exit. Validation does not pin a compute node or require a previous
staging directory to exist.

The isolated runner uses
`/dev/shm/mellow_adamw_cosine_reasonaqa_gbs256_lr_30epochs_frozen_clip_<run-id>`,
which meets the staging tool's required shared prefix. Staging validation and
cleanup behavior are copied unchanged.

This rebuild changes training only. The unified evaluator remains at its
previous compact-route contract and needs separate adaptation for these new
checkpoints.

Static tests compare every unchanged file with the 10-epoch source and cover
the new LR/epoch contract, stage name and final checkpoint audit. No smoke or
remote training is run as part of this change; verify the remote job log and
produced checkpoint for the actual training outcome.
