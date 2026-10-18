# Full ReasonAQA: LR5-based 10-epoch LR Ablation

Rebuilt on 2026-10-18 by copying the complete sibling directory
`reasonaqa_global_batch256_lr_ablation_5epochs`. Training behavior differs
from that source only in the following three respects:

1. Train for 10 epochs, with the same 5% warmup/cosine schedule calculated
   across the full 10-epoch horizon.
2. Use `models/mellow.py` unchanged from `reasonaqa_global_batch256_htsat_frozen_5epochs`:
   HTSAT stays in eval mode and its backbone forward runs under no_grad;
   c2l and projection remain trainable.
3. Use the hard global L2 gradient clip from
   `reasonaqa_global_batch256_global_clip_5epochs`, with threshold 0.5 after
   accumulation and unscale, including its checkpoint/resume handling.

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

All other training settings retain LR5: 8 ranks, microbatch 8 per rank,
accumulation 4, global batch 256, FP32, seed 1234, global valid-answer-token
mean CE, AdamW betas (0.9, 0.95), weight decay 1e-4, and one schema-v2
checkpoint per epoch. Fresh runs initialize from the same SmolLM2/HTSAT
sources. `--max-lr` is required and `min_lr=0.1*max_lr`. With the documented
968,059 examples, the expected schedule is 3,781 updates/epoch, 37,810 total
updates and 1,891 warmup updates; the trainer calculates from actual data.

After syncing code through GitHub, submit from the remote `code/RSmol`:

```bash
bash run_mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090.sh --max-lr 1e-3
```

Replace `1e-3` with the desired max LR. Resources remain
`pdgpu-5090 -c 32 -m 256G -g 8 -n 1`, the same container and
`mellow_c8204d8` environment. Each submission gets an independent directory:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
  mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090/
    formal_10epochs_lr5_frozen_clip_maxlr_<lr>_<timestamp>/
      runtime_10epochs.yaml
      staging_report.json
      checkpoints/<job-id>/model--epo-1.ckpt ... model--epo-10.ckpt
```

Optional arguments remain `--audit-report`, `--mapping-jsonl`, `--output-dir`
and `--resume-checkpoint`. Resume only from this rebuilt route with the same
10-epoch horizon, LR and batch geometry; the route identifier is
`lr_ablation_10epochs_htsat_frozen_global_clip`. It excludes old compact and
earlier masked-prompt attempts. The job stages audio in its own `/dev/shm`
directory and cleans that directory on exit. Validation does not pin a compute
node or require the previous job's staging directory to exist.

This rebuild changes training only. The unified evaluator remains at its
previous compact-route contract and needs separate adaptation for these new
checkpoints.

Local validation: all 7 replication/contract checks passed, including the
complete file list, byte equality of unchanged LR5 files, exact frozen HTSAT
reference, global-clip trainer equivalence and the LM receiving no padding
mask. All 21 Python sources (including the test file), 3 embedded Python
blocks and both Bash entry scripts passed syntax checks. A mocked submission
verified the route/resources/arguments without submitting a job; the final
checkpoint audit accepted a valid fixture and rejected incorrect route,
clipping, HTSAT policy and epoch. The copied LR5 dataset/decoder retain the
source's whitespace to preserve byte equality. No smoke or remote training
was run; PyTorch, weights and data are absent locally.
