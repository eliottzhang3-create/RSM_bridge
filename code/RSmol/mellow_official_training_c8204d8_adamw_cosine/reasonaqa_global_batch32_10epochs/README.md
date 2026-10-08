# Full ReasonAQA: global batch 32, 10 epochs

This directory is an isolated copy of the original full ReasonAQA Mellow AdamW/cosine training runtime. It does not use the MCQ manifest or MCQ training scripts. The model, dataset, trainer, and loss implementation are copied without changes. The existing loss sums answer-token cross-entropies and divides by the globally reduced count of non-padding answer tokens.

The formal route fixes 8 GPU ranks, per-rank batch size 4, gradient accumulation 1, and 10 full epochs, for an effective global batch of 32. It retains the original full ReasonAQA audited data mapping, 5090 resource request, AdamW betas (0.9, 0.95), learning rate 1e-3 to 5e-5 with step cosine and 5% warmup, weight decay 1e-4, FP32 training, and per-epoch checkpoints.

From the remote repository's `code/RSmol` directory, submit with:

```bash
bash run_mellow_official_reasonaqa_adamw_cosine_gbs32_10epochs_5090.sh
```

Optional positional arguments are the audited `path_audit.json`, `train_audio_mapping.jsonl`, output directory, and resume checkpoint, in that order. The default output directory is isolated under `outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs32_5090/`. The launcher stages raw audio under `/dev/shm`, writes `runtime_10epochs.yaml` and `staging_report.json` to the output directory, and checks the final schema-v2 checkpoint.
