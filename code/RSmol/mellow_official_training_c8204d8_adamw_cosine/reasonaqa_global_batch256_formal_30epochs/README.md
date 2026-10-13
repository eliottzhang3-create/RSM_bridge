# Standard Mellow ReasonAQA, 30 epochs

This isolated training route overlays only the files changed for the formal run. Unmodified modules and raw-audio staging come from `../reasonaqa_global_batch256_lr_ablation_5epochs` through `PYTHONPATH`. Model weights and data remain on the remote Linux server.

- Eight GPUs, microbatch eight per GPU, accumulation four: effective global batch 256.
- AdamW, betas `(0.9, 0.95)`, weight decay `1e-4`, FP32. All trainable bridge and LM parameters share a step-level 5% warmup and cosine learning rate from `2e-4` to `1e-5`.
- One global L2 gradient norm clip of 0.5 after accumulation and unscale. HTSAT is frozen in eval/no-grad mode; `c2l`, audio projection, and LM remain trainable.
- When source `filepath2` is empty, the second slot is a zero waveform; audio-read failure fallback is unchanged.
- Text input is prompt through EOS, then answer through EOS, with right padding only after the answer EOS. Audio prefix length is 260. The causal attention mask excludes trailing padding. CE targets only answer tokens and answer EOS, normalized by the global count of valid targets over all ranks and accumulation microbatches.
- Prompt and answer caps remain 129 and 250 tokens including EOS. Truncation preserves EOS.

Submit from the remote `code/RSmol` directory after synchronizing code through GitHub:

```bash
bash run_mellow_official_reasonaqa_gbs256_formal30_5090.sh
```

The wrapper uses the existing PASS path audit and audio mapping defaults, stages raw audio to `/dev/shm`, submits to `pdgpu-5090`, and creates a separate output directory. Use `--audit-report`, `--mapping-jsonl`, or `--output-dir` to override those paths. An interrupted run can be submitted with both `--output-dir` pointing to its original run directory and `--resume-checkpoint` pointing to its schema-v2 checkpoint.
