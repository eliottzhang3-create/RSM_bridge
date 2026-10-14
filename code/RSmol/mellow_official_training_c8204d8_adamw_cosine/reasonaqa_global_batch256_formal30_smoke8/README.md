# Formal 30-epoch ReasonAQA: eight-step diagnostic run

This route uses the unchanged formal model, dataset, and audio staging implementations. Its own trainer adds diagnostics to the first row of the first microbatch on each of eight ranks for each of eight optimizer steps. It still trains with batch 8 per rank and accumulation 4 (GBS 256), and uses the full 30-epoch 5% warmup/cosine schedule. It starts from the pretrained HTSAT and SmolLM2 components. The formal 30-epoch route is not modified.

Submit from the remote `code/RSmol` directory after syncing the code through GitHub:

```bash
bash run_mellow_official_reasonaqa_gbs256_formal30_smoke8_5090.sh
```

Optional inputs are `--audit-report PATH`, `--mapping-jsonl PATH`, and `--output-dir EMPTY_PATH`. The wrapper submits one 8-GPU `pdgpu-5090` job and stages the raw audio into an isolated `/dev/shm` directory. The default output root differs from the formal run.

After a successful job, inspect `diagnostics/rank_0.jsonl` through `rank_7.jsonl` (eight lines each), `diagnostics/smoke8_audit.json` (64-record PASS), `runtime_smoke8.yaml`, and `checkpoints/<job-id>/model--step-8.ckpt`. Each JSONL line contains the waveform and audio-prefix evidence, actual text input IDs and token strings, the LM attention mask, per-position argmax predictions, the answer target for every supervised source position, and the loss and causal coverage ranges. It does not include waveform samples or the full source JSON object.

The attention ranges describe the causal mask implied by the LM and the actual key padding mask passed to it; they are not measured attention weights. The step-8 checkpoint records eight optimizer updates and the full 30-epoch scheduler horizon. This checkpoint is for diagnosis, not a completed 30-epoch model.
