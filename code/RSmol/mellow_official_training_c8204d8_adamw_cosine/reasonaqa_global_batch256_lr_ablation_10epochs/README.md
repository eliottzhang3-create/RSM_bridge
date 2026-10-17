# Standard Mellow ReasonAQA, 10-epoch learning-rate ablation

This is an isolated copy of the formal30 SmolLM2 route. The formal30 training
implementation, text construction, audio prefix, masking, loss reduction,
optimizer, and distributed geometry remain unchanged except for the explicit
10-epoch horizon, the submitted learning-rate bounds, and the missing-second-
audio policy described below. Shared modules and raw-audio staging are loaded
from `../reasonaqa_global_batch256_lr_ablation_5epochs` through `PYTHONPATH`.
Weights and data remain on the remote Linux server.

- Eight GPUs, microbatch eight per GPU, accumulation four: effective global batch 256.
- AdamW, betas `(0.9, 0.95)`, weight decay `1e-4`, FP32, and global L2 hard clipping at `0.5`.
- All trainable bridge, projection, and LM parameters use one optimizer-step 5% warmup followed by cosine decay.
- `--max-lr` is required at submission; `min_lr` is always `0.1 * max_lr`.
- The run is exactly 10 epochs. The trainer computes the total optimizer steps and warmup steps from the staged data.
- HTSAT remains frozen in eval/no-grad mode. The formal30 compact prompt/answer EOS contract, right padding after answer EOS, causal attention mask, 260-position audio prefix, and global valid-answer-token mean loss are retained.
- When source `filepath2` is empty, the second slot is filled by `random.choice(exists_filepaths1)`, matching the historical LR-ablation route. Audio-read failure fallback is unchanged.

Submit from the remote `code/RSmol` directory after synchronizing code through
GitHub:

```bash
bash run_mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090.sh --max-lr 1e-3
```

Each learning rate receives an independent output directory under
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090/`.
The job stages raw audio into a job-local `/dev/shm` directory, writes
`staging_report.json` and `runtime_10epochs.yaml`, and audits the final
`model--epo-10.ckpt` for the submitted learning rate, 10-epoch schedule, batch
geometry, route policies, and loss contract.

Optional arguments are `--audit-report`, `--mapping-jsonl`, `--output-dir`, and
`--resume-checkpoint`. A resume must use a schema-v2 checkpoint from this route
with the same 10-epoch horizon and learning-rate bounds.
