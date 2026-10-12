# ReasonAQA Mellow grouped bridge/LM learning-rate ablation

This route is an isolated five-epoch copy of the official global-token-mean ReasonAQA Mellow AdamW/cosine training route. It keeps the baseline data path, audio-failure behavior, attention/loss implementation, batch geometry, FP32, seed, AdamW betas (0.9, 0.95), weight decay 1e-4, max_grad_norm=0.5, 5% optimizer-step warmup, and checkpoint contract unchanged.

Only the optimizer learning rate is changed to two groups:

- bridge: trainable audio_encoder.base.c2l.* and audio_encoder.projection.*;
- lm: trainable caption_decoder.lm.*.

The frozen HTSAT trunk is not placed in either group. Each group uses min_lr = 0.1 * max_lr and its own warmup/cosine value at every optimizer step. The two intended runs are:

| run | bridge max/min | LM max/min |
|---|---:|---:|
| A | 4e-4 / 4e-5 | 1e-4 / 1e-5 |
| B | 8e-4 / 8e-5 | 1e-4 / 1e-5 |

From the remote repository's code/RSmol directory:

    bash run_mellow_official_reasonaqa_adamw_cosine_gbs256_bridge_lm_lr_5epochs_5090.sh --bridge-max-lr 4e-4 --lm-max-lr 1e-4
    bash run_mellow_official_reasonaqa_adamw_cosine_gbs256_bridge_lm_lr_5epochs_5090.sh --bridge-max-lr 8e-4 --lm-max-lr 1e-4

The wrapper submits vc submit jobs to pdgpu-5090 with the same resource contract as the reference route. The inner script reuses the reference route raw-audio staging implementation, preserving the required /dev/shm basename prefix mellow_adamw_cosine_reasonaqa_gbs256_lr_. It writes a runtime YAML, staging report, and schema-v2 checkpoints, then audits both LR-group bounds in the final checkpoint. Evaluation is intentionally outside this route.

Optional wrapper arguments are --audit-report, --mapping-jsonl, --output-dir, and --resume-checkpoint. A resume checkpoint must come from the same grouped-LR route and matching two LR values; optimizer and scheduler group contracts are checked before restoring state.
