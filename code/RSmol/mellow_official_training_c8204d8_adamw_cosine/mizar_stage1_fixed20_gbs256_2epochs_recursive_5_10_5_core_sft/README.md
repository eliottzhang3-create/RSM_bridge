# Non-MeSH 5-10-5 Core SFT baseline

This isolated route is the two-epoch Mizar Stage 1 HTSAT/audio training
baseline with the non-MeSH recursive 5+10x2+5 text model. The only trainable
parameters are physical text layers 5..14. The fixed schedule executes those
ten shared layers twice: `0..14, 5..14, 15..19`.

All data selection, online fixed20 audio processing, staging, sequence layout,
Mizar padding, answer-only loss, DDP reduction, AdamW grouping, LambdaLR,
GradNorm, checkpointing, and resume behavior follow the isolated Mizar S1
GBS256 implementation. This route changes the text initialization and the
explicit trainability mask only.

## Remote assets

```text
manifest:
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json

resolved paths:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv

text model:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244

full model-only initialization:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_recursive_5_10_5_5090/formal_20261025_160348996763174/formal/checkpoints/model--step-17313.ckpt
```

The initialization checkpoint is loaded strictly as model tensors. Its
optimizer, scheduler, GradScaler, GradNorm state, RNG state, and data cursor
are deliberately not resumed. The source must be route
`htsat_mizar_s1_fixed20_gbs256_recursive_5_10_5_v1` at step `17313`.

## Fixed training contract

```text
rows                         1,477,492
GPUs                         8
microbatch per rank          8
gradient accumulation        4
global batch                 256
epochs                       2
optimizer updates            11,542 (5,771 per epoch)
warmup updates               577
dropped rows per epoch       116
max/min LR                   5e-5 / 5e-6
```

The sequence is `audio1[129] + audio2[129] + SEP[1] + prompt[256] +
answer[300]`, for prefix length 515 and total length 815. Mizar padding is
visible to the model because `attention_mask=None`; external CE masks only
answer PAD targets and includes the answer EOS target.

Frozen modules are kept in evaluation mode, including projection dropout:
HTSAT, c2l, projection, embeddings, physical layers 0..4 and 15..19, final
norm, and lm_head. Frozen layers remain in the autograd graph so the active
middle layers receive gradients. The first real update has an execution audit
that verifies all 20 physical layers and both occurrences of layers 5..14.

## CPU checks

After syncing this directory to the remote Linux checkout, run the tokenizer
audit against a canonical SmolLM2 tokenizer if desired:

```bash
python code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_2epochs_recursive_5_10_5_core_sft/audit_tokenizer.py \
  --reference-tokenizer <canonical-SmolLM2-tokenizer-directory> \
  --checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244
```

The runtime config repeats the non-MeSH config, tokenizer, weight-file,
schedule, and vocabulary checks before any GPU training starts.

## Formal submission

From the remote repository's `code/RSmol` directory:

```bash
bash run_mizar_s1_recursive_5_10_5_core_sft_gbs256_5090.sh \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_recursive_5_10_5_core_sft_gbs256_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

The launcher requests the same `pdgpu-5090` container, 32 CPUs, 256G memory,
and 8 GPUs. The final checkpoint is:

```text
<output-dir>/formal/checkpoints/model--epo-2.ckpt
```

Intermediate checkpoints are written at six evenly spaced steps, ending at
`11,542`. A resume requires a complete checkpoint produced by this same route
and resolved contract.

