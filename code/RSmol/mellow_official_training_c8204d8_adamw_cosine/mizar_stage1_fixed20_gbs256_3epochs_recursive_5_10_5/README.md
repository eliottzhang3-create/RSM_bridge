# HTSAT / 5-10-5 non-MeSH recursive Mizar S1 GBS256

This is an isolated copy of the Mizar S1 GBS256 baseline. It changes the
language-model initialization to the validated non-MeSH 5+10x2+5 recursive
checkpoint. Data selection, online audio processing, staging, sequence layout,
loss reduction, optimizer, scheduler, GradNorm, checkpointing, and resume
cursor remain the baseline implementation.

The local checkout contains code only. The manifest, resolved path audit,
HTSAT weights, recursive text checkpoint, and all audio files are remote
assets on the compute cluster.

## Fixed data and geometry

```text
manifest:
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json

resolved paths:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
```

The route retains 1,477,492 rows: MusicCaps 3,221 rows and all SpeechCraft
splits, including `train-other-500`. It does not re-audit, re-filter, or alter
the audio mapping. Staging copies raw bytes only; decoding, resampling, fixed20
and HTSAT remain online in training. Shared-path startup, background copy and
the eight-rank collective switch are unchanged.

The fixed training geometry is 8 GPUs, microbatch 8 per rank, accumulation 4,
global batch 256, and 3 epochs. This gives 5,771 optimizer updates per epoch,
17,313 total updates, 866 warmup updates, and 116 dropped rows per epoch.

## Model contract

The text checkpoint is:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244
```

`models/recursive_text.py` imports the shared
`code/RSmol/recursive_model_5_10_5.py` explicitly, without global registration, and
loads it with `local_files_only=True`. It rejects any checkpoint that does not
match:

```text
hidden_size=576
logical_layers=30
physical_layers=20
recursive_loops=2
memory_slots=0
router_modules=0
logical_to_physical=0..14, 5..14, 15..19
recursive_loops_scope=middle_only
```

The full recursive LM is trainable and LoRA is disabled. It requires twenty
independent physical layers and rejects router, memory-slot, or MeSH state.
HTSAT remains frozen;
the baseline c2l/projection and parameter-role grouping remain unchanged.

The tokenizer is loaded from the recursive checkpoint itself. The route
requires `!` as PAD, PAD distinct from EOS/SEP, EOS/SEP ID 0, a one-token
encoding for `!`, and tokenizer rows equal to LM embedding rows. Runtime
padding follows Mizar exactly: it calls `add_special_tokens({'pad_token':
'!'})`, derives the ignore index from `encode(tokenizer.pad_token)[0]`, and
uses `padding='max_length'` after appending the EOS text. A source checkpoint
may serialize `<|endoftext|>` as its tokenizer PAD; that metadata is
overridden by the same Mizar operation and is recorded only for audit.

The remote CPU-only comparison confirmed that the canonical SmolLM2 and
`checkpoint-009244` token-to-ID dictionaries are exactly equal (SHA256
`d59adeae1bb1c383e51bcf30fd8ad05a2b82f7e752f72106f0f94c324cdb44b6`).
In that environment, `add_special_tokens` returns 1 while the tokenizer length
stays unchanged. This return value must not be treated as vocabulary growth.
The audit instead compares tokenizer lengths and `!` IDs before/after the
Mizar assignment, checks the checkpoint's actual `config.vocab_size`, and
later checks the loaded LM embedding rows. `CHECKPOINT_TOKENIZER_PASS` logs
the source PAD metadata, API return value, and actual vocabulary growth.
The earlier tokenizer failures happened before training; the remote CPU check
does not establish a successful GPU training run.

## Sequence and optimization

```text
HTSAT audio1[129] + audio2[129] + SEP[1] + prompt[256] + answer[300]
prefix=515, total=815, attention_mask=None, labels=None
```

There is one separator after the contiguous audio prefix. The external loss is
the baseline answer-only cross entropy with rank-local microbatch token means,
division by four during accumulation, and DDP mean reduction.

The optimizer contract is unchanged:

```text
mapper/c2l/projection LR = 6e-4
recursive LM LR           = 3e-4
AdamW betas               = (0.9, 0.999)
eps                       = 1e-8
weight decay              = 0.01 (bias and ndim <= 1 excluded)
LambdaLR                  = 5% warmup + cosine, floor 0.1
GradNorm                  = initial_l2 1, initial_max 10, overdrive 2.5, momentum .995
```

The MeSH route's two-SEP/389-prefix layout, global-token-mean loss, alternate
AdamW betas, and alternate StepCosine settings are intentionally not used.

## CPU preflight

After synchronizing this checkout to the remote Linux server, run the existing
CPU-only checkpoint audit before requesting GPUs:

```bash
python code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5/audit_tokenizer.py \
  --reference-tokenizer <canonical-SmolLM2-tokenizer-directory> \
  --checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244 \
  --output-report /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_recursive_5_10_5_5090/text_tokenizer_audit.json
```

The route's `write_config.py` repeats the config, weight-file, tokenizer-file,
EOS and schedule checks after staging, then records all recursive checkpoint
metadata and hashes in `runtime.json`.

## Formal submission

From the remote repository's `code/RSmol` directory:

```bash
bash run_mizar_s1_gbs256_recursive_5_10_5_5090.sh \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_recursive_5_10_5_5090/formal_recursive_5_10_5_$(date +%Y%m%d_%H%M%S%N)"
```

The wrapper uses the same `pdgpu-5090` container, 32 CPUs, 256G memory and 8
GPUs as the baseline. The default output root is
`mizar_s1_htsat_gbs256_recursive_5_10_5_5090`; the final checkpoint is:

```text
<output-dir>/formal/checkpoints/model--epo-3.ckpt
```

Intermediate checkpoints are written at steps 2,886, 5,771, 8,656, 11,542,
14,428 and 17,313. The route identity is
`htsat_mizar_s1_fixed20_gbs256_recursive_5_10_5_v1`; baseline and MeSH checkpoints are
rejected by strict route and contract checks.

The current repository has no verified evaluator for this new recursive route.
Evaluation should be added only after a successful training report and must
load this route's runtime contract and checkpoint identity.
