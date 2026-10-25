# HTSAT / 5-10x4-5 MeSH Mizar S1 GBS128

This route is an isolated copy of the Mizar Stage 1 fixed20 GBS256 recursive
MeSH trainer. It replaces the initialized recursive text backbone from the
5-10x2-5 model with the 5-10x4-5 model. After the GBS256 run ran out of GPU
memory, the per-rank microbatch was reduced from 8 to 4 while accumulation
remains 4, giving GBS128. Data selection, online audio decoding,
fixed20 processing, staging, sequence layout, answer loss, optimizer,
scheduler, GradNorm, checkpointing, and resume semantics remain aligned with
the Mizar mother route.

The local checkout contains code only. The manifest, resolved path audit,
HTSAT weights, recursive text checkpoint, and audio files are remote assets on
the compute cluster.

## Fixed data and training geometry

```text
manifest:
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json

resolved paths:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
```

The route uses the audited 1,477,492-row mixture, including MusicCaps and all
SpeechCraft splits. Raw audio bytes are staged while decoding, resampling,
fixed20 cropping/padding and HTSAT processing remain online.

The fixed geometry is 8 GPUs, microbatch 4 per rank, accumulation 4, global
batch 128, and 3 epochs: 11,542 optimizer updates per epoch, 34,626 total,
1,731 warmup updates, and 116 dropped rows per epoch. Reducing the microbatch
reduces activation memory per forward/backward pass; reducing accumulation
alone would keep the same microbatch and would not address that peak. Model
parameters and optimizer-state memory are unchanged, so total peak memory
is not expected to halve. Learning-rate values and the 5% warmup/cosine rule
are unchanged; the schedule is recalculated for the new optimizer-step count.

The route directory and wrapper filename retain their historical `gbs256`
names for compatibility. Runtime/checkpoint identity, job name, logs, index
cache and default output directory now identify GBS128. Start in a new output
directory from the same text checkpoint; GBS256 audio-training checkpoints
cannot be resumed under the changed batch geometry and schedule.

## 5-10x4-5 text contract

The text checkpoint is:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/formal_2epochs_18488steps_20261007_004959_3090_v1/checkpoint-018488
```

`models/recursive_text.py` explicitly imports
`code/RSmol/recursive_model_5_10x4_5_mesh.py` and loads the checkpoint with
`local_files_only=True`. The loader and CPU contract audit reject any mismatch
in the following values:

```text
hidden_size=576
logical_layers=50
physical_layers=20
recursive_loops=4
logical_cache_slots=50
memory_slots=7
router_modules=10 (five independent write/read groups)
logical_to_physical=0..4, (5..14) repeated four times, 15..19
mesh_architecture_contract=logical_50_physical_20_5_10x4_5
```

The four-loop model must keep `ablate_after_two_loops=False`; MeSH memory is
transient and is not persisted in the attention cache. The complete recursive
LM is trainable, LoRA is disabled, and HTSAT remains frozen.

The tokenizer is loaded from this checkpoint and then receives Mizar's exact
`add_special_tokens({'pad_token': '!'})` operation. The route requires a
one-token `!` encoding, no vocabulary growth, PAD distinct from EOS/SEP, EOS
and the single audio/text separator at ID 0, and tokenizer rows equal to the
LM embedding rows. The checkpoint is an initialization source only; its Stage
4 optimizer, scheduler, RNG, cursor, and training state are never restored.

## Sequence and optimization

```text
HTSAT audio1[129] + audio2[129] + SEP[1] + prompt[256] + answer[300]
prefix=515, total=815, attention_mask=None, labels=None
```

The external Mizar loss averages valid answer tokens locally on each rank and
then uses the existing DDP reduction. Optimizer settings are unchanged:

```text
mapper/c2l/projection LR = 6e-4
recursive LM LR           = 3e-4
AdamW betas               = (0.9, 0.999)
eps                       = 1e-8
weight decay              = 0.01 (bias and ndim <= 1 excluded)
LambdaLR                  = 5% warmup + cosine, floor 0.1
GradNorm                  = initial_l2 1, initial_max 10, overdrive 2.5, momentum .995
```

## Remote preflight and formal submission

After synchronizing the repository to Linux, run the route's CPU-only static
tests and confirm that the remote checkpoint/config/tokenizer files exist. Run
the tests from the repository root because the test files live in `tests/`;
the training wrapper itself is submitted from `code/RSmol`.

```bash
cd "$(git rev-parse --show-toplevel)"
python -m unittest discover -s tests \
  -p 'test_mizar_s1_recursive_mesh_5_10x4_5_static.py' -v
```

The wrapper performs the prepared-index audit before submitting one formal job.
From the remote repository's `code/RSmol` directory:

```bash
bash run_mizar_s1_gbs256_recursive_mesh_5_10x4_5_5090.sh \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs128_recursive_mesh_5_10x4_5_3090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

The wrapper uses the current remote allocation: `pdgpu-3090`, 64 CPUs, 256G
memory and 8 GPUs. Its shortened job name stays below the 60-character limit.
It calls the
isolated `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_5_10x4_5/run.sh`
route and creates a unique x4 staging directory. The expected final checkpoint
is:

```text
<output-dir>/formal/checkpoints/model--epo-3.ckpt
```

Intermediate checkpoints are written at steps 5,771, 11,542, 17,313, 23,084,
28,855 and 34,626. The route identity is
`htsat_mizar_s1_fixed20_gbs128_recursive_mesh_5_10x4_5_v1`; x2, previous x4
GBS256 and old formal30 checkpoints are rejected by the strict contract and
resume checks.
