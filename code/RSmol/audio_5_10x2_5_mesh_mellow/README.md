# Isolated MeSH audio route

This route combines the existing 5-10x2-5 MeSH checkpoint with Mellow's
HTSAT interface and ReasonAQA manifests.  It does not modify the legacy
5-10-5, Mellow, or text-only MeSH scripts.

The contract is 32 kHz mono audio normalized to 10 seconds by cropping the
first 10 seconds or right-padding.  Empty audio2 rows reuse audio1 and, when
possible, the same encoded prefix.  HTSAT is frozen; Mellow c2l (527 to 768),
the projection/downsampling bridge, MeSH, and routers are trainable.  The
mapper follows Mellow exactly: random c2l, concatenate the 1x768 latent/CLS
with the projected framewise map, then a randomly Xavier-initialized
bias-free 768->576->576 nonlinear residual projection with dropout 0.5 and
LayerNorm, followed by CLS-preserving 8x average pooling.  Each audio becomes
129 tokens; two audios plus two separators form a 260-position audio prefix.

The formal route is 8 GPUs, microbatch 8 per GPU, GA 4 (effective global
batch 256), 3 epochs, max LR 1e-3, cosine schedule, warmup
`ceil(total_optimizer_steps * 0.05)`, and
gradient clipping 0.5.  Formal checkpoints are saved every 1000 optimizer
steps and only the newest four complete checkpoints are retained.  Each
sample is tokenized as `prompt + answer` before
the batch is right-padded to its longest complete text sequence.  Only the
real answer interval has labels; all audio, separator, prompt, and trailing
batch-padding positions are `-100`.  The standard causal-LM shift therefore
predicts the first answer token from the immediately preceding real prompt
token without inserting padding between prompt and answer.

GPU execution must go through the repository-level `vc submit` wrappers; do
not run the CUDA Python/torchrun scripts directly on a login node:

```text
run_audio_stage4_5_10x2_5_mesh_mellow_5090.sh  # 1-GPU forward/backward audit
run_audio_stage5_5_10x2_5_mesh_mellow_5090.sh  # 8-GPU topology smoke
run_audio_stage7_5_10x2_5_mesh_mellow_5090.sh  # 8-GPU 10-step/reload smoke
run_audio_formal_5_10x2_5_mesh_mellow_5090.sh  # 8-GPU formal 3-epoch training
run_audio_checkpoint_audit_5_10x2_5_mesh_mellow_5090.sh  # 1-GPU standalone checkpoint reload audit
run_audio_perf20_5_10x2_5_mesh_mellow_5090.sh  # isolated 8-GPU PERF20 baseline/profile
```

The wrappers accept the same arguments as their underlying runtime scripts and
submit them inside the project GPU image.

PERF20 fixes microbatch 8/GPU, GA=4, 20 optimizer steps, BF16, seed 0, and the
current drop12 manifest plus second-round MeSH checkpoint defaults. It never
saves/reloads/prunes checkpoints and refuses an existing output directory. The
baseline is submitted with `bash code/RSmol/run_audio_perf20_5_10x2_5_mesh_mellow_5090.sh`;
the independent profiling run adds `--profiler`. Profiler collection is rank0
only, CPU+CUDA, with optimizer-step granularity and default
`skip_first=4, wait=1, warmup=1, active=2, repeat=1`; stacks, memory, and shapes
are opt-in. Trace and operator summaries are under `output_dir/profile/`, and
the JSON report records schedule/options/artifact paths.

Each PERF20 report includes rank0 CUDA-aware device timings and explicitly
named host timings for data wait (including `next(data_iter)`), scheduler, and
metrics enqueue; the metrics/collectives device timing includes the one unified
CUDA/NCCL completion sync at the end of each optimizer step. Per-microbatch
sequence/token counts, global token throughput, and steady-state distributions
are included. Each completed profiler schedule cycle is exported from the
`on_trace_ready` callback into a unique `profile/cycle_<nn>_step_<nnnn>/`
directory; the report's `profiler.artifacts` list contains that cycle's trace
and operator-summary paths. By default steady state is optimizer steps 6-20,
excluding profiler wait/warmup/active steps in a profile run.
PERF20 also disables per-forward router statistics because their internal
GPU-to-CPU copies would add synchronization overhead to the timing path; the
first-step MeSH gradient/path audit remains enabled.
The repository's Windows checkout cannot run the remote CUDA/Mellow/HTSAT
validation; use the 5090 `vc submit` launcher for the actual measurement.
