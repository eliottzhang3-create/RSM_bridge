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
run_audio_storage_probe_5090.sh  # read-only local-storage/memory discovery
```

The wrappers accept the same arguments as their underlying runtime scripts and
submit them inside the project GPU image.

PERF20 currently uses 32 CPU cores and synchronous loading (`num_workers=0`
per rank), after the 64-core/two-worker experiment exposed worse shared-storage
tail latency and rank skew. It fixes microbatch 8/GPU, GA=4, 20 optimizer
steps, BF16, seed 0, and the current drop12 manifest plus second-round MeSH
checkpoint defaults. It never
saves/reloads/prunes checkpoints and refuses an existing output directory. The
baseline is submitted with `bash code/RSmol/run_audio_perf20_5_10x2_5_mesh_mellow_5090.sh`;
the independent profiling run adds `--profiler`. Profiler collection is rank0
only, CPU+CUDA, with optimizer-step granularity and default
`skip_first=4, wait=1, warmup=1, active=2, repeat=1`; stacks, memory, and shapes
are opt-in. Trace and operator summaries are under `output_dir/profile/`, and
the JSON report records schedule/options/artifact paths.

The input-path control adds `--preload-data`. Before rank0 enters the profiler
or any optimizer-step timer, every rank materializes the exact 80 CPU batches
needed by 20 steps at GA=4, then all ranks cross one pre-measurement barrier.
Measured training takes batches only from the rank-local list, so shared-storage
reads, audio decode/resampling, tokenization, and collate are outside the timed
region. The default command is
`bash code/RSmol/run_audio_perf20_5_10x2_5_mesh_mellow_5090.sh --preload-data`;
its unique output directory starts with `perf20_preloaded_`. The report records
per-rank preload duration, barrier wait, CPU tensor bytes, row-order hash, and
exact loaded/consumed counts. Any count other than 80 is a hard failure.

The persistent unique waveform store default is the manifest-scoped v3 store
(`rsmol_reasonaqa_train_unique_waveforms_32k_10s_f32_v3`).  In addition to the
ordinary `shared_waveform_store` mmap control, the isolated
`shared_waveform_store_tmpfs` control copies the complete validated store into
the node's `/dev/shm` before `torchrun`, then all eight ranks open the same
node-shared files.  The wrapper checks `metadata.json.status=PASS`, rejects an
unfinished `BUILDING` store, verifies `/dev/shm` capacity with a 5 GiB safety
margin, and removes only its uniquely named staging directory on exit.  This
mode is opt-in and does not change the persistent-store or partition controls.

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
Each rank retains local data-wait, CUDA-event forward/backward, and completed
step-wall timings. A single `gather_object` runs after all 20 measured steps;
the rank0 report contains raw rank rows, per-rank distributions, and per-step
cross-rank skew/slowest-rank summaries without adding a measured-step
collective.
PERF20 also disables per-forward router statistics because their internal
GPU-to-CPU copies would add synchronization overhead to the timing path; the
first-step MeSH gradient/path audit remains enabled.
The repository's Windows checkout cannot run the remote CUDA/Mellow/HTSAT
validation; use the 5090 `vc submit` launcher for the actual measurement.

## Historical checkpoint-011343 MMAU evaluation

The original three-epoch fixed-260 checkpoint is evaluated through an
isolated legacy adapter rather than the current shared-store or compact
partition evaluator:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
audio_5_10x2_5_mesh_mellow/
formal_restart_save500_20260910_105248/checkpoint-011343
```

Run the complete 1,000-row MMAU test-mini and official scorer with:

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mmau_test_mini_audio_5_10x2_5_mesh_mellow_legacy_fixed260_5090.sh
```

The adapter validates the pre-contract checkpoint schema written by the
September 10 trainer: 3 epochs, 11,343 optimizer steps, 568 warmup steps,
LR `1e-3 -> 0`, save interval 500, eight-rank RNG state, and final saved
cursor `epoch=2,batch_in_epoch=15124,global_step=11343`. It does not relabel
the artifact as a shared-store, compact-prefix, or answer-EOS checkpoint.

Inference matches the checkpoint's training-time audio layout. A single-audio
question encodes audio1 once with HTSAT, reuses that embedding for slot two,
and invokes the bridge separately for both slots, producing a fixed 260-token
prefix. Decoded prediction text is passed verbatim to the official scorer;
the leading `a)`-`d)` label is not removed. The current full-denominator and
status contracts still apply: skipped rows become empty incorrect predictions,
`prompt_length_audit.prompt_exceeds_max_tokens` must be inspected, and a
top-level `PASS` requires the official scorer to report all 1,000 samples.

Before designing a persistent waveform-cache staging path, run
`bash code/RSmol/run_audio_storage_probe_5090.sh`. It requests the same queue,
image, CPU, memory, GPU, and single-node shape as formal training, but only
performs read-only mount/capacity/cgroup discovery. Its persistent JSON report
classifies tmpfs, overlay, network filesystems, and local filesystems, and only
marks a writable local path eligible when at least 150 GiB is free. Raw `df`,
`findmnt`, `lsblk`, `/proc`, cgroup, and VM diagnostics are stored beside the
JSON report. A separate bounded I/O benchmark should be designed only after a
real candidate path has been identified.

The CPU-only `scripts/prepare_audio_waveform_shards.py` builder creates the
persistent Parastor cache before any training reader is enabled. It scans the
three currently used audio roots, deduplicates canonical paths, applies the
same mono/32-kHz/10-second preprocessing as `ReasonAQADataset`, globally
shuffles once with a fixed seed, and writes 64 balanced, immutable raw-float32
shards plus a path-to-shard/row/offset index. Shards are completed atomically;
an interrupted build can use `--resume`, which validates the source
path/size/mtime inventory, configuration, and index hash before skipping
complete shards. `metadata.json.status=PASS` and the absence of `BUILDING` are
the completion contract. The future reader will shuffle shard order and rows
inside each shard independently per epoch; shard generation alone does not
change the existing training Dataset/Sampler.
