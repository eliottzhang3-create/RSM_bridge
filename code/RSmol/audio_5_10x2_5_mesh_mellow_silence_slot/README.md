# Fixed-260 runtime-silence MeSH audio experiment

This package is isolated from the compact `audio_5_10x2_5_mesh_mellow`
mainline.  It retains the mainline's six audited component partitions,
rank-local RAM preload/release lifecycle, 5-10-10-5 MeSH execution, Mellow
mapper, answer-EOS-v2 labels, 8-GPU microbatch 8 / GA 4 schedule, optimizer,
10 epochs, checkpoint cadence, and seed.  Its learning-rate schedule uses
max LR `1e-3`, warmup `ceil(total_steps * 0.05)`, and cosine decay to min LR
`1e-4`.

Its only intended experimental change is the audio-prefix contract:

- Every row has two 129-token audio slots plus two separators: 260 prefix
  tokens in total.
- The exact layouts are `A1 + SEP + ZERO_WAVE + SEP` for structurally
  single-audio rows and `A1 + SEP + A2 + SEP` for dual-audio rows.
- A structurally single-audio row uses its real audio in slot 1 and an exact
  all-zero `[1, 1, 320000]` waveform in slot 2.
- The zero waveform is made on the current GPU inside `forward`; it is never
  saved, placed in a partition store, decoded, or transferred from CPU.
- Only one zero waveform is passed through HTSAT/c2l per microbatch that
  contains single-audio rows.  Its embedding is expanded before the trainable
  bridge, so per-row bridge dropout and accumulated gradients are retained.
- A true two-audio row uses both real waveforms.  An explicit identical-path
  dual row remains dual and may reuse the first encoder embedding.
- Loss covers only real answer tokens, including the terminal
  `<|endoftext|>` token.  Both audio slots, both separators, the prompt, and
  trailing batch padding use label `-100`.

The checkpoint contract is
`component_partitions6_rank_ram_fixed260_runtime_silence_second_slot_answer_eos_v2`.
Its unique config filename is
`audio_mesh_fixed260_silence_slot_config.json`.  Compact-mainline, SmolLM2,
recursive, and older fixed-260 checkpoints cannot resume this route.  Formal
training accepts only this route's own PASS 20-step and 20-to-22 resume
reports.

Run all GPU work through the repository submission wrappers on `pdgpu-5090`.
The wrappers require a new, empty `--output-dir`.

```bash
# 1. Initial 20 optimizer steps.
bash run_audio_partition_smoke20_5_10x2_5_mesh_mellow_silence_slot_5090.sh \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_silence_slot/smoke20_20260921

# 2. Resume exactly two steps from the step-20 artifact into another directory.
bash run_audio_partition_resume2_5_10x2_5_mesh_mellow_silence_slot_5090.sh \
  --resume-from /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_silence_slot/smoke20_20260921/checkpoint-000020 \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_silence_slot/resume2_20260921

# 3. Fresh formal training from the canonical text MeSH checkpoint.
bash run_audio_partition_formal_5_10x2_5_mesh_mellow_silence_slot_5090.sh \
  --smoke20-report /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_silence_slot/smoke20_20260921/partition_training_report.json \
  --smoke-resume-report /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_silence_slot/resume2_20260921/partition_training_report.json \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_silence_slot/partition_formal_answer_eos_v2_10epochs_20260921
```

The smoke report records single/dual row counts, runtime-silence encoder calls,
effective second-encoder batch size, step time, peak allocated CUDA memory,
the exact MeSH trace/gradient audit, store-miss invariants, and partition RAM
release evidence.  This local checkout cannot validate CUDA, remote weights,
or the remote partition stores; those checks complete only in the submitted
smoke jobs.
