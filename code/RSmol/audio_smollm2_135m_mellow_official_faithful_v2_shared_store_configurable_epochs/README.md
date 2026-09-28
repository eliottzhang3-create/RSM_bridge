# Mellow official-faithful-v2 SmolLM2 shared-store training

## Status on 2026-09-28

This is a fully isolated reproduction route for Mellow training commit
`c8204d8eb99b4384fd7a76ad57995731e0c0c2bf`.

The local implementation, Python compilation, and static contract tests are ready.
Remote smoke20, resume2, BatchNorm-buffer, NCCL synchronization, and formal-training
results have not yet been registered as PASS.

Do not use these files for formal training until this route's own smoke20 and resume2
reports both say PASS. Local checks are not a remote GPU PASS.

The older `audio_smollm2_135m_mellow_shared_store_configurable_epochs` route remains
unchanged. Its checkpoints and reports are rejected by this v2 contract.

## Differences addressed by v2

1. HTSAT weights remain frozen, while the HTSAT wrapper and backbone remain in train
   mode. SpecAugment, DropPath, and train-mode BatchNorm behavior can therefore run.
2. Both audio separators use literal token ID 0, matching the pinned Mellow decoder.
   Padding remains the literal `!` token and is audited independently.
3. The fixed 0.5 gradient clip is replaced by the pinned per-parameter Mellow
   `GradNormTracker`.
4. A disabled `GradScaler` preserves the official backward, unscale, tracker, step,
   and update call order while forward and backward remain FP32.

Training contract:

```text
smollm2_mellow_official_faithful_v2_variable_store_random_audio2_fixed639_
htsat_train_separator0_official_gradnorm_adam_epoch_cosine_exact_resume_v1
```

## Model and sequence contract

- Standard 30-layer SmolLM2-135M decoder.
- Frozen HTSAT parameters loaded from the external AudioSet checkpoint.
- Train-mode HTSAT modules.
- Trainable Mellow c2l(527 to 768) and trainable 768 to 576 to 576 bridge.
- Two independent HTSAT and bridge passes for every row.
- Missing audio2 is sampled from the sorted non-empty filepath1 pool.
- Long audio uses a random inclusive 10-second crop; short audio is right-padded.

```text
audio1 129 + separator ID 0 + audio2 129 + separator ID 0
+ prompt 129 + answer 250 = 639 tokens
```

The complete variable-length waveform store is staged once under the route-specific
`/dev/shm/rsmol_smollm2_mellow_official_faithful_v2_*` prefix. Eight ranks mmap
the same waveform file inode.

## Batch and optimization contract

Qualification smoke/reference/resume:

```text
8 GPUs x microbatch 4 x gradient accumulation 1 = global batch 32
```

Formal training:

```text
8 GPUs x microbatch 8 x gradient accumulation 4 = global batch 256
30 epochs
Adam, learning rate 1e-3, weight decay 1e-4
epoch-level CosineAnnealingLR with eta_min 0
save every 5000 optimizer steps and at the final step
retain the newest four complete checkpoints
```

With 968,059 rows, formal training has 3,781 optimizer steps per epoch and
113,430 total optimizer steps. All submission wrappers use `pdgpu-3090`, one node,
eight GPUs, 32 CPUs, and 256 GiB memory.

## GradNormTracker

The pinned tracker uses initial L2 norm 0.5, initial max norm 5.0, overdrive factor
2.5, and EMA momentum 0.995. It measures each named parameter's L2 and infinity
norms, applies one shared minimum scale to all gradients, and checkpoints only the
official `running_norm` state. Names come from the unwrapped owner model.

## HTSAT buffers and resume

DDP uses `broadcast_buffers=False`, so train-mode BatchNorm buffers may differ by
rank inside an epoch. At each completed epoch, the trainer broadcasts the complete
rank-0 model state and optimizer tensor state.

Mid-epoch checkpoints save every rank's HTSAT named buffers in
`htsat_buffers_by_rank.pt`. Complete checkpoints also contain the tracker state,
Adam state, scheduler state, cursor, and all eight rank RNG states. Resume loads the
rank-specific HTSAT buffers before the first resumed forward and restores RNG again
after constructing the DataLoader iterator.

## Runtime audits

The first optimizer step must prove that HTSAT is in train mode but gradient-free;
c2l, bridge, embeddings, all text layers, and LM head have finite gradients; both
separator positions are token ID 0; at least one expected BatchNorm buffer changes;
and the GradNormTracker initializes finite per-parameter state.

Formal training requires both a passing v2 smoke20 report and a passing v2 resume2
report. The resume report must descend from the smoke checkpoint-000020 and finish
at checkpoint-000022.

## Remote commands

Run from the remote `code/RSmol` directory.

```bash
bash run_audio_smollm2_shared_store_smoke20_configurable_epochs_135m_mellow_official_faithful_v2_3090.sh --epochs 30

bash run_audio_smollm2_shared_store_resume2_configurable_epochs_135m_mellow_official_faithful_v2_3090.sh --epochs 30 --resume-from /actual/v2/smoke/checkpoint-000020

bash run_audio_smollm2_shared_store_formal_configurable_epochs_135m_mellow_official_faithful_v2_3090.sh --epochs 30 --smoke20-report /actual/v2/smoke/shared_store_training_report.json --smoke-resume-report /actual/v2/resume/shared_store_training_report.json
```

The reference22 entry remains an optional uninterrupted diagnostic. It does not
restore the retired cross-job bitwise-loss gate.

## Local verification boundary

The Windows checkout does not contain Torch, HTSAT, remote weights, or the waveform
store. Python compilation, static tests, and diff checks can run locally. Actual
SpecAugment, DropPath, BatchNorm updates, NCCL broadcasts, and exact resume continuity
must be established by remote smoke20 and resume2 reports.
