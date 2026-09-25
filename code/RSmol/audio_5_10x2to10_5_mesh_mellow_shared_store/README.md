# Variable-depth Audio MeSH shared-store route

This package is isolated from the fixed two-loop implementation. It preserves
the audited ReasonAQA fixed-260-token two-slot audio pipeline and the twenty
physical 5-10-5 decoder modules, while one integer recursive depth is sampled
uniformly from 2 through 10 for each micro-step and broadcast from rank zero.

Router migration from the fixed checkpoint is explicit: pre routers come from
the old pre routers, loop1 routers from old loop 0, refine_write and out_read
from old loop 1, and the only new router (refine_read) is initialized from old
loop-0 read. At depth two, refine_read is intentionally unused.

The eventual formal schedule defaults to 7 epochs with warmup equal to 5% of
the real optimizer-step count (rounded up). Phase 1--4 utilities in
scripts create the initialization artifact, audit every structural depth
automatically, and run a real depth-10 activation-memory preflight before any
training smoke/resume/formal job is enabled.

## Phases 5--8

Phase 5 is the isolated training implementation.  It reuses the audited v3
unique waveform store, fixed 260-token two-slot data path, trained HTSAT/c2l
and bridge weights, and the migrated variable-depth MeSH artifact.  The model
samples one integer `T ~ Uniform{2,...,10}` for each micro-step on rank zero and
broadcasts it to all eight ranks.  Because `refine_read` is intentionally
unused at `T=2`, DDP reduces gradients on every micro-step instead of using a
`no_sync` accumulation window.  The four reduced micro-step gradients are then
accumulated before the optimizer step.

Phase 6 is a fresh 20-optimizer-step smoke and writes `checkpoint-000020`.
Phase 7 restores model, optimizer, scheduler, all eight rank RNG states, and
the recursive-depth generator/histogram, then runs exactly two more steps to
`checkpoint-000022`.  Phase 8 is the gated seven-epoch formal run and accepts
only the matching Phase 6 and Phase 7 PASS reports.

For the canonical 968,059-row ReasonAQA manifest, world size 8, micro-batch 8,
and gradient accumulation 4, the formal shape is 3,781 optimizer steps per
epoch, 26,467 total steps, and `ceil(26467 * 0.05) = 1,324` warmup steps.

Entry points:

```text
scripts/train_audio_shared_store_5_10x2to10_5_mesh_mellow_ddp.py
scripts/stage_audio_shared_store_5_10x2to10_5_mesh_mellow.sh
run_audio_shared_store_5_10x2to10_5_mesh_mellow_smoke20_5090.sh
run_audio_shared_store_5_10x2to10_5_mesh_mellow_resume2_5090.sh
run_audio_shared_store_5_10x2to10_5_mesh_mellow_formal_5090.sh
```

Remote execution order:

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

# Phase 6
bash run_audio_shared_store_5_10x2to10_5_mesh_mellow_smoke20_5090.sh

# Phase 7: substitute the actual Phase-6 directory.
bash run_audio_shared_store_5_10x2to10_5_mesh_mellow_resume2_5090.sh \
  --resume-from /hpc_stor03/.../smoke20_7epochs_.../checkpoint-000020

# Phase 8: substitute the two actual PASS report paths.
bash run_audio_shared_store_5_10x2to10_5_mesh_mellow_formal_5090.sh \
  --smoke20-report /hpc_stor03/.../smoke20_7epochs_.../shared_store_training_report.json \
  --smoke-resume-report /hpc_stor03/.../resume2_7epochs_.../shared_store_training_report.json
```

Every run refuses a non-empty output directory.  Formal checkpoint resume is
supported by passing `--resume-from` together with the same two Phase 6/7
reports; all training, store, initialization, phase-gate, scheduler, RNG, and
depth-sampler contracts must remain identical.
