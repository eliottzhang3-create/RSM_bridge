# Fixed 5-10-5 recursive Mellow partition baseline

This is the isolated fixed-recursion comparison route for the current Audio
MeSH experiment. Its text model owns twenty independent physical decoder
modules. Physical modules 0--4 run once, modules 5--14 run twice, and modules
15--19 run once, giving the exact thirty-entry logical schedule:

```text
0..14, 5..14, 15..19
```

There are no MeSH memory slots or routers. The canonical text initialization
remains:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/
formal-epoch2-continue-20260902_184936/checkpoint-step-009244
```

The current training contract is:

```text
recursive_5_10_5_component_partitions6_rank_ram_compact_audio_answer_eos_v2
```

Except for the fixed recursive text backbone, training matches the current
Audio MeSH partition route:

- audited six-component waveform store;
- one complete partition cloned into anonymous CPU RAM per DDP rank;
- strict per-rank RSS and cgroup-anon release before another preload;
- frozen HTSAT, trainable Mellow c2l, bridge, and complete text model;
- compact 130-token single-audio and 260-token dual-audio prefixes;
- exactly one supervised `<|endoftext|>` at every answer end;
- 8 GPUs, microbatch 8/GPU, gradient accumulation 4, global batch 256;
- AdamW with betas 0.9/0.95, weight decay 0.1, max LR 1e-3;
- 5% warmup, cosine decay, gradient clipping 0.5;
- 10 epochs, 37,810 optimizer steps, save every 500, retain four;
- no periodic validation.

All jobs use `pdgpu-5090`, 32 CPU cores, 256 GiB RAM, and 8 GPUs.

## Required smoke and resume gate

Use new empty output directories:

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

SMOKE_OUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_recursive_mellow/partition_smoke20_eos_v2_20260919
RESUME_OUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_recursive_mellow/partition_resume2_eos_v2_20260919

bash run_audio_5_10_5_recursive_mellow_smoke20_5090.sh \
  --output-dir "$SMOKE_OUT"

bash run_audio_5_10_5_recursive_mellow_resume2_5090.sh \
  --resume-from "$SMOKE_OUT/checkpoint-000020" \
  --output-dir "$RESUME_OUT"
```

The first job executes `p2:10 -> release -> p0:10 -> release` and publishes
`checkpoint-000020`. The second restores that artifact, executes
`p1:2 -> release`, proves finite nonzero changes in representative text,
bridge, and c2l parameters, and publishes `checkpoint-000022`.

Both reports must be real remote `PASS` reports:

```text
$SMOKE_OUT/partition_training_report.json
$RESUME_OUT/partition_training_report.json
```

The first-step audit requires the exact 20-physical/30-logical schedule, both
middle passes to carry finite gradients, all physical layers and audio mapper
parameters to have finite gradients, HTSAT to remain frozen, no router/memory
parameters, compact prefixes, and supervised answer EOS.

## Formal training

Formal training starts fresh from the canonical text initialization. Smoke
checkpoints are evidence for the gate, not formal-training initial weights.

```bash
FORMAL_OUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_recursive_mellow/partition_formal_eos_v2_10epochs_20260919

bash run_audio_5_10_5_recursive_mellow_formal_5090.sh \
  --output-dir "$FORMAL_OUT" \
  --smoke20-report "$SMOKE_OUT/partition_training_report.json" \
  --smoke-resume-report "$RESUME_OUT/partition_training_report.json"
```

The formal gate rereads both reports and their checkpoint markers/configs. It
validates the route-specific contract, inventory, seed, schedule, cursors,
resume lineage, eight-rank RSS/cgroup release, recursive trace and gradients,
compact prefix/EOS, and text/bridge/c2l resume updates.

For a formal continuation, pass a formal checkpoint via `--resume-from`, keep
all contract values unchanged, use a new empty output directory, and pass the
same two smoke reports.

## Historical contract

The old trainer remains in the repository for historical inspection:

```text
scripts/train_audio_5_10_5_recursive_mellow_ddp.py
audio_5_10_5_recursive_mellow_composite_v1
```

It used online manifests, fixed 260-token prefixes, three epochs, 11,343
steps, and a different checkpoint format. Any checkpoint from that contract,
a text-only checkpoint, a MeSH checkpoint, or another baseline is not a valid
`--resume-from` source for partition-v2 training.

The user has since provided the completed formal artifact
`partition_formal_eos_v2_10epochs_20260919/checkpoint-037810`. The evaluation
entry points below independently re-audit its full partition-v2 contract;
local code inspection alone is not recorded as a remote evaluation `PASS`.

## MMAU test-mini and MMAR evaluation

The isolated fixed-recursive evaluation entries are:

```text
scripts/evaluate_mmau_test_mini_audio_5_10_5_recursive_mellow.py
scripts/evaluate_mmar_audio_5_10_5_recursive_mellow.py
run_mmau_test_mini_audio_5_10_5_recursive_mellow_5090.sh
run_mmar_audio_5_10_5_recursive_mellow_5090.sh
```

Both submission wrappers default directly to the complete 1,000-question
official evaluation, one GPU in `pdgpu-5090`, and this route's completed
`partition_formal_eos_v2_10epochs_20260919/checkpoint-037810`. They have
separate append-only output directories under `audio_5_10_5_recursive_mellow`.
The official data, prompt, scorer, and result materialization are shared with
the established MMAU/MMAR pipelines, while model loading and checkpoint
auditing use only the fixed-recursive model. Every generated token audits the
exact 30-call physical-layer trace `0..14, 5..14, 15..19` under greedy,
full-recompute, `use_cache=False`, 32-token inference. Inference exceptions
invalidate the evaluation report instead of silently producing a low score.

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mmau_test_mini_audio_5_10_5_recursive_mellow_5090.sh
bash run_mmar_audio_5_10_5_recursive_mellow_5090.sh
```

The default output directories must be new/empty on the first run. Existing
owned directories resume append-only progress, so a rerun after a model-side
failure should use a distinct output directory after fixing the issue.
