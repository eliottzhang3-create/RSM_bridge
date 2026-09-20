# Original SmolLM2-135M audio partition baseline

This is the isolated comparison route for the current Audio MeSH experiment.
Its only intended model difference is the text backbone: it loads the original
standard 30-layer SmolLM2-135M `LlamaForCausalLM` from
`/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2`. It has no MeSH memory,
router, logical-layer reuse, or recursive parameters.

The current training contract is:

```text
smollm2_component_partitions6_rank_ram_compact_audio_answer_eos_v2
```

All other training behavior mirrors the current Audio MeSH partition route:

- the audited six-component waveform store;
- one complete partition cloned into anonymous CPU RAM per DDP rank;
- strict per-rank RSS and cgroup-anon release before the next preload;
- frozen HTSAT, trainable Mellow c2l, bridge, and complete text model;
- compact 130-token single-audio and 260-token dual-audio prefixes;
- prompt and answer joined before batch padding;
- exactly one supervised `<|endoftext|>` at the end of every answer;
- 8 GPUs, microbatch 8/GPU, gradient accumulation 4, global batch 256;
- AdamW, max LR 1e-3, 5% warmup, cosine decay, gradient clipping 0.5;
- 10 epochs, 37,810 optimizer steps, save every 500, retain four.

The model keeps `compact_single_audio_prefix=False` by default so historical
fixed-prefix checkpoints remain usable by their generation and evaluation
scripts. The partition trainer explicitly enables compact mode.

## Current files

```text
audio_smollm2_135m_mellow/model.py
audio_smollm2_135m_mellow/data.py
scripts/train_audio_partitioned_smollm2_135m_mellow_ddp.py
scripts/train_audio_smollm2_135m_mellow_smoke20_ddp.sh
scripts/train_audio_smollm2_135m_mellow_resume2_ddp.sh
scripts/train_audio_smollm2_135m_mellow_formal_ddp.sh
run_audio_smollm2_135m_mellow_smoke20_3090.sh
run_audio_smollm2_135m_mellow_resume2_3090.sh
run_audio_smollm2_135m_mellow_formal_3090.sh
scripts/evaluate_mmau_test_mini_audio_smollm2.py
scripts/evaluate_mmar_audio_smollm2.py
run_mmau_test_mini_audio_smollm2_5090.sh
run_mmar_audio_smollm2_5090.sh
```

All training jobs use `pdgpu-3090`, 32 CPU cores, 256 GiB RAM, and 8 GPUs.
The old `_5090.sh` training filenames are compatibility forwarders to the
canonical 3090 wrappers.

## Required smoke and resume gate

Use new, empty output directories:

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

SMOKE_OUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/partition_smoke20_eos_v2_20260918
RESUME_OUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/partition_resume2_eos_v2_20260918

bash run_audio_smollm2_135m_mellow_smoke20_3090.sh \
  --output-dir "$SMOKE_OUT"

bash run_audio_smollm2_135m_mellow_resume2_3090.sh \
  --resume-from "$SMOKE_OUT/checkpoint-000020" \
  --output-dir "$RESUME_OUT"
```

The first job executes `p2:10 -> release -> p0:10 -> release` and publishes
`checkpoint-000020`. The second restores that complete checkpoint, executes
`p1:2 -> release`, proves finite nonzero changes in representative text,
bridge, and c2l parameters, and publishes `checkpoint-000022`.

Both reports must be real remote `PASS` reports before formal training:

```text
$SMOKE_OUT/partition_training_report.json
$RESUME_OUT/partition_training_report.json
```

## Formal training

```bash
FORMAL_OUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/partition_formal_eos_v2_10epochs_20260918

bash run_audio_smollm2_135m_mellow_formal_3090.sh \
  --output-dir "$FORMAL_OUT" \
  --smoke20-report "$SMOKE_OUT/partition_training_report.json" \
  --smoke-resume-report "$RESUME_OUT/partition_training_report.json"
```

The formal gate validates the baseline-specific training contract, partition
inventory, seed, 0-to-20 and 20-to-22 cursors, resume lineage, all eight rank
release records, cgroup anon release, standard 30-layer SmolLM2 gradients,
compact prefixes, supervised EOS, and text/bridge/c2l resume updates.

For a formal continuation, use a new empty output directory, pass the formal
checkpoint via `--resume-from`, retain `--epochs 10` and every other contract
value, and pass the same two smoke reports.

## MMAU and MMAR evaluation

The current evaluation target is:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/
partition_formal_eos_v2_10epochs_20260918/checkpoint-037810
```

Both evaluators use one GPU on `pdgpu-5090`, audit the completed partition-v2
checkpoint before inference, use the compact 130-token single-audio prefix,
and share the current MeSH benchmark I/O/scoring protocol. Model text is
passed unchanged to the official evaluator. Run smoke and full with the same
output directory so the full job resumes after the first five rows.

## Historical checkpoint

The completed historical baseline remains at:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/
formal_20260911_v1/checkpoint-011343
```

It belongs to the old fixed-260-prefix, online-data, three-epoch artifact
contract. It remains available for historical ReasonAQA sample generation,
but it is not a valid `--resume-from` source for partition-v2 training and is
rejected by the current MMAU/MMAR evaluators.
