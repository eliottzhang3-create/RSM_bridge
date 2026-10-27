# HTSAT / Mizar S1 GBS256 / MeSH init0907

This directory is a complete isolated copy of
`../mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh/`.
The only training change is the default text initialization checkpoint:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_resume_000500_nonfatal_router_20260907_115533/checkpoint-009244
```

The full MeSH text model (including routers) and tokenizer are loaded through
the original loader. This starts a new audio-training run at step 0; it does
not restore the text checkpoint's optimizer, scheduler, RNG or data cursor.
HTSAT and c2l/projection initialization follow the parent route unchanged.
The shared `recursive_model_5_10x2_5_mesh.py` implementation is unchanged.

## Training settings inherited from the parent

- Full trainable MeSH: hidden 576, 20 physical / 30 logical layers,
  `5 + 10 x 2 + 5`, 5 transient memory slots, 6 routers, no LoRA.
- Frozen/eval HTSAT, trainable c2l/projection, online fixed20 audio.
- Hybrid 1,477,492-row manifest and the same resolved-path TSV, source mapping,
  shuffle, shared-file startup, background copying and collective local switch.
- `129 + 129 + 1 + 256 + 300 = 815` sequence, prefix 515,
  Mizar padding, `attention_mask=None`, original answer-only CE reduction.
- 8 GPUs x microbatch 8 x accumulation 4 = global batch 256, 3 epochs,
  5,771 updates/epoch, 17,313 total updates, 866 warmup updates.
- FP32, mapper LR 6e-4 / LM LR 3e-4, original AdamW, LambdaLR and GradNorm.
- Original checkpoint steps: 2,886, 5,771, 8,656, 11,542, 14,428 and 17,313;
  full optimizer/scheduler/GradNorm/scaler/RNG/data-cursor restoration.
- `mellow_c8204d8`, `pdgpu-5090`, 32 CPUs, 256G, 8 GPUs, 1 node,
  same container as the parent. Direct formal submission, no smoke prerequisite.

## Submit training

After synchronizing the code through GitHub, run on the remote login node:

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_init0907_5090.sh --stage-workers 4
```

The wrapper defaults to the checkpoint above and creates a timestamped output
directory under:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_recursive_mesh_init0907_5090/formal_<timestamp>/
```

An explicit output directory can also be supplied:

```bash
bash run_mizar_s1_gbs256_recursive_mesh_init0907_5090.sh \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_recursive_mesh_init0907_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

Files beneath that output directory include:

```text
formal/runtime.json
formal/training_report.json
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
```

## Isolation and resume

The route identity is
`htsat_mizar_s1_fixed20_gbs256_recursive_mesh_init0907_v1`.
Outputs, job logs, index cache and tmpfs prefixes are separate from the parent.
The original resume checks accept only a full formal checkpoint produced by
this same route/code/data/recipe. To resume this new route after interruption:

```bash
bash run_mizar_s1_gbs256_recursive_mesh_init0907_5090.sh \
  --stage-workers 4 \
  --resume-checkpoint /absolute/path/to/this_route/formal/checkpoints/model--step-N.ckpt \
  --output-dir /absolute/path/to/a/new/output_directory
```

The text initialization directory is not a `--resume-checkpoint` input.

## Verification scope

Local Python compilation, syntax checks for both shell entry points, and 31
existing parent/baseline/shared-start regression checks passed.
The existing startup checks and optional runtime checks are retained without
adding new training gates. `reference_audit.json` is copied historical evidence
from the parent, not a new audit or a GPU result for this checkpoint.
The local Windows checkout contains no training weights/data or PyTorch;
actual checkpoint loading and training are verified by the remote formal job.
The parent's existing `recursive_mesh` evaluation variant still defaults to
its original run; it does not automatically select this new route.
