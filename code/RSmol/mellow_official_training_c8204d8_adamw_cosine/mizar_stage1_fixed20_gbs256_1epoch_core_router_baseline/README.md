# Core/router frozen baseline

This route is an isolated one-epoch comparison baseline copied from
`mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh`. The recursive MeSH
architecture and original audio path are unchanged. It starts from the
completed model-only checkpoint:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_recursive_mesh_5090/formal_recursive_mesh_padfix_20261025_003124507033014/formal/checkpoints/model--step-17313.ckpt
```

The initialization loader requires the source route and step `17313`, checks
the complete state-dict key set and shapes, and compares every loaded tensor
exactly. Source optimizer, scheduler, scaler, GradNorm and RNG state are not
continued; formal training starts a fresh one-epoch schedule.

## Model and freeze contract

The forward path remains the original fixed20 route:

```text
HTSAT(audio1) -> c2l -> projection -> 129 tokens
HTSAT(audio2) -> c2l -> projection -> 129 tokens
audio[129+129] + SEP[1] + prompt[256] + answer[300] -> original MeSH
```

There is no stage2 tap, audio memory, cross-attention, frequency embedding or
perceive router. The original recursive schedule remains
`Prelude(0..4) -> Core(5..14) -> Core(5..14) -> Coda(15..19)`.

Only these parameters are trainable:

- physical Core layers `5..14`, shared by both recursive executions;
- the original three write routers and three read routers.

HTSAT, `c2l`, `projection`, `embed_tokens`, Prelude `0..4`, Coda `15..19`,
final norm and `lm_head` are frozen. The complete old audio path runs in eval
mode and under `no_grad`, including projection dropout.

## Training contract

- 8 GPUs x microbatch 8 x gradient accumulation 4 = global batch 256;
- original 1,477,492-row data mixture and fixed20 audio policy;
- one epoch, 5,771 optimizer updates, 289 warmup updates;
- every trainable group uses AdamW max LR `5e-5`, cosine floor `5e-6`;
- original AdamW betas, weight decay, GradNorm, FP32 forward, and disabled
  autocast/TF32 behavior.

The formal run uses the original shared-audio startup, background staging and
collective switch to local staged files. There is deliberately no smoke mode.

## Submission

From the repository root:

```bash
bash code/RSmol/run_mizar_s1_core_router_baseline_gbs256_5090.sh
```

Optional overrides include `--output-dir`, `--init-checkpoint`,
`--resume-checkpoint`, `--resolved-paths`, `--text-model-dir`,
`--htsat-root`, `--index-cache-root` and `--stage-workers`.

The final checkpoint is:

```text
<output-dir>/formal/checkpoints/model--epo-1.ckpt
```

The formal report verifies strict initialization, the exact trainable
whitelist, frozen-parameter digest stability, finite gradients, Core/router
updates, sampler and LR schedule, DDP synchronization, and full checkpoint
reload state.
