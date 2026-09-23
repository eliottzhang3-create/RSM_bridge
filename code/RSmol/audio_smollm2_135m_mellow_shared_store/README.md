# Original SmolLM2-135M shared-store configurable-epoch baseline

This route is the isolated comparison for the configurable-epoch Audio MeSH
shared-store experiment.  It keeps the complete ReasonAQA manifest, node-local
`/dev/shm` store, full-manifest distributed shuffle, fixed 260-token two-slot
audio prefix, optimizer, scheduler, checkpoint cadence, and 20+2 gate.  Its
only intended model change is replacing MeSH with the original standard
30-layer SmolLM2-135M model.

The route contract is:

```text
smollm2_node_shared_unique_store_fullshuffle_fixed260_audio_reuse_answer_eos_v2
```

For single-audio rows, one HTSAT embedding is reused for slot two and the
trainable bridge is invoked separately for both slots.  Dual-audio rows retain
their two distinct waveforms.  HTSAT stays frozen; Mellow c2l, the bridge, and
all original SmolLM2 parameters are trainable.  Router, memory, recursive, and
shared-loop parameters are forbidden.

With 968,059 rows, 8 GPUs, microbatch 8 and gradient accumulation 4:

```text
steps_per_epoch = 3,781
30 epochs       = 113,430 optimizer steps
warmup          = ceil(113,430 * 0.05) = 5,672 steps
```

Smoke20, resume2, and formal must use the same explicit `--epochs 30`.  Formal
training requires this route's two real PASS reports; reports or checkpoints
from MeSH, partition, compact-prefix, or historical online routes are rejected.

