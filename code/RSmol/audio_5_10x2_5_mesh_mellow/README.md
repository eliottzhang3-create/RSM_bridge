# Isolated MeSH audio route

This route combines the existing 5-10x2-5 MeSH checkpoint with Mellow's
HTSAT interface and ReasonAQA manifests.  It does not modify the legacy
5-10-5, Mellow, or text-only MeSH scripts.

The contract is 32 kHz mono audio normalized to 10 seconds by cropping the
first 10 seconds or right-padding.  Empty audio2 rows reuse audio1 and, when
possible, the same encoded prefix.  HTSAT is frozen; Mellow c2l (527 to 768),
the projection/downsampling bridge, MeSH, and routers are trainable.

The formal route is 8 GPUs, microbatch 4 per GPU, GA 1, 3 epochs, max LR
1e-3, cosine schedule, warmup `ceil(total_optimizer_steps * 0.05)`, and
gradient clipping 0.5.  Only answer tokens have labels; all audio, separator,
prompt, and answer padding labels are `-100`.
