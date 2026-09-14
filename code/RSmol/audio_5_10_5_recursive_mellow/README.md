# Fixed 5-10-5 recursive Mellow audio route

This isolated comparison route trains the historical fixed SmolLM2 5-10-5
checkpoint on the same ReasonAQA three-epoch contract as the MeSH and original
SmolLM2 audio experiments.  The text model owns 20 physical decoder modules;
the middle physical modules 5--14 are each executed twice, producing the exact
30-entry logical schedule.  There are no MeSH memory slots or routers.

The route exposes only FORMAL training.  Its default text initialization is:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244
```

`--model-path` supplies this text-only initialization.  `--resume-from` is
reserved for a complete composite checkpoint produced by this route; the two
arguments are not interchangeable.  The formal contract is 8 GPUs, micro 8,
GA 4, effective batch 256, three epochs, LR 1e-3 to zero, 5% warmup, BF16,
save every 500 optimizer steps, and retention of the latest four complete
checkpoints.  With the current drop12 manifest this is expected to be 11,343
optimizer steps with 568 warmup steps.

Submission entry point:

```bash
bash code/RSmol/run_audio_5_10_5_recursive_mellow_formal_5090.sh
```

User arguments are appended after the canonical defaults and may select an
explicit fresh output directory or a valid FORMAL `--resume-from` checkpoint.

