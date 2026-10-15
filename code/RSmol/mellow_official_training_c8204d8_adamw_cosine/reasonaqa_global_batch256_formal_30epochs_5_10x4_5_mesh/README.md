# Formal30 ReasonAQA: 5-10x4-5 MeSH text-backbone comparison

This is a new isolated route copied from the repaired 5-10x2-5 formal30 route. It is used only for the four-loop, seven-memory-slot text checkpoint below. The original SmolLM2 route, the 5-10x2-5 route, and the shared recursive implementation are not modified by this route.

The fixed text initialization is:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/formal_2epochs_18488steps_20261007_004959_3090_v1/checkpoint-018488
```

Only the HF model artifacts needed by the text model are loaded from this checkpoint. Its optimizer, scheduler, RNG, data cursor, historical step, and training_state.pt are never restored. The multimodal bridge, audio projection, and new AdamW/scheduler start from the same fresh state as the SmolLM2 formal30 baseline. The dataset tokenizer remains the canonical SmolLM2 tokenizer and is checked for vocabulary compatibility.

The x4 model contract is logical_50_physical_20_5_10x4_5_mesh: 20 physical layers execute 50 logical layers as 5 + 10x4 + 5; the attention cache has 50 logical slots; MeSH memory has seven transient slots; and there are five independent read/write router groups (ten router modules). The loader rejects incomplete state dictionaries, non-FP32 weights, a wrong schedule, shared router objects, persistent memory, or the inference-only two-loop ablation flag.

The multimodal training contract is unchanged from the SmolLM2 formal30 entry: 8 GPUs, per-rank batch 8, accumulation 4, effective global batch 256; 30 complete epochs; FP32 with mixed precision disabled, while TF32 follows the baseline environment contract; AdamW betas (0.9, 0.95), weight decay 1e-4, max LR 4e-5, min LR 4e-6, 5% warmup and step-level cosine; global L2 gradient clipping 0.5; the same ReasonAQA inputs, sampler, workers, HTSAT weights and eval/no-grad freeze policy; and the same 260-position audio prefix, prompt/answer EOS and padding rules, causal mask and answer/EOS-only global-token loss.

Submit after synchronizing the repository through GitHub, from the remote code/RSmol directory:

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mellow_official_reasonaqa_gbs256_formal30_5_10x4_5_mesh_5090.sh
```

The wrapper uses the same pdgpu-5090, container, mellow_c8204d8 environment, 8 GPUs, 32 CPUs and 256G allocation as the baseline, but has an independent job name, log name and output root:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_formal30_5_10x4_5_mesh_5090/formal_30epochs_<RUN_TAG>/
```

It performs no smoke or shortened run. Raw-audio staging is rebuilt for every job under a unique /dev/shm directory and removed by the exit trap. Source checks run before staging. Shell-wrapper byte drift, node names, PIDs, timestamps, output paths and old staging paths are provenance only; protected training Python, runtime semantics, the x4 recursive model, weights, tokenizer and software/precision identity remain hard checks. A different compute node and a newly rebuilt staging tree are supported.

Outputs include source_comparison_audit.json, runtime_30epochs.yaml, staging_report.json, comparison_audit.json, rank initialization reports, epoch checkpoints and final_comparison_audit.json. The expected final model is checkpoints/<job-id>/model--epo-30.ckpt. Resume accepts only a schema-v2 checkpoint produced by this x4 route in a new output directory; it never accepts the historical x4 Stage 4 checkpoint as a multimodal resume.

Local validation completed for this isolated route: 20 regression tests, Python compilation and git diff --check. No local PyTorch, remote weights, data or HTSAT are available, so remote loading, CUDA memory and 30-epoch completion remain to be established from the formal job logs and checkpoint.

