# Formal30 ReasonAQA: 5-10x2-5 MeSH text-backbone comparison

This isolated route is copied from `../reasonaqa_global_batch256_formal_30epochs`, the implementation behind `run_mellow_official_reasonaqa_gbs256_formal30_5090.sh`. Only the text backbone and its weights change. Unmodified modules and raw-audio staging come from the same `../reasonaqa_global_batch256_lr_ablation_5epochs` through `PYTHONPATH`. Original routes and the shared recursive implementation are unchanged. Model weights and data remain on remote Linux.

The fixed text initialization is:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244
```

Load only its HF text weights, including trained routers; never load its `training_state.pt` or historical optimizer/scheduler/RNG/step. HTSAT uses the baseline pretrained weights; c2l/projection and the multimodal optimizer/scheduler start fresh. Architecture: hidden 576, 30 logical / 20 physical layers, 5 + 10x2 + 5, five transient memory slots and three write/read router pairs. The canonical baseline `/models/SmolLM2` tokenizer remains the dataset tokenizer. This is a text-backbone replacement comparison: the recursive checkpoint has additional text-training history, so it is not an architecture-only ablation.

- Eight GPUs, microbatch eight per GPU, accumulation four: effective global batch 256.
- AdamW, betas `(0.9, 0.95)`, weight decay `1e-4`, FP32. All trainable bridge and LM parameters share a step-level 5% warmup and cosine learning rate from `4e-5` to `4e-6`.
- One global L2 gradient norm clip of 0.5 after accumulation and unscale. HTSAT is frozen in eval/no-grad mode; `c2l`, audio projection, and LM remain trainable.
- When source `filepath2` is empty, the second slot is a zero waveform; audio-read failure fallback is unchanged.
- Text input is prompt through EOS, then answer through EOS, with right padding only after the answer EOS. Audio prefix length is 260. The causal attention mask excludes trailing padding. CE targets only answer tokens and answer EOS, normalized by the global count of valid targets over all ranks and accumulation microbatches.
- Prompt and answer caps remain 129 and 250 tokens including EOS. Truncation preserves EOS.

Submit from the remote `code/RSmol` directory after synchronizing code through GitHub:

```bash
bash run_mellow_official_reasonaqa_gbs256_formal30_5_10x2_5_mesh_5090.sh
```

The wrapper uses the baseline PASS path audit/mapping defaults, stages raw audio to `/dev/shm`, and submits to `pdgpu-5090` with the same eight GPUs, 32 CPUs, 256G, container and `mellow_c8204d8` environment. It directly trains 30 epochs: no smoke, shortened schedule or extra optimizer steps. `--audit-report`, `--mapping-jsonl`, and `--output-dir` override paths only; compare using the same data inputs. Text initialization and training hyperparameters are fixed.

Default output root:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_formal30_5_10x2_5_mesh_5090/formal_30epochs_<RUN_TAG>/
```

An explicit fresh output directory must be empty:

```bash
bash run_mellow_official_reasonaqa_gbs256_formal30_5_10x2_5_mesh_5090.sh \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_formal30_5_10x2_5_mesh_5090/formal_30epochs_my_run
```

Startup checks (read-only; no model forward/backward or optimizer steps):

1. Before audio staging, `--sources-only` checks the protected training Python AST against `baseline_sources.json`. Dataset, Mellow/audio, sampler, optimizer/scheduler, loss and training code stay identical. Trainer AST must equal the baseline after removing only allowed loading/audit/identity hooks; decoder AST differs only at text construction. Comments, formatting, BOM and newline style do not change the Python hash. Shell launchers, requirements text and unused/shadowed modules are historical provenance only: byte differences appear in `provenance_only_changes` and do not abort the job. Actual changes to protected Python (including prompt strings) are rejected with the file and expected/actual AST hashes.
2. Compare the runtime config against the baseline writer's actual config expression, allowing only the text weight path and identity metadata. Reject CLI overrides of audited training/data/model sections.
3. Record text/baseline HF artifact hashes, HTSAT hash, canonical tokenizer compatibility, data input hashes, code hashes and actual Python/Torch/Transformers/torchaudio/CUDA/cuDNN/TF32 settings. Startup/resume identity checks only persistent text/HTSAT/tokenizer artifacts, actual training Python AST and software/precision settings; raw source hashes are report-only. Hostname, PID, timestamps, job/output paths, staging reports and former `/dev/shm` paths are not compatibility requirements. Never install/upgrade the environment or change TF32 settings.
4. Require complete recursive state loading and FP32. Match baseline attention implementation and cache/output defaults; do not inherit the old adapter's BF16 audio casts or forced `use_cache=False`. Avoid global AutoModel registration so ordinary SmolLM2 remains ordinary.
5. Each text load executes the exact baseline SmolLM2 load to retain its RNG consumption, then isolates additional recursive initialization inside saved Python/NumPy/CPU/CUDA RNG states. Rank 0 reconstructs the baseline model on CPU inside preserved RNG state to require identical audio initial parameters/buffers and post-construction CPU RNG. This adds CPU loading time, no training updates.
6. Audit each rank's FP32 and trainability/freeze status and save audio/RNG hashes. Final checkpoint must satisfy all baseline optimizer/scheduler/epoch/batch checks plus this route's source identity, 20 physical text layers and router presence.

Outputs: `source_comparison_audit.json`, `runtime_30epochs.yaml`, `staging_report.json`, `comparison_audit.json`, `checkpoints/<job-id>/initialization/rank_<rank>.json`, epoch checkpoints, and `final_comparison_audit.json`. Final model: `checkpoints/<job-id>/model--epo-30.ckpt`. Logs: `code/RSmol/log/mellow_official_reasonaqa_formal30_mesh.<RUN_TAG>.JOB.log`.

Resume only this route's schema-v2 full checkpoint with identity schema 2 and compatible training code/environment/model artifacts. A new compute node, new staging root and new output directory are supported. Each submission stages the current dataset afresh; the exit trap deletes that job's temporary audio tree on normal/error exit. Resume and final checkpoint audit never require a previous job's stage tree to survive. Use an epoch checkpoint and a new output directory, for example:

```bash
bash run_mellow_official_reasonaqa_gbs256_formal30_5_10x2_5_mesh_5090.sh \
  --resume-checkpoint /actual/run/checkpoints/JOB_ID/model--epo-10.ckpt \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_formal30_5_10x2_5_mesh_5090/resume_from_epoch10
```

Resume retains baseline semantics: rank main-process RNG and optimizer/scheduler are saved, but epoch-internal DataLoader cursors and persistent-worker full states are not. Do not claim sample/augmentation-identical continuation. Ordinary SmolLM2, older MeSH audio, Mellow-v0 and Stage 4 text checkpoints are rejected as audio resume.

Local validation: `python -m unittest discover -s tests -p test_mellow_formal30_mesh_comparison.py -v` (20 tests, including wrapper drift, core-code drift, rebuilt staging and post-cleanup checkpoint audit), Python compilation, Bash `-n`, source equivalence and whitespace checks. The first remote submission completed staging but stopped before config generation/training because the old gate rejected the baseline submit wrapper's byte hash. This revision makes that hash report-only without changing training. No local PyTorch/data/weights exist; remote loading/training of the fixed route has not been run. The recursive implementation targets Transformers 4.54.1, while the old environment snapshot lists 4.47.1; actual installed version and training API compatibility remain remote checks. No automatic environment or training fallback is provided. Actual memory, loss and completion require the formal job logs/checkpoint. No evaluator is implemented for this new route; do not mix older evaluator contracts.
