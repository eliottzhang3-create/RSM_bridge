# Mizar S1 fixed20 with USAD2-Small, GBS256, 3 epochs

This is a fully isolated audio-encoder variant copied from `../mizar_stage1_fixed20_gbs256_3epochs`. It replaces the frozen HTSAT frontend with locally loaded USAD2-Small and a trainable Mellow-style mapper. The parent route and other encoder routes remain untouched. Local source code is synchronized through GitHub; model weights, manifests, audio, checkpoints, and outputs remain on the Linux server.

## Fixed training agreement

The route keeps the parent training/data recipe: the same 1,477,492-row hybrid Mizar manifest and resolved-path mapping; same online fixed20 decoding, async byte staging and collective switch; same text/tokenizer/templates, rank RNG and sample order; same 8 GPUs × microbatch 8 × accumulation 4 = global batch 256; 3 epochs, 5,771 updates per epoch, 17,313 total updates, 866 warmup updates, and 116 dropped rows per epoch. The answer-only loss, `attention_mask=None`, FP32 forward, autocast off, TF32 off, GradScaler, AdamW groups/betas/decay, Mizar warmup-cosine floor, GradNormTracker, checkpoint/resume state, and output audit remain inherited from the parent implementation.

The model is initialized from the local base SmolLM2 directory and the frozen local USAD2-Small weights. This is a new run: HTSAT/other-route checkpoints cannot initialize or resume it.

| Setting | Same as the HTSAT parent |
| --- | --- |
| Manifest | hybrid 1,477,492 rows, all 8 source counts unchanged |
| Batch / duration | 8 GPUs, microbatch 8, accumulation 4, GBS256, 3 epochs |
| Optimizer updates | 5,771 per epoch; 17,313 total; 866 warmup |
| AdamW | mapper LR 6e-4; full LM LR 3e-4; betas (0.9, 0.999); epsilon 1e-8 |
| Decay | 0.01; bias and tensors with at most one dimension excluded |
| Schedule / clipping | Mizar LambdaLR warmup-cosine, floor 0.1; original GradNormTracker |
| Precision | FP32; autocast off; TF32 off; original GradScaler enabled |
| Text / CE | prompt 256, answer 300; PAD `!` (17), SEP/EOS 0; shifted answer-only CE |
| Trainable modules | full base SmolLM2 and projection; audio backbone frozen/eval |

## Audio and mapper contract

Raw audio decoding and fixed20 policy are unchanged. Decode to float32, average channels, resample to 32 kHz, then center-crop or right-pad a single original recording to 20 seconds. For an explicit pair, independently center-crop/right-pad both recordings to 10 seconds and concatenate in original order. The dataset returns one `[B,640000]` waveform. The model resamples this complete waveform once to 16 kHz and makes one USAD2 forward; it does not split into separate encoder calls.

The required USAD2-Small config is 16 kHz, 384 hidden dimensions, 12 layers and 6 attention heads. The verified 20-second forward returns `x=[B,999,384]` and `x_lengths=[999]`. The encoder is loaded from the explicit local `USAD2` Python package with local safetensors only, then frozen and held in eval mode. No Hugging Face dynamic-module cache or network fallback is used. The loader checks exact state loading and the 20-second tensor contract.

USAD2 has no CLS output. For each fixed20 audio, calculate a valid-frame mean as one global summary, prepend it to the 999 frame features, and pass all 1,000 vectors through the mother Mellow projection: bias-free `Linear(384,576)`, GELU, bias-free `Linear(576,576)`, dropout 0.5, residual addition, LayerNorm. Keep the projected global vector. Apply non-overlapping temporal average pooling of size 4 to the 999 projected frames; the last group averages its three valid frames. The resulting mapper output is `1 global + 250 temporal = 251` tokens. This is fixed, non-learned pooling, not a convolution.

The LM sequence is `audio[251] + SEP[1] + prompt[256] + answer[300] = 808` vectors. The audio prefix is 251 tokens; prefix including SEP and prompt is 508 tokens. The audio mapper and all SmolLM2 parameters train; all USAD2-Small parameters remain frozen.

## Assets and environment

Defaults on the remote server:

```text
USAD2-Small: /hpc_stor03/sjtu_home/jinwei.zhang/models/USAD2
SmolLM2:     /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2
Conda:       mellow_c8204d8
```

The USAD2 package must include `__init__.py`, `config.json`, `configuration_usad2.py`, `modeling_usad2.py`, `usad_model.py`, `usad_modules.py`, and nonempty `model.safetensors`. An empty `__init__.py` is valid. The resolved directory must be named `USAD2`, as in the verified local import. The pre-submit asset checker uses only the standard library. At runtime the route imports source from the asset directory itself and uses the existing `mellow_c8204d8` environment (Torch 2.11.0+cu128, torchaudio 2.11.0+cu128, Transformers 4.54.1, safetensors 0.8.0, NumPy 2.2.6 and soundfile 0.14.0 in the checked environment); it does not install or upgrade packages.

The inherited `models/htsat.py` and `models/config.py` are inactive copies retained from the parent. The active frontend imports `models/usad2.py`; it does not import HTSAT or require torchlibrosa. USAD2 source files and weights remain external assets.

## Local checks

From repository root:

```powershell
python -m py_compile code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_usad2_small/contract.py
python -m py_compile code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_usad2_small/models/usad2.py
python -m py_compile code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_usad2_small/models/mellow.py
python -m py_compile code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_usad2_small/models/decoder.py
python -m py_compile code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_usad2_small/train.py
python -m py_compile code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_usad2_small/write_config.py
python -m unittest discover -s tests -p "test_mizar_s1_usad2_small_route.py" -v
python -m unittest discover -s tests -p "test_mizar_s1_usad2_small_shared_start.py" -v
python -m unittest discover -s tests -p "test_mizar_s1_usad2_offline_loader.py" -v
git diff --check
```

The workspace Python does not contain PyTorch. The 24 local tests use standard-library fixtures and mocked command execution; they do not load real tensors or weights. The parent regression suite has 18 tests. `audit_reference.py` also checks byte-identical data/staging files, function ASTs, the complete optimizer loop (apart from a one-time gradient diagnostic), the original projection class and initialization, 1,344 official text/RNG cases, 8 tokenizer cases, 17,314 LR positions and 10,000 scalar clipping cases. Results are recorded in `reference_audit.json`.

`runtime_checks.py` supplies 11 additional CPU tensor checks for the remote environment, including one whole-waveform resample/encoder call, projection-before-pooling ordering, final partial-group handling, frozen features/trainable projection, CE offsets and optimizer resume. These use a synthetic encoder and do not load real weights. Run them optionally after GitHub sync:

```bash
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate mellow_c8204d8
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_usad2_small
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python runtime_checks.py
```

The user's standalone real-weight CPU 20s forward has passed (`x=1x999x384`, `x_lengths=[999]`, `mel=1x1998x128`). Integration tensor checks, 8-GPU training and peak memory at microbatch 8 have not been run from this workspace. Formal training performs real source-by-source encoder/LM/CE probes, checks the first mapper/LM gradients, and verifies that the complete frozen USAD2 state remains unchanged at the end.

## Formal submission

After GitHub sync, log into the remote server and run:

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_usad2_small_gbs256_5090.sh \
  --usad2-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/USAD2 \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_usad2_small_gbs256_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

The submitter uses `pdgpu-5090`, 32 CPU cores, 256 GB RAM, 8 GPUs and one node. It creates a distinct output directory, validates the local USAD2 assets, prepares/reuses the CPU index, then submits formal training. There is no smoke-mode gate; the first actual encoder/LM/CE probes run inside the job before training updates. Do not resume from an HTSAT or CED run. A resume must use a full checkpoint from this exact USAD2 route, source version, assets, data and training contract.

The expected final model is `formal/checkpoints/model--epo-3.ckpt`; intermediate checkpoints are written at the inherited six points. `formal/training_report.json` must say `PASS` with 17,313 optimizer steps before treating the training as complete. The local workspace cannot load the remote weights or run the 8-GPU job, so those validations happen only after submission.

Resume only from a complete checkpoint created by this route, with the same source, assets, environment and training contract. The route restores optimizer, scheduler, scaler, GradNorm, sampler cursor and per-rank RNG. For example:

```bash
RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_usad2_small_gbs256_5090/formal_REPLACE_WITH_EXISTING_TIMESTAMP
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_usad2_small_gbs256_5090.sh \
  --usad2-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/USAD2 \
  --stage-workers 4 --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-REPLACE_WITH_SAVED_STEP.ckpt"
```

Inspect `formal/config_audit.json`, `formal/first_gradient_rank*.json`, `formal/metrics.jsonl` and `formal/training_report.json`. The metrics include GPU peak allocated memory and staging/read-mode status. Existing HTSAT evaluation entrypoints have not been adapted to this USAD2 route.
