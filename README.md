# RSM_bridge

## Handoff status for the next Codex session (updated 2026-09-02)

This section is the authoritative handoff summary.  It supersedes older
planning text below whenever the two conflict.  “Verified” means that the
user supplied a remote PASS report or a concrete successful checkpoint;
“pending” means that the code exists or is planned but has not passed the
required remote audit yet.

### Verified experiments

* **Original SmolLM2-135M:** the single-GPU `pdgpu-5090` inference smoke
  passed.  The source checkpoint is
  `/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2`.
* **15R recursive model (main line):** Stage 1 conversion and inference
  audits passed.  It has 15 unique physical layers (source layers
  `1,3,5,...,25,27,30`, using the checked-in conversion metadata) executed
  twice for a 30-layer logical forward.  Stage 2 single-GPU toy-batch and
  real-tokenizer short training audits passed.  Stage 4 Gates A, C, D and E
  passed, and formal training was run on eight `pdgpu-5090` GPUs.  The
  first-epoch checkpoint is
  `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4/formal-20260829_183543/checkpoint-step-009244`;
  an epoch-2 continuation was also produced at
  `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4/formal-epoch2-20260831_165359/checkpoint-step-009244`.
* **5-10-5 recursive model:** conversion, Stage 1 audits, Gate A, Gate D and
  Gate E passed.  The model has prefix layers 1–5, middle source layers
  `6,8,10,12,14,16,18,20,22,24`, and suffix layers 26–30; the middle ten
  physical layers are executed twice.  Its source/target model directory is
  `/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5`, and its formal
  checkpoint is
  `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-20260830_172928/checkpoint-step-009244`.
* **5-10-5 non-recursive linear baseline:** conversion and smoke/evaluation
  plumbing exist.  A formal run reached step 5,500 and later stopped because
  of the trainable-parameter checksum audit at step 5,832; that checksum
  check was explicitly disabled for the continuation experiment.  Do not
  describe this run as a completed formal training unless a later report
  proves it.
* **Stage 3 evaluation:** the checked-in evaluator uses `lm-eval==0.4.12`
  with local parquet overlays for HellaSwag, the 57-subject MMLU group,
  GSM8K, ARC-Easy and ARC-Challenge.  The original model, 15R checkpoints,
  and 5-10-5 checkpoints have been evaluated through the remote wrapper;
  every claimed score must be tied to the corresponding external output
  directory and JSON report.

### Current pending experiment: dynamic 5-10×r-5 with selective BPTT (方案 B)

Updated 2026-09-02: the authoritative variant is now `5-10xr-5` (written
conceptually as `5-10×r-5`), with `r∈{4,5,6,7}` sampled once per complete
optimizer step using increasing power weights `w(r)=r²`: probabilities are
`16:25:36:49 / 126`. Rank 0 samples a stateless `(seed, optimizer_step)`
value and broadcasts it to all eight ranks; all sixteen microbatches in that
accumulation window use the same `r`. Non-training inference defaults to
`r=7`.

The formal target is currently limited to **4,500 optimizer steps** with
`ceil(0.05×4500)=225` warmup steps. The dataset and other training
hyperparameters remain unchanged. The isolated implementation is pending
remote conversion, Stage 1, Gate A, Gate D and Gate E audits; local checks do
not constitute a remote PASS.

Implemented isolated entry points are:

```text
code/RSmol/recursive_model_5_10xr_5.py
code/RSmol/scripts/convert_stepwise_5_10xr_5.py
code/RSmol/scripts/convert_stepwise_5_10xr_5.sh
code/RSmol/scripts/smoke_recursive_5_10xr_5.py
code/RSmol/scripts/smoke_recursive_5_10xr_5.sh
code/RSmol/scripts/train_stage4_5_10xr_5_ddp.py
code/RSmol/scripts/train_stage4_5_10xr_5_ddp.sh
code/RSmol/run_convert_stepwise_5_10xr_5_3090.sh
code/RSmol/run_smoke_recursive_5_10xr_5_3090.sh
code/RSmol/run_stage4_5_10xr_5_3090.sh
```

The earlier `5-10x7-5` files are superseded handoff artifacts and are not the
authoritative implementation. Existing 15R, fixed 5-10-5 recursive and
5-10-5 linear files remain isolated and unchanged.

The Stage 1 smoke audits every `r=4,5,6,7` for dynamic forward/backward
lengths, cache slots and incremental decoding, generation r-persistence,
reload, and the selective-gradient contract.  Gate A additionally exercises
all four backward schedules before its synchronized sampled-r, 16-microbatch
optimizer update.

The architecture contract for this **new isolated variant** must not modify
the 15R or existing 5-10-5 files.  The source mapping is:

```text
prefix:  source layers 1,2,3,4,5
middle:  source layers 6,8,10,12,14,16,18,20,22,24 (10 layers)
suffix:  source layers 26,27,28,29,30
forward: prefix + (middle × r) + suffix = 50/60/70/80 logical executions
storage: 20 unique physical decoder layers
```

方案 B means that all selected middle loops are executed in the forward pass,
but only the final four middle calls receive shared-parameter gradients; the
hidden-state autograd path traverses every selected middle call.  The suffix
and prefix remain trainable.  This is a research variant and is **not verified
yet**.  The superseded fixed-r files
`recursive_model_5_10x7_5.py`, `scripts/convert_stepwise_5_10x7_5.py`,
`scripts/smoke_recursive_5_10x7_5.py`, and
`scripts/train_stage4_5_10x7_5_ddp.py` are superseded handoff artifacts;
they are not part of the dynamic variant and must not be used for conversion,
smoke, or training.

The requested output model directory is expected to be outside the Git
checkout, for example
`/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10xr-5`.  New-variant
conversion and training wrappers target `pdgpu-3090`; the already running
15R formal job continues to use `pdgpu-5090`.

The following fixed-r entry-point list is retained only as historical evidence
of the superseded draft; the authoritative dynamic entry points are listed
above:

```text
code/RSmol/scripts/convert_stepwise_5_10x7_5.py
code/RSmol/scripts/smoke_recursive_5_10x7_5.py
code/RSmol/scripts/train_stage4_5_10x7_5_ddp.py
code/RSmol/run_smoke_recursive_5_10x7_5_3090.sh
code/RSmol/scripts/smoke_recursive_5_10x7_5.sh
code/RSmol/scripts/train_stage4_5_10x7_5_ddp.sh
```

The intended validation sequence is deliberately shorter than the original
15R route: (1) convert from the actual SmolLM2 `config.json`; (2) run the
new model's Stage-1-style forward/cache/generation/reload audit on one
`pdgpu-3090`; (3) run the isolated synthetic Gate A DDP audit (Gate A does
not need Gate-B data); (4) reuse the existing Gate-B JSON for Gate D and
FORMAL, then run Gate D (ten optimizer steps) and Gate E resume smoke on eight
`pdgpu-3090` GPUs; only after those checks pass, launch the 4,500-step formal
training.  Do not run Gate B again for this variant unless the audit manifest
changes.

### Shared data and training contract

The training corpus is the `text` column of
`/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset`.  The audit
found 85 parquet shards, 10,058,156 raw rows and approximately 25.6 GB.  Each
shard has 119 row groups and 118,331 or 118,332 rows.  Gate B is a CPU-only
three-shard streaming sample (`pyarrow.parquet.ParquetFile`), not a full
token-length census; it must not create a shared Arrow cache.  The external
Gate-B report currently used by multi-card gates is supplied with
`RSMOL_STAGE4_AUDIT_REPORT`.

Unless a variant explicitly documents a different contract, the formal
training settings are: eight ranks, micro-batch 8 per rank, gradient
accumulation 16, effective global batch 1,024 samples, context length 1,024,
9,244 optimizer steps, AdamW with `betas=(0.9,0.95)`, `eps=1e-8`,
`weight_decay=0.1`, `amsgrad=false`, maximum learning rate `2e-4`, minimum
learning rate `2e-5`, linear warmup for `ceil(0.05*9244)=463` steps followed by
cosine decay, progress/loss/LR/speed every 10 steps, and complete checkpoints
every 500 steps plus the final step, retaining at most three verified
checkpoints.  Loss is next-token prediction over every non-padding token;
only dynamic padding positions are masked.  Gradient accumulation must use
the window-global valid-token denominator exactly once.

All model/data/checkpoint/output paths above are remote and outside the Git
checkout.  GPU work must be submitted through the corresponding `vc submit`
wrapper; never run model loading or training directly on the login node.


## Stage 3 offline benchmark evaluation (2026-08-28)

Stage 3 is the current evaluation gate for the original SmolLM2-135M and the
untrained recursive SmolLM2-15R checkpoint.  It evaluates the two external
models on the official lm-evaluation-harness tasks `hellaswag`, `mmlu`,
`gsm8k`, `arc_easy`, and `arc_challenge`.  Stage 4 training is paused while
these pre-up-training baselines are collected; this section does not claim a
remote evaluation has passed.

The implementation is:

```text
code/RSmol/run_stage3_eval_5090.sh       # vc submit wrapper; one pdgpu-5090 GPU
code/RSmol/scripts/evaluate_stage3.sh   # submitted-job runtime layer
code/RSmol/scripts/evaluate_stage3.py   # validation, local overlays, lm_eval API, audit
tests/test_stage3_static.py              # dependency-light Windows contract tests
```

The only supported remote entry point is:

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only origin main
cd code/RSmol
RSMOL_STAGE3_MODEL=both bash run_stage3_eval_5090.sh
```

Use the same wrapper for the three supported modes:

```bash
RSMOL_STAGE3_MODEL=both RSMOL_STAGE3_VALIDATION_ONLY=1 bash run_stage3_eval_5090.sh  # preflight only
RSMOL_STAGE3_MODEL=both RSMOL_STAGE3_SMOKE=1 bash run_stage3_eval_5090.sh             # two docs/task
RSMOL_STAGE3_MODEL=both bash run_stage3_eval_5090.sh                                  # formal run
```

The wrapper requests `pdgpu-5090`, container
`docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`, `-c 8 -m 32G -g 1
-n 1`, and environment `rsmol`.  It does not submit a job from this
checkout.  `RSMOL_STAGE3_MODEL` may be `original`, `recursive`, or `both`.
The external model defaults are the supplied paths
`/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2` and
`/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-15R`; override them with
`RSMOL_STAGE3_ORIGINAL_MODEL` and `RSMOL_STAGE3_RECURSIVE_MODEL` when running
another external checkpoint.
`RSMOL_STAGE3_TASKS` is a comma-separated subset of the five official task
names.  `RSMOL_STAGE3_SMOKE=1` limits each task to two documents; set
`RSMOL_STAGE3_VALIDATION_ONLY=1` to validate paths, parquet fields, checksums,
installed task YAMLs, and protocol without loading a model.  The fixed seed
default is `0`, batch size is `1`, dtype is `bfloat16`, and device is
`cuda:0`.  Use `RSMOL_STAGE3_OUTPUT_DIR` (or the alias
`RSMOL_STAGE3_OUTPUT_ROOT`) for an explicit fresh output root.  No global
`--num_fewshot` is passed: native task defaults are retained.

The read-only benchmark snapshot must exist at
`/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets` with this exact
source mapping:

```text
Rowan_hellaswag/data/{train,validation,test}-00000-of-00001.parquet
cais_mmlu/{57 subject dirs}/{dev,test,validation}-00000-of-00001.parquet
openai_gsm8k/main/{train,test}-00000-of-00001.parquet
allenai_ai2_arc/ARC-Easy/{train,validation,test}-00000-of-00001.parquet
allenai_ai2_arc/ARC-Challenge/{train,validation,test}-00000-of-00001.parquet
```

The 57 MMLU directory names expected by the preflight are:
`abstract_algebra`, `anatomy`, `astronomy`, `business_ethics`,
`clinical_knowledge`, `college_biology`, `college_chemistry`,
`college_computer_science`, `college_mathematics`, `college_medicine`,
`college_physics`, `computer_security`, `conceptual_physics`, `econometrics`,
`electrical_engineering`, `elementary_mathematics`, `formal_logic`,
`global_facts`, `high_school_biology`, `high_school_chemistry`,
`high_school_computer_science`, `high_school_european_history`,
`high_school_geography`, `high_school_government_and_politics`,
`high_school_macroeconomics`, `high_school_mathematics`,
`high_school_microeconomics`, `high_school_physics`, `high_school_psychology`,
`high_school_statistics`, `high_school_us_history`, `high_school_world_history`,
`human_aging`, `human_sexuality`, `international_law`, `jurisprudence`,
`logical_fallacies`, `machine_learning`, `management`, `marketing`,
`medical_genetics`, `miscellaneous`, `moral_disputes`, `moral_scenarios`,
`nutrition`, `philosophy`, `prehistory`, `professional_accounting`,
`professional_law`, `professional_medicine`, `professional_psychology`,
`public_relations`, `security_studies`, `sociology`, `us_foreign_policy`,
`virology`, and `world_religions`. The helper directories `all/` and
`auxiliary_train/` are not used.

The local overlay sets only `dataset_path: parquet` and
`dataset_kwargs.data_files`.  Official task YAML prompt formatting,
`doc_to_text`, choices, targets, `process_docs`, few-shot behavior, answer
extraction, and metrics remain unchanged.  HellaSwag uses validation;
MMLU uses the original 57-subject `mmlu` group, `dev` few-shot and `test`.
The pinned 0.4.12 MMLU template omits its `num_fewshot` field, so the runtime
records that incompatibility and applies a task-scoped `num_fewshot=5` API
override only for the MMLU group (never a global override);
GSM8K uses `openai_gsm8k/main` train/test with deterministic generation
(`do_sample=false`, `temperature=0`); and ARC Easy/Challenge are reported as
separate test `acc`/`acc_norm` tasks.  `auxiliary_train` and GSM8K `socratic`
are never selected.

The remote runtime pins `lm-eval==0.4.12`, `transformers==4.54.1`, and
`datasets==3.6.0`; `pyarrow` must also be importable for the parquet schema
preflight.  It fails before inference if these versions or the installed task
configs differ.  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
and `HF_DATASETS_OFFLINE=1` are set before importing the evaluation stack;
`local_files_only=True` is passed to Transformers.  The recursive model
imports and calls `register_auto_class()` before lm_eval's HF backend creates
the model.  Its audit requires logical depth 30, physical depth 15, loops 2,
unique parameter storage, and a forward hook trace `0..14` twice.  No
`device_map` is used.

Each model has its own fresh external output directory, and each task has a
fresh child directory.  A non-empty output is rejected.  A formal run writes
`lm_eval_results.json`, optional `log_samples.json`, `task_protocol.json`,
`summary.json`, `summary.csv`, `run_config.json`, and `audit_report.json`.
Task/process stderr and runtime logs are preserved under the diagnostic log
root, while the audit records package versions, git
identity, model/tokenizer paths, the complete parquet manifest and SHA-256
checksums, protocol/config details, sample/failure counts, timestamps, and GPU
information.  Model/checkpoint artifacts and benchmark data remain outside
the Git checkout.  Diagnostic/runtime and vc-submit logs default to the
established checkout directory
`/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol/log`; set
`RSMOL_STAGE3_LOG_ROOT` (or the legacy alias
`RSMOL_STAGE3_SUBMIT_LOG_ROOT`) to choose another allowed log directory.
The `log_samples.json` files under each external output task directory are
benchmark sample artifacts, not runtime logs.
The evaluator also neutralizes lm_eval 0.4.12's non-essential `git describe`
result-metadata probe; some containers expose a non-executable `git`, which
otherwise causes a scored task to be reported as failed during finalization.

Local checks (which do not require CUDA, lm_eval, or benchmark data) are:

```powershell
python -m py_compile code/RSmol/scripts/evaluate_stage3.py
python -m unittest tests.test_stage3_static
python -m unittest discover -s tests
git diff --check
```

Only these local/static checks can be reported from a Windows checkout.  The
actual validation and benchmark results remain pending remote execution on
the supplied GPU/data environment.

Historical planning notes below contain earlier statements about Stage 3
up-training and the absence of a recursive checkpoint.  They predate the
current Stage 3 implementation and must not be read as the present protocol;
the current status is the offline benchmark gate above, with Stage 4 paused.

## Stage 4 DDP audit and fixed-step pilot (2026-08-27)

Stage 4 is paused as of 2026-08-28 pending completion and review of the
Stage 3 benchmark baselines above.  The existing DDP code and gate notes are
historical implementation context; no Stage 4 job is part of the Stage 3
evaluation protocol.

Stage 4 is implemented in `code/RSmol/scripts/train_stage4_ddp.py` and is
intended for the external `/hpc_stor03` task output area only.  It does not
write model parameters, checkpoints, or reports into either Git checkout.
The five historical gates and the independent formal mode are:

* Gate A: synthetic DDP recursive-model audit (forward `0..14` twice,
  backward second loop then first, label shift, finite/non-zero gradients,
  optimizer update, rank checksums and exact 16 microbatches).
* Gate B: CPU-only single-process 85-Parquet footer/schema pre-audit plus a
  deterministic three-shard streaming/tokenizer sample; run
  `code/RSmol/scripts/audit_stage4_dataset.py` directly (it never submits a
  GPU job or starts `torchrun`).
* Gate C: real-data 8-rank short pilot (`--max-optimizer-steps 1` or `2`).
* Gate D: fixed-step ten-optimizer-step smoke by default, with periodic
  complete checkpoints.
* Gate E: resume smoke in a new output directory; optimizer, scheduler, RNG,
  step, manifest, and one `data_cursors_by_rank` cursor per DDP rank are
  restored.  The iterator performs a coarse
  epoch/shard/complete-microbatch skip from the saved cursor; because the
  bounded in-shard shuffle buffer cannot be reconstructed exactly, reports set
  `data_cursor_restored=false` and never claim bitwise-exact data resume.
  Gate E rejects legacy checkpoints that contain only a single `data_cursor`.

Formal Stage 4 training is now a separate `--gate FORMAL` mode.  It enforces
the production contract of 9,244 optimizer steps, schedule domain 9,244,
linear warmup of 463 steps, `save_every=500`, and retention of the newest
three verified complete checkpoints.  The mandatory save steps are
500, 1000, ..., 9000, and 9244; rank 0 writes each checkpoint through the
existing atomic full-checkpoint contract and all ranks synchronize at saves.
The optimizer is explicitly `AdamW(betas=(0.9, 0.95), eps=1e-8,
weight_decay=0.1, amsgrad=false)`.  Accumulation uses unreduced local loss
sums and one window-global valid shifted-token denominator across all 16
microbatches; because DDP averages rank gradients, each backward uses the
scale `world_size / global_window_valid_tokens` and does not divide by the
accumulation count a second time.
Formal reports include `mode=formal`, target/actual steps, cumulative samples
and tokens, stop reason, final checkpoint, and retained checkpoints.  A
formal run that exhausts data before 9244 is reported as not reached and does
not claim PASS.  Formal checkpoints can be resumed with
`--gate FORMAL --resume-from`; the scheduler domain remains 9244 and the
optimizer/RNG/cumulative counters, manifest, and per-rank cursors are
restored.  Resume uses the same coarse bounded-shuffle cursor semantics and
does not claim bitwise-exact data order.  Formal checkpoints are intentionally
rejected by Gate E, which remains the bounded two-step resume smoke.

Use the same fixed vc wrapper for a remote formal submission:

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
RSMOL_STAGE4_GATE=FORMAL \
RSMOL_RECURSIVE_OUTPUT_DIR=/external/recursive-checkpoint \
RSMOL_STAGE4_AUDIT_REPORT=/external/stage4-gate-b/stage4_gate_B_audit.json \
bash run_stage4_5090.sh
```

The fixed configuration is `code/RSmol/stage4_default_config.json`:
`world_size=8`, `micro_batch_size=8`, `gradient_accumulation_steps=16`,
`learning_rate=2e-4`, `max_lr=2e-4`, `min_lr=2e-5`,
`scheduler_type=linear_warmup_cosine`, `log_interval_steps=10`,
`context_length=1024`, and `max_optimizer_steps=10`.  Gate D therefore
targets `1,280` samples/rank (`160` local microbatches).  The independent
FORMAL mode targets `9,244` steps with global consumption of `9,465,856`
samples at the global effective batch of `1024`.  Warmup defaults
to `ceil(0.05 * total_steps_for_schedule)` (one step for Gate D), followed by
cosine decay to `min_lr`.  With the sample-only Gate B report, startup fail-fast
checks exact footer raw-row capacity for every deterministic rank assignment;
effective trainable rows remain explicitly unknown until rank-local training
streaming observes them.  Reports retain raw rows, effective-row scope, and
remaining raw rows.

Gate B runs directly on the remote server with the CPU environment.  It reads
all 85 Parquet footers, but streams/tokenizes only three shards selected from
the sorted manifest with `seed=0` (override with `--sample-shards` only when
needed).  It reports the sampled token-length distribution, p50/p95/p99,
empty/blank text, missing/source distribution and examples, and over-context
rows.  These content statistics are explicitly marked as a three-shard sample,
not as a full-corpus estimate.  Progress is printed for every footer and at
least every 10,000 sampled rows.  Direct `pyarrow.parquet.ParquetFile`
iteration uses bounded batches and does not create a Datasets Arrow cache;
optional HF cache variables point to a dedicated local `/tmp` root and the
report records the cache audit.  The script refuses to overwrite a non-empty
output directory:

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
python -u code/RSmol/scripts/audit_stage4_dataset.py \
  --model-path /external/recursive-checkpoint \
  --data-dir /hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset \
  --output-dir /external/stage4-gate-b-YYYYMMDD \
  --report-path /external/stage4-gate-b-YYYYMMDD/stage4_gate_B_audit.json \
  --world-size 8 --seed 0 --context-length 1024 \
  --sample-shards 3 --progress-every-rows 10000
```

The script records all 85 shard footers, deterministic eight-rank
`rank_shards`, per-rank raw-row capacity, the Gate D smoke target of `1,280`
samples/rank, and formal global/remaining-row totals for the separate
`9,244`-step target.  It does not claim per-rank effective
rows from the three-shard sample; the training gates use exact footer row
capacity and perform their own rank-local streaming/tokenization.  For
Gate C/D, pass its external JSON via `RSMOL_STAGE4_AUDIT_REPORT`; this keeps
rank 0 from rescanning the full corpus during a multi-card training launch.
Those GPU-backed gates use the `vc submit` wrapper (fixed 8 x 5090,
`-c 32 -m 256G -g 8 -n 1`).  For Gate D omit `RSMOL_STAGE4_GATE` (it defaults
to `D`).  Gate E uses a fresh `RSMOL_STAGE4_OUTPUT_DIR` and sets
`RSMOL_STAGE4_RESUME_FROM` to the latest complete Gate D checkpoint.  Every
gate emits JSON reports and the training runtime appends resource samples to
`runtime_monitor_rank*.jsonl`; cache variables are redirected to task-local
`/tmp` paths.

这是 Recursive SALM（暂称 RSM）实验的代码同步仓库。仓库的主要作用是在本地 Windows 机器上由 Codex
协助编写、检查和维护代码，再通过 GitHub 将代码同步到远程 Linux HPC 服务器执行模型转换、预训练、up training
和音频推理训练。

本 README 是项目的长期背景记录和操作约定。后续新的 Codex 窗口应先阅读本文件，再开始修改代码或设计远程实验。
其中“已验证”表示有实际日志、报告或检查结果支持；“计划”表示当前研究路线；“待确认”表示不能自行猜测，
需要由用户提供远程信息或实验结果。

## 快速开始与当前结论

这是一个“本地写代码、GitHub 中转、远程 HPC 提交运行”的代码同步仓库。任何新 Codex 窗口开始工作前，必须先阅读
本 README；涉及远程 GPU、模型、数据、训练、评估或审计的操作，必须通过已提交到仓库的 `vc submit` wrapper 执行。
不能在登录节点直接运行模型加载或推理，也不能把远程权重、数据、checkpoint、日志和 outputs 提交到 GitHub。

截至 2026-08-26，原始 SmolLM2-135M 的 `pdgpu-5090` 推理 smoke 已成功通过。当前项目仍处于“原始模型基线完成，
Stepwise 层转换尚未开始”的阶段；尚不存在可用于后续 up training 的递归 checkpoint。

当前最重要的路径和入口：

```text
本地 Windows checkout：
C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge

远程 Linux checkout：
/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM

首个远程 smoke 提交入口：
code/RSmol/run_smoke_smollm2_inference_5090.sh
```

远程后续运行的固定顺序：

```text
本地修改 → 本地检查 → git commit → git push
→ 远程 git pull --ff-only
→ 通过对应的 vc submit wrapper 提交
→ 用户带回作业日志、报告和 traceback
```

## 第一部分：背景原理与不可违反的工程规则

### 1. 项目目标

本项目参考以下两类工作：

- [Mellow: A Small Audio Language Model for Reasoning](https://arxiv.org/abs/2503.08540)：探索小规模音频语言模型的
  音频理解和推理能力，采用音频编码器、非线性映射器和语言模型的模块化组合，并使用面向音频推理的数据训练。
- [Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA](https://arxiv.org/abs/2410.20672)：
  通过层参数共享把普通 Transformer 转换为递归 Transformer，并可用按深度/层位置区分的低秩适配模块放松严格共享。

项目的总体路线是：

```text
SmolLM2-135M 原始语言模型
        ↓
Stepwise 层压缩：唯一层数缩减为一半
        ↓
递归执行：压缩后的层块循环两次
        ↓
使用 SmolLM2-135M-10B 做文本 up training，恢复压缩造成的性能损失
        ↓
接入音频编码器和映射器
        ↓
使用类似 Mellow 的音频-文本训练方法，获得音频推理能力
```

这里的“缩小”主要指参数存储中的唯一 Transformer 层数量减少；循环两次后，逻辑计算深度仍然与原始模型相当。
因此，参数量、显存占用、计算量、训练吞吐和推理延迟必须分别测量，不能仅凭层数减半推断所有资源都减半。

### 2. 语言模型基座

第一阶段使用 Hugging Face 的 [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
作为原始、未压缩的语言模型基座。

当前应按公开配置理解其关键结构为：

- 约 135M 参数的 decoder-only Transformer；
- 30 个 Transformer 层；
- hidden size 为 576；
- intermediate size 为 1536；
- 9 个 query attention heads、3 个 key/value heads；
- 词表约 49,152；
- 最大位置长度为 8,192；
- 官方 checkpoint 以 BF16 权重发布，并使用 Llama 兼容的模型结构。

上述信息是设计起点。真正运行时必须读取远程实际 checkpoint 的 `config.json` 并审计，不能只依赖 README 或模型名。
如果实际配置与这里不同，实际 checkpoint 配置和通过脚本记录的配置优先。

2026-08-26 的已提交推理 smoke 已读取并验证远程实际配置：上述 30 层、hidden size、attention heads、词表、最大位置
长度和 BF16 均与 checkpoint 一致；实际加载类为 `transformers.models.llama.modeling_llama.LlamaForCausalLM`，
参数量为 `134,515,008`。

### 3. Stepwise 递归压缩

#### 3.1 目标构型

当前目标是把原始 30 层模型变为 15 个唯一层组成的共享 block，并将这 15 层按顺序执行两次：

```text
原始模型：L = 30 个各不相同的层
目标模型：K = 15 个唯一层，logical depth = 15 × 2
```

代码实现不应把“复制两份层”误认为“参数共享”。正确实现必须让两个循环使用同一组基础层参数；如果使用层位置专属
适配器，则适配器参数另行统计，不得把它们混入共享基础层参数。

如果未来使用不同层数的模型，必须先检查 `L % 2 == 0`，再由配置计算 `K = L / 2`；不能把 15、30 等数值散落在代码中。

#### 3.2 Stepwise 初始化原则

Stepwise 方法不是随机初始化，也不是简单取原始模型的前一半层。它应当从原始模型的 30 个层中按照深度间隔选择
15 个源层来初始化共享 block，同时保留原始网络的浅层和深层信息；论文描述中明确强调首层和末层的保留。

因此，转换脚本必须：

1. 明确记录原始层索引到目标唯一层索引的映射表；
2. 保证映射包含原始模型的开头和结尾信息；
3. 对 attention、MLP、normalization 等属于该层的参数整体搬运，不能只复制部分线性层；
4. 保留 embedding、final norm、LM head、tokenizer 和特殊 token 配置的兼容性；
5. 在写出压缩 checkpoint 前，检查所有目标共享层都有来源、形状一致、dtype 一致；
6. 保存转换元数据，包括源 checkpoint、源配置、目标层数、循环次数、层映射、随机种子和代码版本。

主线默认使用 Stepwise，不应未经记录改成 Average、Lower、随机初始化或从头训练。其他初始化方法若要比较，必须使用
独立的实验名、输出目录和评估记录。

#### 3.3 递归 forward 的语义

递归模型的 forward 应保持普通因果语言模型的输入/输出语义：

```text
h = embedding(input_ids)
for loop_idx in [0, 1]:
    for layer_idx in [0, ..., 14]:
        h = shared_layers[layer_idx](h, position information, attention mask, cache state)
h = final_norm(h)
logits = lm_head(h)
```

实现时必须特别验证：

- 两次循环的层顺序完全一致；
- 两次循环确实访问同一组共享基础参数；
- causal attention mask、padding mask、position ids 和 KV cache 在第二次循环中仍然正确；
- `use_cache=True` 和 `use_cache=False` 的 logits 语义一致；
- 转换前后 tokenizer、embedding、LM head 和 loss 的接口没有被无意改变；
- 单步 forward、批量 forward、训练反向传播和生成分别通过独立 smoke 检查。

递归计算不是普通的“把 30 层切成前 15 层再重复调用一个 Python 函数”这么简单。位置编码、缓存、梯度图、
activation checkpointing 和分布式 wrapping 都可能受第二次循环影响，必须由测试明确覆盖。

### 4. Relaxed Recursive Transformer 参考原则

Relaxed Recursive Transformer 是本项目的重要参考方向，但不能在没有实验记录的情况下自动替代严格共享的递归基线。

严格递归模型中，同一个唯一层在两次循环中完全共享参数。Relaxed 版本则允许每个逻辑深度或每个循环位置附加独立的低秩
增量。例如基础线性变换可以写成：

```text
h = W_shared x + ΔW_loop x
ΔW_loop x = B_loop A_loop x
```

参考论文中的关键点是：

- 基础层参数仍然共享；
- 不同循环/深度使用不同的低秩 LoRA 增量；
- LoRA rank 控制严格共享与原始非共享模型之间的容量和参数量折中；
- 可以对“原始层权重 - 共享层权重”的残差做 truncated SVD，并以 `B = UΣ`、`A = Vᵀ` 的方式初始化低秩增量；
- 论文中的 relaxed up training 不是普通的“冻结基础模型、只训练零初始化 LoRA”，因为共享基础层也需要学习代表多个深度的中心参数。

当前工程规则：

- 首先保留一个不带 relaxed LoRA 的 Stepwise Recursive 基线，用于判断单纯参数共享的损失；
- 如果实现 layer-wise/loop-wise LoRA，必须明确它属于哪一个逻辑层、哪一次循环，并单独统计参数量；
- SVD 初始化必须保存 rank、截断策略、残差来源和误差统计；
- 严格递归 checkpoint 与 relaxed checkpoint 不得混用；
- 不得把普通 PEFT 的默认零初始化、全局 LoRA 或随机适配器称作论文中的 Relaxed Recursive Transformer。

### 5. 文本 up training

文本恢复阶段使用 Hugging Face 数据集
[EleutherAI/SmolLM2-135M-10B](https://huggingface.co/datasets/EleutherAI/SmolLM2-135M-10B)。

当前已知数据集信息：

- modality 为 text；
- 有一个 `train` split；
- 数据记录至少包含 `text` 和 `source` 字段；
- 数据集卡显示约 10,058,156 条记录、约 25.6 GB 下载大小；
- 它从 SmolLM2 语料中抽样，包含 FineMath、Stack-Edu、InfiMM-WebMath、Cosmopedia V2、FineWeb-Edu 和 DCLM-Edu
  等来源的组合；
- 数据集名称中的“10B”是数据规模提示，不应替代远程实际 tokenizer 统计得到的 token 数。

当前远程数据落盘形式已经确认：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset/
  data/train-00000-of-00085.parquet
  ...
  data/train-00084-of-00085.parquet
```

该目录当前约占 24G。它位于 Git checkout 外部，只能作为远程输入使用；当前 smoke 不读取它。Parquet 的完整字段、总行数、
tokenizer 后的长度分布和训练切分策略，必须在后续数据准备/训练任务中通过独立的 `vc submit` 作业确认，不能根据文件名
或数据集名称猜测。

up training 的目的不是重新预训练一个新模型，而是让由原始 SmolLM2 权重转换得到的递归模型适应新的参数共享结构，
尽量恢复原始模型的语言建模能力。训练和比较时至少要保留以下基线：

1. 原始 SmolLM2-135M，不做层压缩；
2. Stepwise Recursive，转换后不 up training；
3. Stepwise Recursive，使用该数据集 up training；
4. 如实现 relaxed 版本，再加入 relaxed Recursive 的独立比较。

每个阶段都应记录有效 token 数、sequence packing 方式、context length、batch size、gradient accumulation、学习率、
warmup、scheduler、weight decay、dtype、训练步数、随机种子、代码 commit 和 checkpoint 路径。不能用“跑了若干 epoch”
替代精确 token/step 记录，也不能只凭训练 loss 宣布恢复成功。

文本 up training 的首要验收对象是：

- 转换前后的模型结构和参数归属；
- causal LM loss 与 label shift；
- 原始模型和递归模型在相同 tokenizer、数据窗口和评估脚本下的 perplexity/accuracy；
- checkpoint 保存后重新加载的输出一致性；
- 递归模型的唯一参数量、逻辑层数和实际循环次数。

### 6. 音频接入与 Mellow 风格训练

音频阶段参考 Mellow 的模块化方法，而不是把音频波形直接塞入语言模型。目标结构是：

```text
audio waveform
    ↓
pretrained audio encoder
    ↓
audio feature sequence
    ↓
non-linear mapper / projector
    ↓
SmolLM2 hidden-size compatible audio representation
    ↓
audio-conditioned text prompt
    ↓
recursive language model generates reasoning response
```

Mellow 的公开实现和论文采用 HTSAT 类音频编码器、映射器和 SmolLM2 语言模型，并训练音频 grounded reasoning 数据；
其任务可以包含单音频理解、音频问答、音频蕴含和双音频差异解释。我们的项目沿用“音频编码器 → mapper → LM”的方法论，
但以下内容目前仍属于待确认项，不能从 Mellow 自动推断：

- 最终使用哪一个音频编码器及其远程权重路径；
- 首版支持一个音频还是两个音频输入；
- 音频采样率、最大时长、特征帧率和音频 token 压缩策略；
- mapper 的层数、激活函数、输出维度和是否使用边界 embedding；
- 音频编码器、mapper、共享 Transformer 基础层和 loop-wise LoRA 的冻结/训练划分；
- 使用 Mellow 的 ReasonAQA、其子集、重新构造的数据，还是用户指定的其他音频推理数据；
- 文本 prompt、答案格式、chain-of-thought 是否保留，以及 loss 只监督答案还是监督完整响应。

音频模型的目标 hidden size 必须与 SmolLM2 的实际 `hidden_size` 对齐。任何新增音频 token、特殊 token、prefix mask、
position id 和 label mask 都必须在模型接口和数据 collator 中有明确的契约。音频前缀通常不应直接作为文本目标，
音频相关位置应按实验定义使用 `-100` 或其他明确的非监督标记。

音频阶段应先完成“文本递归模型可稳定训练和加载”的 gate，再接入编码器。音频能力提升必须与纯文本能力变化分开报告，
不能把音频训练后的 loss 下降直接解释成语言模型恢复成功。

### 7. 研究阶段隔离与证据规则

当前建议的阶段边界如下：

```text
Stage 0  原始 SmolLM2 加载、tokenizer 和文本基线
Stage 1  Stepwise 层映射与严格递归模型构造
Stage 2  递归模型无训练 forward / generation / loss / reload 审计
Stage 3  原始与未训练递归模型的离线 benchmark evaluation（当前）
Stage 4  递归文本训练与 checkpoint 验证（paused，待 Stage 3 基线完成）
Stage 5  音频编码器、mapper 和音频输入协议接入
Stage 6  Mellow 风格音频推理训练与评估
```

阶段之间不得混淆：

- 转换前 checkpoint、递归 checkpoint、relaxed checkpoint 和音频 checkpoint 使用独立目录；
- 任何 checkpoint 只有在结构、配置和 trainability 审计通过后，才能作为下一阶段输入；
- “目录存在”“有中间 loss”“训练步数达到预期”都不是完成证据；
- 完成状态至少需要终端成功信息、最终 checkpoint 检查、关键统计和对应评估结果；
- 远程日志中未出现的事实不能写入 README 作为已验证结论；
- 实验设计有变化时，更新 README 的当前状态和日期，并明确旧结果是否仍可比较。

## 第二部分：本地、GitHub 与远程协作规则

### 8. 仓库角色和边界

本仓库是 code-sync workspace，不是完整的远程运行环境。Git 中允许并鼓励保存：

- Python 源代码；
- shell 脚本和 `vc submit` 包装脚本；
- 小型 JSON/YAML/TOML 配置；
- 数据 schema、转换元数据和小型测试 fixture；
- README、实验记录和审计说明；
- 不包含模型参数的大型流程配置。

以下内容原则上只保留在远程服务器或其他专用存储，不得提交到 GitHub：

- 原始或派生模型权重；
- optimizer/scheduler/RNG checkpoint；
- 大型数据集、音频文件、缓存和下载目录；
- `outputs/`、`checkpoints/`、训练日志、TensorBoard 文件和 profiler trace；
- 远程临时文件、环境目录、密钥、token 和机器本地配置。

如果某个预处理步骤生成了新的文件，提交前必须判断它是“小型、可复现的代码配置”，还是远程运行产物。后者应加入
`.gitignore`，而不是为了方便调试提交进仓库。

### 9. 当前本地信息

本项目本地 Windows checkout：

```text
C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
```

GitHub 中转仓库：

```text
https://github.com/eliottzhang3-create/RSM_bridge
```

当前本地路径和 GitHub 远端已经确认。新的远程 Linux checkout 根目录不能沿用旧项目路径，也不能根据仓库名称自行猜测。
当前已经确认的新项目远程信息如下：

```text
远程 Linux checkout（Git 仓库目录）：
/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM

远程模型目录（仓库外，只读输入）：
/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2

远程文本数据目录（仓库外，只读输入）：
/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset
```

远程 checkout 目录名 `RSLAM` 与 GitHub 仓库名 `RSM_bridge` 不一致是允许的；Git 连接由仓库内的
`.git/config` 和 `origin` URL 决定，不依赖本地目录名。模型目录中目前已准备 SmolLM2 checkpoint 和 tokenizer
文件；数据目录中目前已准备 `data/` 及其 README。模型权重、数据集和后续生成的 checkpoint 不进入本仓库。

当前本地项目初步结构为：

```text
C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge\
  .gitignore
  README.md
  code\
    RSmol\
      log\
      plugins\
      scripts\
  data\                  # 远程数据不放入此处
  models\                # 远程权重不放入此处
  outputs\               # 运行产物，已被 .gitignore 忽略
```

Git 不记录空目录；目录只有在其中出现未被忽略的代码或配置文件后，才会随提交同步到 GitHub。

### 10. 本地与远程的职责划分

本地 Windows 侧：

- Codex 修改代码、README、配置和提交脚本；
- 使用 Windows 路径；
- 做静态检查、语法检查、轻量级单元测试和 Git diff 检查；
- 通过 GitHub push 代码；
- 不要求本地拥有远程模型权重或完整数据集。

远程 Linux HPC 侧：

- 执行模型下载/加载、数据准备、训练、评估和 checkpoint 审计；
- 使用 Linux 绝对路径；
- 保存权重、数据、outputs、日志和缓存；
- 先 `git pull` 获取本地已 push 的代码，再通过提交脚本运行任务；
- 将关键日志、错误 traceback、文件列表和审计结果带回对话。

Codex 默认只能操作本地 checkout，不能假设自己能登录或控制远程服务器。涉及远程状态的结论必须来自用户提供的
远程日志、命令输出或已同步的远程文件内容。

Windows 路径和 Linux 路径不可混写。例如，README 中必须明确区分本地的
`C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge` 与远程的 `/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM`；脚本中也必须
根据运行平台使用对应路径。

### 11. 建议的远程目录约定

当前远程根目录已经确认；下面的逻辑分区用于说明代码、远程输入和运行产物的边界，实际目录名可根据后续项目设计调整：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/
  README.md                 # 与 GitHub 同步
  .gitignore                # 与 GitHub 同步
  code/RSmol/               # 当前项目代码和提交脚本，同步
    run_smoke_smollm2_inference_5090.sh
    scripts/
      smoke_smollm2_inference.sh
      smoke_smollm2_inference.py
    log/                     # 远程作业日志，不提交
  data/                      # 仅放小型可复现配置或 manifest，不放数据集
  models/                    # 仅放代码约定，不放权重
  outputs/                   # 远程运行产物，不提交
```

真正的模型和文本数据位于 checkout 外部：`/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2` 和
`/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset`。不要把这两个路径复制到 Git 仓库内。

如果数据或模型位于仓库外的共享存储，README 应记录其准确远程路径和只读/可写属性。公共数据目录若规定只读，
只能在私有工作目录生成 manifest、索引和进度文件，不能改写、解包覆盖或向公共目录写入。

### 12. Git 同步流程

#### 12.1 首次连接远程目录

远程 `/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM` 是用户预先创建的空目录时，在其中执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
ls -la
git clone https://github.com/eliottzhang3-create/RSM_bridge.git .
git remote -v
git status --short --branch
```

`git clone ... .` 会把 GitHub 仓库直接克隆到当前目录，并自动建立 `origin`。如果目录并非空目录，先停止，
不要删除其中内容，也不要在未确认的情况下强行初始化或覆盖；应先检查目录中的文件是否需要保留。

首次同步的顺序是：本地提交并 push，远程再 clone；如果远程已经先 clone 过，则在 push 完成后执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only origin main
```

之后每轮代码同步都遵循“本地 `commit` → 本地 `push` → 远程 `pull --ff-only`”。远程默认只拉取代码，
不从远程反向 push，避免把服务器上的临时改动混入 GitHub。

#### 12.2 当前首个 SmolLM2 推理 smoke

当前首个 smoke 只验证远程已经准备好的原始 SmolLM2-135M，不涉及 Stepwise 转换、递归模型、音频模型、数据集或
ms-swift 注册。它必须作为一个 GPU 作业提交，不能直接在登录节点运行 Python 模型加载或推理。

代码结构为：

```text
code/RSmol/
  run_smoke_smollm2_inference_5090.sh       # 仅调用 vc submit
  scripts/
    smoke_smollm2_inference.sh              # 作业内 runtime wrapper
    smoke_smollm2_inference.py              # 模型加载、forward、generation 和报告
```

当前 smoke 的固定远程配置为：

```text
conda environment: rsmol
queue:             pdgpu-5090
container:         docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1
resource:          -c 8 -m 32G -g 1
model:             /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2
```

远程拉取当前 Git commit 后，在仓库中执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only origin main
cd code/RSmol
bash run_smoke_smollm2_inference_5090.sh
```

可通过环境变量覆盖 prompt、生成长度或报告路径；这些变量会由提交 wrapper 安全传递给作业内 runtime：

```bash
RSMOL_SMOKE_PROMPT='Gravity is' \
RSMOL_SMOKE_MAX_NEW_TOKENS=32 \
bash run_smoke_smollm2_inference_5090.sh
```

成功证据至少包括提交作业日志中的 `[result] status=OK`、生成文本，以及 `outputs/RSmol/` 下的 JSON 报告。
失败时保留完整 traceback。该 smoke 不保存模型权重，不读取文本数据集，也不修改远程模型目录。

2026-08-26 已完成的 smoke 结果：

```text
ACTIVE_ENV             = rsmol
Python                 = 3.10.20
PyTorch                = 2.11.0+cu128
CUDA                   = 12.8
GPU                    = NVIDIA GeForce RTX 5090
Transformers           = 4.54.1
model class            = LlamaForCausalLM
parameters             = 134,515,008
parameter dtype        = bfloat16
forward logits         = (1, 3, 49152)
cache type             = DynamicCache
cache length           = 3
generated new tokens   = 32
result                 = OK
```

该作业使用 `local_files_only=True` 和离线环境变量，确认远程本地 checkpoint 能够被 Transformers 加载并生成文本。
这只证明原始模型基线的推理链路正常，不代表 Stepwise 转换、递归参数共享、训练或音频能力已经通过。

当前 smoke 的生成报告示例路径为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/outputs/RSmol/
  smollm2_inference_smoke_20260826_142647.json
```

本地标准流程：

```text
修改代码或文档
→ 本地静态检查
→ git status
→ git diff
→ git add
→ git commit
→ git push origin main
```

远程标准流程：

```text
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only origin main
确认 commit 和脚本版本
提交对应的 vc submit 包装脚本
查看远程日志和输出
```

提交前必须检查：

- 是否误加入权重、数据、checkpoint、outputs 或日志；
- shell 脚本是否使用正确的远程路径；
- 训练脚本、submit wrapper 和 README 是否描述同一个实验配置；
- 是否与已有输出目录或 job name 冲突；
- 本次提交是否只包含当前任务范围内的改动。

未经用户明确要求，不使用 `git reset --hard`、`git checkout --` 等会丢失用户改动的命令。

### 13. 远程任务提交规则

所有涉及 GPU、模型加载、推理、训练、数据读取、评估或 checkpoint 审计的远程任务，即使只有几秒或只有一个 batch，
也必须通过 `vc submit` 进入作业节点。登录节点只用于 Git、文件查看和提交命令，不直接运行这些计算任务。

远程长任务必须由两层脚本组成：

1. runtime script：在远程作业节点中激活环境、设置变量、定位仓库根目录、执行 Python/训练命令并写日志；
2. submit wrapper：调用 `vc submit`，指定队列、容器、CPU、内存、GPU、job name、工作目录和 runtime script。

runtime script 通常应遵循以下模式：

```bash
#!/bin/bash
set -euo pipefail

# 1. 激活远程环境
# 2. 由脚本位置计算仓库根目录
# 3. 设置 PYTHONUNBUFFERED、随机种子和必要的 CUDA 变量
# 4. 检查模型、数据、配置和输出目录
# 5. 执行训练/评估/审计
# 6. 打印清晰的成功或失败标志
```

submit wrapper 应把用户需要覆盖的路径和实验参数通过环境变量安全传入，并把作业日志写入远程 `log/` 或指定日志目录。
不要把大量动态逻辑塞进 `vc submit --cmd` 的单行字符串中；复杂逻辑应放在可审查、可单独测试的 runtime script 里。

旧项目记录的队列资源上限是按请求 GPU 数量计算：每张 GPU 最多 8 个 CPU core、32G 主机内存。因此，若该服务器规则仍然
适用于新项目：

- 单卡通常不超过 `-c 8 -m 32G -g 1`；
- 8 卡通常不超过 `-c 32 -m 256G -g 8`；
- 必须检查实际 `vc submit` 参数，不能只根据脚本文件名中的 `_5090` 或 `_4090` 判断资源池；
- 当前首个 SmolLM2 推理 smoke 已确认使用 `pdgpu-5090`、`rsmol`、上述 container 和单卡 `-c 8 -m 32G -g 1`；
- 后续正式训练的 queue、container、GPU 型号和资源请求仍需按具体任务单独确认。

每个新远程任务提交前，先确认：

- 用户要运行的是 smoke、转换、up training、正式训练、checkpoint save/resume、评估还是数据准备；
- runtime script 是当前 Git commit 中的版本；
- 模型和数据路径确实指向目标版本；
- 输出根目录是新的或明确允许覆盖；
- world size、per-device batch、gradient accumulation 和有效 batch 计算一致；
- 任务是否需要保存 optimizer、scheduler、RNG 和数据位置；
- 任务是否满足队列资源限制。

### 14. 本地检查与远程证据

修改 Python 文件后，至少执行：

```bash
python -m py_compile path/to/changed_file.py
```

在 Windows PowerShell 中如有必要，也可以逐个运行 Python 文件的导入级 smoke，但不能因为本地没有 CUDA、模型或远程
数据就宣称远程训练链路已经通过。

远程 GPU smoke、模型推理和训练只能通过对应的 `run_*.sh` submit wrapper 执行。不能绕过 wrapper 直接运行 runtime
shell 或 Python 文件；作业内验证必须保留 job log、报告和实际 Git commit。

远程任务的合格记录至少应包含：

- 运行的 Git commit；
- 实际模型、tokenizer、音频编码器和数据路径；
- 实际环境、容器、GPU 数量和资源申请；
- 关键配置、seed、有效 token/样本数和训练步数；
- 关键 stdout/stderr 或日志位置；
- checkpoint 结构、参数归属和 reload 检查；
- 最终成功标志和失败时的完整 traceback。

模型结构、层映射、共享参数、LoRA/mapper trainability、loss mask 和 checkpoint contract 都应有自动化审计，而不应
只靠人工看几行日志。任何“通过”结论都要注明是本地检查、远程 smoke、正式训练中间结果还是最终审计。

### 15. README 维护规则

README 既是项目说明，也是后续 Codex 的实验记忆。以后新增内容应遵循：

- 当前主线放在靠前位置；
- 历史路线必须标注为 historical/superseded；
- 每次状态变化写明日期；
- 已验证、计划、待确认、失败和禁止复用的 checkpoint 分开记录；
- 远程路径、环境、资源、数据版本和 checkpoint 路径必须完整且可复制；
- 如果代码文件名保留旧名但运行语义已经变化，必须显式说明；
- 不把推测写成事实，不把中间结果写成正式完成；
- 不删除仍然对复现有价值的失败原因，但要说明它是否仍影响当前主线。

## 第三部分：当前项目状态（截至 2026-08-26）

### 16. 已完成事项

- 本地 Windows checkout 为 `C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge`；
- GitHub 中转仓库为 `https://github.com/eliottzhang3-create/RSM_bridge`；
- 远程 Git checkout 为 `/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM`；
- 远程 conda environment 为 `/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/rsmol`；
- 远程模型目录为 `/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2`；
- 远程文本数据目录为 `/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset`；
- 已完成原始 SmolLM2-135M 的 `pdgpu-5090` 单卡推理 smoke；
- 已确认原始模型有 30 个 Transformer 层；当前递归目标仍为 15 个唯一层循环两次；
- 已建立 `code/RSmol/` 下的 runtime、submit wrapper 和 Python smoke 代码结构；
- 已更新 `.gitignore`，远程 outputs、logs、checkpoint 和权重不进入 GitHub。

### 17. 已验证的远程模型与数据

模型目录总大小约 260M，核心文件包括 `config.json`、`model.safetensors`、`tokenizer.json`、
`tokenizer_config.json`、`vocab.json` 和 `merges.txt`。远程 smoke 读取到的配置为：

```text
model_type              = llama
architectures           = LlamaForCausalLM
num_hidden_layers       = 30
hidden_size             = 576
intermediate_size       = 1536
num_attention_heads     = 9
num_key_value_heads     = 3
vocab_size              = 49152
max_position_embeddings = 8192
torch_dtype             = bfloat16
```

数据目录约占 24G，文件命名为 `data/train-xxxxx-of-00085.parquet`。当前 smoke 没有读取数据集；Parquet 的完整字段、
总行数、tokenizer 后长度分布和实际训练采样策略仍未验证。

### 18. `rsmol` 环境快照

以下版本来自用户在远程 `rsmol` 环境中的静态检查，不代表所有组件都已经在训练作业中验证：

| 组件 | 版本 | 当前用途 |
|---|---:|---|
| Python | 3.10.20 | 运行时 |
| pip | 26.0.1 | 包管理 |
| torch | 2.11.0+cu128 | 模型与训练核心 |
| transformers | 4.54.1 | SmolLM2 加载与生成 |
| tokenizers | 0.21.4 | tokenizer |
| safetensors | 0.8.0 | checkpoint 读取 |
| huggingface-hub | 0.36.2 | Hub 接口，当前使用本地离线文件 |
| datasets | 3.6.0 | Parquet/数据集读取 |
| pyarrow | 23.0.1 | Parquet 后端 |
| accelerate | 1.13.0 | 模型加载和后续训练支持 |
| numpy | 2.2.6 | 数值基础库 |
| peft | 0.18.1 | 后续 relaxed/layer-wise LoRA 候选 |
| trl | 0.18.0 | 后续训练扩展候选 |
| tensorboard | 2.20.0 | 日志记录 |
| ms-swift | 4.4.2 | 已安装，但当前不作为递归模型实现基础 |
| modelscope | 1.35.3 | ms-swift 相关依赖 |
| torchaudio | 2.11.0+cu128 | 后续音频阶段候选 |
| librosa | 0.11.0 | 后续音频预处理候选 |
| soundfile | 0.14.0 | 后续音频文件读取候选 |

`python -m pip check` 已返回 `No broken requirements found.`。当前缺少的 `deepspeed`、`bitsandbytes`、`flash-attn`、
`wandb`、`av`、`resampy` 和 `omegaconf` 暂不视为当前原始推理 smoke 的问题；是否安装必须由具体训练/音频任务决定，
不能为了“凑齐依赖”盲目升级或安装。

### 19. 已完成的原始推理 smoke 证据

提交脚本使用：

```text
queue       = pdgpu-5090
container   = docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1
resource    = -c 8 -m 32G -g 1
environment = rsmol
```

2026-08-26 作业日志显示：

```text
GPU                  = NVIDIA GeForce RTX 5090
CUDA                 = 12.8
model parameters     = 134,515,008
parameter dtype      = bfloat16
forward logits       = (1, 3, 49152)
cache type           = DynamicCache
cache length         = 3
generated new tokens = 32
result               = OK
```

该 smoke 使用 `local_files_only=True` 和离线环境变量，验证了原始本地 checkpoint 的加载、BF16 forward、KV cache 和
确定性 generation。报告路径示例为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/outputs/RSmol/
  smollm2_inference_smoke_20260826_142647.json
```

这项结果只证明原始模型推理基线正常，不证明递归模型、层共享、训练、数据管线或音频模型正常。

### 20. 当前代码入口与下一步

当前首个 smoke 的唯一远程执行入口是：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only origin main
cd code/RSmol
bash run_smoke_smollm2_inference_5090.sh
```

下一阶段按以下顺序推进：

1. 编写 Stepwise 转换脚本，从实际 `config.json` 读取层数并构造明确的 30→15 层映射；
2. 保存递归模型 checkpoint 及转换元数据，审计参数形状、dtype、层映射和唯一参数量；
3. 实现严格 Recursive baseline 的自定义 `PreTrainedModel`/forward，并通过独立的 `vc submit` smoke 验证 forward、
   cache、generation、loss 和 reload；
4. 使用文本数据进行 up training，保留原始模型、未训练递归模型和训练后递归模型的可比评估；
5. 递归文本模型稳定后，再决定是否加入 Relaxed layer-wise LoRA；
6. 最后确定音频 encoder、mapper、音频数据协议和 Mellow 风格训练流程。

### 21. 尚未完成、不可提前假定的事项

- Stepwise 的最终层索引选择和映射算法尚未写入代码；
- 尚无 15 个唯一层循环两次的递归 checkpoint；
- 尚未完成递归模型的 KV cache、loss、反向传播、checkpoint reload 和分布式包装验证；
- 尚未开始文本 up training，也没有正式 loss、perplexity 或 benchmark 结果；
- 尚未注册 RSM 递归模型到 ms-swift；
- 尚未确定 relaxed LoRA 的作用模块、rank、SVD 初始化和冻结策略；
- 尚未确定音频 encoder、mapper、音频权重、数据集和评估 benchmark；
- 正式训练任务的 batch、学习率、token budget、保存/恢复策略和资源配置仍需逐任务确认。

在上述事项确认前，任何 Codex 窗口都不得把计划写成已完成事实，不得擅自改写远程模型/数据路径、checkpoint 路径或正式
训练超参数，也不得绕过 `vc submit` 直接运行远程 GPU 任务。

## 参考资料

- [Mellow paper](https://arxiv.org/abs/2503.08540)
- [Mellow reference implementation](https://github.com/soham97/Mellow)
- [Relaxed Recursive Transformers paper](https://arxiv.org/abs/2410.20672)
- [SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [SmolLM2-135M-10B dataset card](https://huggingface.co/datasets/EleutherAI/SmolLM2-135M-10B)

## 22. Independent `5_10xpoisson_parcae` variant

This section documents the new namespace only; the historical 5-10xr-5
implementation and its audit history above remain unchanged.  The new model
maps source 30 layers to 20 physical modules: prefix 5, shared middle 10,
and suffix 5.  Its logical depth is `5 + 10*T_i + 5`, ranging from 50 to 110.

Training samples one independent `T_i` for every sequence in every local
microbatch from the exact truncated standard Poisson distribution:

```text
P(T=k) = [exp(-7) * 7^k / k!] / Z,  k in {4,5,6,7,8,9,10}
Z = sum(exp(-7) * 7^k / k!) for k=4..10
```

The private CPU generator seed is derived from
`(base_seed, rank, optimizer_step, microbatch_index)`.  The local maximum
`Tmax` is used for left alignment, `tau_i=Tmax-T_i`; aligned steps before
`tau_i` are no-op/copy operations.  No depth vector is broadcast between
ranks.  The first five outputs are `e`, and injection always uses
`PN(e)=PreludeNorm(e)`:

```text
u_t = Abar(h_t) + Bbar(PN(e))
h_{t+1} = MiddleBlockStack(u_t)
```

`A_log`, `dt_bias`, `B`, softplus `dt`, exponential decay, and identity-B
initialization are recorded in the converted checkpoint metadata; all three
injection tensors carry `_no_weight_decay=True` and are placed in the actual
AdamW `weight_decay=0` group.  PreludeNorm is a single `LlamaRMSNorm` call,
and `h0` is a fresh per-forward, per-sequence truncated-normal like-init
state (`state_init_std=initializer_range`, with the explicit embedding scale
recorded in metadata), not a learned parameter.  The hidden-input graph is
retained for all calls.  Earlier aligned recurrent
calls use `torch.func.functional_call` with detached injection and all ten
middle-block parameters; only the final four aligned calls retain parameter
gradient edges.  Training always sets `use_cache=False`; scalar inference
accepts explicit `r/T=4..10` and defaults to 7, with cache slots rebuilt for
the selected schedule.

Run the isolated workflow through the vc submission wrappers (these wrappers
submit jobs; do not run `scripts/*.sh` directly on a login node):

```bash
RSMOL_5_10XPOISSON_PARCAE_SOURCE_CHECKPOINT=/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2 \
  bash code/RSmol/run_convert_stepwise_5_10xpoisson_parcae_3090.sh
bash code/RSmol/run_audit_stage1_5_10xpoisson_parcae_3090.sh
bash code/RSmol/run_stage4_5_10xpoisson_parcae_3090.sh  # default GATE=D, 8 GPUs
RSMOL_5_10XPOISSON_PARCAE_STAGE4_GATE=FORMAL \
  bash code/RSmol/run_stage4_5_10xpoisson_parcae_3090.sh  # 9244 steps
```

The `scripts/convert_stepwise_5_10xpoisson_parcae.sh`,
`scripts/audit_stage1_5_10xpoisson_parcae.sh`, and
`scripts/train_stage4_5_10xpoisson_parcae_ddp.sh` files are vc job payloads.
The `run_*_3090.sh` files are the supported login-node submission entrypoints;
Stage 4 payloads themselves invoke eight-rank `torchrun` inside the job.

The Stage 4 formal target is 9,244 optimizer steps, 16 microbatches per
optimizer window, 8 sequences per local microbatch, and `ceil(5%)=463`
warmup steps.  Acceptance requires `compileall`, `git diff --check`, the
dedicated Poisson/schedule/PreludeNorm/additive-injection/selective-gradient
static tests, Stage 1 vector-gradient isolation plus real scalar inference
audits for every `r=4..10`, default `r=7`, logical cache slots and incremental
append, generation propagation, and AutoModel reload, and a successful
ten-step real-data smoke.  Conversion writes `conversion_metadata.json` and a
completion marker atomically in the new output directory.  The smoke wrapper
uses eight-rank `torchrun`; it is not a single-process check.

Real-data Stage 4 streams only the rank's deterministic hashed shard
assignment.  Every non-empty text row is tokenized with
`add_special_tokens=False, truncation=True, max_length=1024`; rows shorter
than 1024 are retained, and each microbatch is dynamically padded with a
padding-only valid mask.  Fixed shard order rolls over to deterministic
epochs, carrying a partial batch across the boundary.  Checkpoints save the
exact epoch/shard/row/pending-row cursor, so resume does not replay a large
prefix of microbatches.  The preaudit verifies the unchanged 85-shard
manifest, schema, sampled tokenization, and at least one trainable row per
rank; raw parquet row counts are never presented as training capacity.

FORMAL additionally requires runtime `WORLD_SIZE=8`, microbatch size 8,
accumulation 16, context length 1024, 9244 optimizer/scheduler steps,
warmup 463, save interval 500, retention 3, and `max_microbatches=None`.
Formal reports append one compact scalar metric per optimizer step containing
loss, learning rate, valid tokens, depth/Tmax histograms, the numeric
`total_grad_norm`, and its `total_grad_norm_finite_nonzero` status.  Only step 1, checkpoint steps, and the final step carry detailed
gradient audits.  Gate D/E retain the full per-window depth and gradient audit
detail.

Every detailed/audit-point gradient report is fail-closed across all ranks:
physical prefix/middle/suffix layers, the injection group, parameters, and
the `total_grad_norm_finite_nonzero` check must pass before
`optimizer.step` (the total gradient norm is finite and >0).
Checkpoint markers enumerate every model and tokenizer file, including model
index files for sharded weights; resume rejects any incomplete or
non-reloadable offline artifact.

Stage 1 writes separate `scalar_inference_all_r`, `default_r7`,
`cache_contract`, `generation_contract`, and `reload_contract` report entries.
The cache audit checks every logical K/V slot before and after a one-token
append, rejects reuse under a different scalar `r`, and forks CPU/CUDA RNG so
the cache/no-cache prompt comparison uses the same fresh like-init `h0`.
