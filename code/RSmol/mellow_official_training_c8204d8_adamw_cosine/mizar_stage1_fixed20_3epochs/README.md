# HTSAT / SmolLM2：Mizar Stage 1，3 epochs

这是一条独立训练路线，基于 `../reasonaqa_global_batch256_lr_ablation_5epochs/` 的模型架构，按本地官方 Mizar 源码移植 Stage 1 训练语义。原 ReasonAQA 训练目录与启动脚本保持原样。当前并行配置固定为 **8 GPUs × microbatch 4 × gradient accumulation 1 = global batch 32**。

本地只保存代码；权重和音频由远程计算节点读取。无需在服务器安装官方 Mizar 仓库，也无需下载 CED 权重或生成 CED 特征缓存。

## 采用的官方实现

对应本地下载的 `kaiyangli-ml/Mizar_159M`：

- `configs/s1.yaml.in`：Stage 1 配方。
- `runtime/stage1/training/trainer.py`：优化器分组、调度器、loss、精度及更新顺序。
- `runtime/stage1/data/audiotext_dataset.py`、`template.py`：文本构造、分词及固定长度。
- `runtime/stage1/utils/utils.py`：GradNormTracker。
- `pipeline/build_ced_cache.py`：fixed20 的波形处理规则。

`reference_audit.json` 记录上述参考文件 SHA256 和本路线代码身份；`audit_reference.py` 用标准库从官方源码提取函数，逐项比较文本及随机状态、分词调用参数、完整调度曲线和 GradNorm 标量策略。此审计不代替真实音频、张量或 DDP 验证。

## 数据与音频

固定输入文件：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_filtered.json
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_audiocaps_csv_retry1/resolved_paths.tsv
```

| 来源 | 保留行数 |
| --- | ---: |
| reasonaqa_oneaudio | 756,968 |
| AudioCaps | 113,844 |
| Clotho | 4,798 |
| CompA-R | 249,161 |
| LP-MusicCaps-MTT | 19,632 |
| SpeechCraft | 124,790 |
| TACOS | 42,141 |
| 合计 | **1,311,334** |

MusicCaps 与 SpeechCraft `train-other-500` 已过滤。SpeechCraft 使用共享目录的原始 LibriTTS clean-100/360，音质来源与官方 LibriTTS-R 存在差异。

`stage_data.py` 按审计 TSV 的 `(field, raw_path)` 查找成功映射。ReasonAQA 使用 `filepath1_original`，其他来源使用 `audio_path`；过滤后的行号不用于匹配原审计。映射失败、原文件缺失、空文件或来源计数变化都会中止。

原 manifest 的字段和值保持不变，仅为本次任务生成带 `s1_raw_paths` 的 JSONL 和索引。准备该索引时顺便统计来源数并计算原始行的 canonical SHA256 和映射 SHA256；不是额外扫描原始 JSON 字节，也不是重新生成过滤文件。

原始音频按实际文件去重并复制到独占的 `/dev/shm/mizar_s1_<job-id>`。staging 只做路径映射、容量/文件大小检查和原始字节复制，不解码、重采样、裁剪、补零或生成音频特征。训练在线执行：

1. soundfile 解码为 float32，所有声道取均值，重采样至 32 kHz。
2. 单个原始文件先中心裁剪或右侧补零到 20 秒，再分为连续的两个 10 秒。
3. 真正双文件各自中心裁剪或右补零到 10 秒，再按顺序拼接。只有显式 `filepath2_original` / `audio_path2` 才表示第二个原始文件，并且同样要求成功审计。
4. 两个 10 秒分别进入现有 HTSAT。历史 `filepath1` / `filepath2` 是旧的处理后路径，不能据此判断原始双文件。

smoke 只复制各 rank 实际前 8 批需要的音频，以及每个来源 1 行训练前向探针需要的音频；探针通过训练数据加载器在线解码。正式训练最终复制过滤后全部保留行引用的原始音频。同一文件被多行引用只复制一次，不会整目录复制数据集，也不会复制已排除的 MusicCaps / train-other-500 行专属音频。两种模式都读取完整 manifest 以保持相同采样顺序和数据身份。

当前默认使用后台 staging，启动流程为：

1. 用私有 RNG 重建 `data_order(seed=1234)`、本轮 `torch.randperm(seed=1234+epoch)` 和八个连续 rank 分片；按每个文件在实际训练中最早使用的 batch 排序。初次训练使用第 1 个 epoch 的顺序，resume 使用 checkpoint 的 epoch/batch 游标。
2. 来源探针文件优先，预先复制前 **512 个 global steps** 的去重音频（八卡合计 16,384 次样本使用）。用 `--stage-start-steps N` 调整；`STAGING_PLAN` 会记录 128/256/512/1024/2048 等档位的实际文件数/字节数。
3. 发布 `TRAINING_READY` 后立即启动 trainer；同一个后台 CPU 进程继续按顺序复制剩余文件，每 128 批发布八卡共同的就绪进度。默认 16 个复制线程，可用 `--stage-workers N` 调整。
4. 每个文件先复制到 `.partial`，检查大小后原子重命名。训练在加载下一批前检查就绪状态，追上复制时明确等待，后台失败、进程消失、心跳超时或等待超过 600 秒则报错。索引和文件路径不随复制进度改写。
5. 全量复制完成后发布 `COMPLETE` 和最终 `PASS` 报告，此后不再逐批轮询。已复制文件保留给后续 epochs；开始下一 epoch 前要求全部保留文件完成，包括本轮被 `drop_last` 丢弃的行专属文件。

smoke 的预拷贝门槛上限为 2 步，其余 6 步所需文件走后台流程，仍只执行 8 次真实更新。模型加载期间后台可能已经复制完成；本地并发测试另外主动阻塞复制，验证训练跨越初始边界和等待后续 epoch。原有 8 项小型张量测试仍不自动运行。

后台只做原始字节复制，不解码或占用 GPU；没有归档文件或第二份完整音频副本。它缩短训练启动前的等待，不减少最终全量音频的内存占用，也不能保证复制始终领先训练。当前日志中全量音频为 333,510 个去重文件、173.48 GiB。建立完整 metadata 索引的 CPU 时间仍然存在；并行复制与训练共享申请的 32 CPU / 256 GiB 内存。

复制前检查 tmpfs 和可用 cgroup 内存，额外保留 32 GiB；不足则在训练前明确退出。默认任务申请 256 GiB 主存，是否容得下全部原始音频以 `staging_report.json` 的实际字节数为准。清理只允许带本任务所有权标记的直接子目录，退出时自动执行。

## 序列与可训练参数

```text
HTSAT(10s)[129] + HTSAT(10s)[129] + SEP[1] + prompt[256] + answer[300]
                        audio 258          prefix 515       total 815
```

保留两段各自的 clip token 和原有平均池化，删除两段音频之间的 SEP，只在完整音频 prefix 后保留一个 SEP。该长度来自现有 HTSAT 架构，与官方 CED 的 126 个音频 token 不同。

HTSAT 主干固定 `eval()`、`no_grad()`；原有 c2l（527→768）、残差 projection（768→576，dropout 0.5）和完整 SmolLM2 参与训练。没有新增循环模块或 LoRA。

prompt 和 answer 各自在分词前追加 `" <|endoftext|>"`，然后分别截断/补齐到 256 / 300；若 EOS 被截断，不强制插回。PAD 为已有词表的 `!`（当前 SmolLM2 为 ID 17）；SEP 与 EOS 都是 ID 0。LM 接收 `attention_mask=None`、`labels=None`、`use_cache=False`，固定 padding 位置参与因果注意力。

外部 next-token CE 只预测 answer，忽略从 `tokenizer.pad_token_id` 解析出的 PAD 目标（当前为 ID 17）；EOS（ID 0）是正常监督目标。prompt / audio 不计 loss。先求每个 rank 有效 answer token 的均值，再由 DDP 平均梯度，与旧路线的全局 token 加权均值不同。

## 优化与采样

| 设置 | 当前实现 |
| --- | --- |
| Epochs / 并行 | 3；8 × 4 × 1，GBS32 |
| Optimizer | AdamW，betas=(0.9, 0.999)，eps=1e-8 |
| 基础学习率 | 完整 LM 3e-4；c2l + projection 6e-4 |
| Weight decay | 0.01；bias 和 ndim≤1 参数为 0 |
| Schedule | 官方 LambdaLR，5% warmup + cosine，最低比例 0.1；optimizer 后调用 scheduler |
| 梯度处理 | 官方 GradNormTracker，L2/max 初值 1/10，overdrive 2.5，momentum 0.995，共用最小缩放系数 |
| 精度 | FP32 forward，autocast 关闭，GradScaler 开启，TF32 关闭 |
| 随机种子 | 数据初始 shuffle=1234；epoch sampler=1234+epoch；模型/文本 RNG=1234+rank |
| DataLoader | workers=0，drop_last=True；官方连续 rank 分片 |

官方配置虽然标注 float16，当前参考 trainer 没有在 forward 使用 autocast；此处按其实际执行路径实现 FP32 + GradScaler。GradNormTracker 不等于固定 global clip 0.5 或 `clip_grad_norm_(..., 1.0)`。

每轮 **40,979** 次更新、丢弃 **6** 行；3 轮共 **122,937** 次更新，warmup **6,147** 次。smoke 仍使用这个完整训练 horizon，仅在第 8 次真实 optimizer update 后结束。LambdaLR 初始比例为 0.1，因此第一次更新实际 LR 为 LM 3e-5、mapper 6e-5。

官方默认 4×8 与这里的 8×4 具有相同 GBS32，但 rank 分片、随机状态和 rank 内 token 均值随并行布局变化；本路线不承诺相同参数轨迹。其他明确差异为 HTSAT 架构、在线解码，以及过滤后的数据混合与原始 LibriTTS 音质。

## 远程执行

将本路线、外层脚本和现有 `code/RSmol/scripts/audit_mizar_s1_raw_audio.py` 经 GitHub 同步到服务器后，从远程仓库的 `code/RSmol/` 目录执行。默认复用 `mellow_c8204d8` conda 环境与既有 5090 容器。预训练目录默认：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2
/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt
```

如实际目录不同，用 `--text-model-dir` 和 `--htsat-root` 指定。manifest 路径固定，不提供替换入口。

直接提交 3 epochs 正式训练；默认模式为 formal，无需 smoke 报告，也不自动执行 smoke：

```bash
bash run_mizar_s1_5090.sh --mode formal \
  --stage-start-steps 512 --stage-workers 16 \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_5090/formal_async_v1
```

提交入口、节点启动脚本和配置生成程序均已移除 smoke PASS 硬门槛及 `--smoke-report` 参数；不读取或比较历史 smoke 的代码/数据/资产身份。仍记录本次配置、权重 SHA256 和软件版本，核对数据映射、tokenizer、参数组、精度、HTSAT 冻结和 staging 就绪状态。预拷贝步数/复制线程数只控制 I/O，不进入训练语义合同。

如需手动诊断，仍可显式用 `--mode smoke` 单独执行 8 次真实更新，但该模式不参与 formal 的启动流程。

只有本路线同一完整配方的 formal checkpoint 可以用 `--mode formal --resume-checkpoint ...` 恢复。保存 optimizer、scheduler、scaler、GradNorm、每个 rank RNG 和 epoch/batch 游标；smoke checkpoint 不接受用于 formal resume。

## 审计产物与验收

每种模式在 `<output-root>/smoke/` 或 `<output-root>/formal/` 写入：

- `staging_report.json`：数据身份、映射、启动档位预算、计划/实际复制数、容量与耗时。预拷贝完成为 `TRAINING_READY`，全量完成后原子更新为 `PASS`。
- `staging_progress.json`：后台 phase、已复制文件数/字节、连续就绪 batch、吞吐和剩余 cgroup/tmpfs 容量。节点本地另有 `STAGING.json` 供训练读取；异常阶段为 `FAILED`。
- `runtime.json`：完整配置、资产 SHA256、软件版本、训练更新数、已知差异。
- `config_audit.json`、`trainable_params.txt`：实际参数组、LR、PAD/EOS、精度、冻结状态、每来源 GPU 前向探针。
- `metrics.jsonl`：实际更新数、loss、使用的/下一步 LR、梯度范数和缩放、显存、采样行，以及累计 staging 等待时间/复制是否全部完成。
- `training_report.json`：最终成功报告；失败则尽可能写入 `failure_rank<N>.json` 并使任务失败。
- `checkpoints/`：完整状态 checkpoint。

上一版 8 项小型张量测试已由用户在远程通过（2026-10-21，PyTorch 2.11.0+cu128 / CUDA 12.8，8 张卡可见），按用户要求注释掉 `run.sh` 中的自动执行入口。文件仍保留，如需手动诊断，可在本路线目录执行 `python runtime_checks.py`。本次修正其中的 PAD/EOS 用例，加入 PAD=17 的忽略检查与 EOS=0 的监督检查；修正后的张量用例尚未远程执行。这 8 项单元检查与下面的 8 次真实训练更新是两个步骤。

远程 `smoke_v2` 曾被错误的 `pad_id == 0` 预检阻断，修正 PAD=17 / EOS=SEP=0 后，用户确认 `smoke_v3` 的 8 次真实 optimizer updates 已 PASS。随后 formal 在全量 staging 到约 215,040/333,510 文件时无 traceback 中止；GPU 空闲回收是用户基于平台经验的判断，尚无平台退出原因证明。当前按用户要求采用后台 staging 并直接 formal，不要求重跑 smoke。

预拷贝就绪并启动 trainer 后，每来源 1 行通过训练数据加载器在线解码并经过真实 HTSAT→mapper→LM→CE。仅当显式选择 smoke 时，每卡实际训练 8 批，共 256 次样本使用，并检查第 1/8 步三个可训练组件的非零有限梯度、权重变化及 HTSAT 完全不变。正式训练直接按完整 3 epochs 配方更新。

用户日志中 8 项测试的主体耗时为 0.034 秒，跳过它们对长时间 GPU 空闲的改善有限。索引完整 manifest 和复制原始音频仍需要准备时间；日志与 `staging_report.json.phase_seconds` 分别记录加载映射、建立索引/选择文件及复制耗时。

smoke checkpoint 为 `checkpoints/model--step-8.ckpt`，保存后重新读取并核对模型张量及完整训练状态。formal 在 20,490 / 40,979 / 61,468 / 81,958 / 102,448 / 122,937 步保存，最终文件为 `formal/checkpoints/model--epo-3.ckpt`。

这不是全量音频解码审计；未抽到的坏文件仍可能在正式训练中暴露。训练期间解码失败会带原始行、来源和路径中止，不会默默跳过。

本地验证记录：本次移除 smoke 门槛后，路线标准库测试 13 项、后台 staging 并发/故障测试 9 项通过；已有过滤/映射测试 11 项在此前通过。当前测试验证无 smoke 报告的完整 formal 配置生成、八卡连续分片与 resume 索引、跨预拷贝边界等待、后续 epoch 等待全量、原子复制、后台失败/进程消失/心跳超时、PAD/EOS 合同，以及实际 Bash formal 入口在正常/配置失败时的并行执行和进程清理模拟。Python 与两份 Bash 启动脚本语法检查通过。此前官方函数对照覆盖 1,344 次文本与随机状态比较、8 次 tokenizer 参数比较、122,938 个 LR 位置、10,000 次 GradNorm 标量比较。本地没有 PyTorch、数据、预训练权重或 CUDA；后台 staging 版本的真实 8 卡训练尚待远程报告。现有统一 evaluator 尚未适配此路线的新序列合同与 checkpoint。

## 对已有 MMAU 生成结果按选项标签重评分

同步代码后，可在登录节点直接执行以下 CPU 命令，不加载模型、音频或 checkpoint，也不重新生成：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
python scripts/rescore_mmau_mellow_author_reply.py \
  /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_5090/formal_async_v3/eval/mmau_test_mini_step122937_retry2/raw_generations.jsonl \
  --expected-rows 1000
```

直接复用旧的标签评分规则：`generated_text.split(')')[0].lower()` 与正确选项标签比较；不额外 trim、去 EOS 或提取答案正文。`B) anything` 对正确标签 b 判对，` b) ...`、`answer is b) ...` 不会被纠正；若没有 `)`，比较整个输出的小写形式。正确标签来自每条记录的 `official_record.answer` 与原始顺序的 `official_record.choices`，不是重新排序选项。

JSONL 输入默认要求完整 1,000 条 generated 记录、唯一 ID 与连续 row_index，损坏行、重复或缺行会报错；答案无法精确对应选项则保留分母并记录为错误。只读取原始 `generated_text`，不使用可能经过处理的 `model_output`。输出到输入文件同级的 `choice_label_rescore/`，可用 `--output-dir` 指定其他目录：

日志顶层的 `question`/`choices` 按推理时相同的提取函数与 `official_record` 校验，允许生成阶段已做的两端空白清理，并在 `input_audit.metadata_normalization_counts` 记录条数；实际题目或选项不一致仍报错。该校验不清理模型输出，`generated_text` 的前导空白仍按原评分规则保留。

- `mellow_author_reply_evaluation.txt`：总分、sound/music/speech 与难度分组分数。
- `mellow_author_reply_evaluation.json`：上述分数及逐题完整输出、正确标签、判分与元数据错误。
- `mellow_author_reply_rescore_report.json`：输入文件 SHA256、完整性检查、评分规则与摘要。

这是同一份生成结果上的额外评分规则，不是 MMAU 官方 scorer 的分数，也不改变本次 Mizar S1 的推理协议。原始 JSONL 与既有官方评分文件保持不变。脚本仅使用 Python 标准库；本地验证包含 1,000 条模拟记录的 `python -S` 命令行测试，不代表已读取远程真实输出或已获得模型分数。

## 许可证

模型基础保留 Mellow MIT `LICENSE`。移植的 Mizar 部分保留 `licenses/Mizar-ClearBSD.txt`，归属说明见 `NOTICE`。
