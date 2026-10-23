# HTSAT / SmolLM2：Mizar Stage 1，GBS256 消融，3 epochs

这是一条从 GBS32 Mizar/HTSAT 路线隔离复制的 batch-size 消融，沿用 `../mizar_stage1_fixed20_3epochs/` 架构及训练合同，按本地官方 Mizar 源码移植 Stage 1 训练语义。GBS32 原路线、ReasonAQA 训练目录、启动脚本和 checkpoint 合同保持原样。此路线固定为 **8 GPUs × microbatch 8 × gradient accumulation 4 = global batch 256**，仅正式训练，不配置 smoke 模式或 smoke 门禁。

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

原始音频按实际文件去重并复制到独占的 `/dev/shm/mizar_s1_gbs256_<job-id>`。staging 只做路径映射、容量/文件大小检查和原始字节复制，不解码、重采样、裁剪、补零或生成音频特征。训练在线执行：

1. soundfile 解码为 float32，所有声道取均值，重采样至 32 kHz。
2. 单个原始文件先中心裁剪或右侧补零到 20 秒，再分为连续的两个 10 秒。
3. 真正双文件各自中心裁剪或右补零到 10 秒，再按顺序拼接。只有显式 `filepath2_original` / `audio_path2` 才表示第二个原始文件，并且同样要求成功审计。
4. 两个 10 秒分别进入现有 HTSAT。历史 `filepath1` / `filepath2` 是旧的处理后路径，不能据此判断原始双文件。

正式训练复制过滤后全部保留行引用的原始音频。同一文件被多行引用只复制一次，不会整目录复制数据集，也不会复制已排除的 MusicCaps / train-other-500 行专属音频。

当前默认使用后台 staging，启动流程为：

1. 用私有 RNG 重建 `data_order(seed=1234)`、本轮 `torch.randperm(seed=1234+epoch)` 和八个连续 rank 分片；每个 optimizer update 包含每 rank 连续 32 行（4 个 microbatch）。按每个文件在实际训练中最早使用的 optimizer update 排序。resume 使用 checkpoint 的 epoch/update 游标。
2. 来源探针文件优先，预先复制前 **64 个 optimizer updates** 的去重音频（八卡合计 16,384 次样本使用），与 GBS32 路线预拷贝样本量相同。用 `--stage-start-steps N` 调整；`STAGING_PLAN` 会记录 16/32/64/128/256 等档位的实际文件数/字节数。
3. 发布 `TRAINING_READY` 后立即启动 trainer；同一个后台 CPU 进程继续按顺序复制剩余文件，每 16 个 optimizer updates 发布八卡共同的就绪进度。默认 16 个复制线程，可用 `--stage-workers N` 调整。
4. 每个文件先复制到 `.partial`，检查大小后原子重命名。训练在加载下一批前检查就绪状态，追上复制时明确等待，后台失败、进程消失、心跳超时或等待超过 600 秒则报错。索引和文件路径不随复制进度改写。
5. 全量复制完成后发布 `COMPLETE` 和最终 `PASS` 报告，此后不再逐批轮询。已复制文件保留给后续 epochs；开始下一 epoch 前要求全部保留文件完成，包括本轮被 `drop_last` 丢弃的行专属文件。

64-step 预拷贝和 16-step 后台发布窗口按有效样本量缩放；本路线不提供 smoke 模式。后台 CPU 复制可能在模型加载期间继续推进。

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
| Epochs / 并行 | 3；8 × microbatch 8 × accumulation 4，GBS256 |
| Optimizer | AdamW，betas=(0.9, 0.999)，eps=1e-8 |
| 基础学习率 | 完整 LM 3e-4；c2l + projection 6e-4 |
| Weight decay | 0.01；bias 和 ndim≤1 参数为 0 |
| Schedule | 官方 LambdaLR，5% warmup + cosine，最低比例 0.1；optimizer 后调用 scheduler |
| 梯度处理 | 官方 GradNormTracker，L2/max 初值 1/10，overdrive 2.5，momentum 0.995，共用最小缩放系数 |
| 精度 | FP32 forward，autocast 关闭，GradScaler 开启，TF32 关闭 |
| 随机种子 | 数据初始 shuffle=1234；epoch sampler=1234+epoch；模型/文本 RNG=1234+rank |
| DataLoader | workers=0，drop_last=True；官方连续 rank 分片 |

官方配置虽然标注 float16，当前参考 trainer 没有在 forward 使用 autocast；此处按其实际执行路径实现 FP32 + GradScaler。GradNormTracker 不等于固定 global clip 0.5 或 `clip_grad_norm_(..., 1.0)`。

每轮 **5,122** 次 optimizer updates，丢弃 **102** 行；3 轮共 **15,366** 次更新，warmup **768** 次。每步累计 4 个 microbatch 的 rank-local token mean，每个 loss 除以 4 后反向传播；前三个 DDP microbatch 使用 `no_sync()`，最后一个同步。累积结束后仅执行一次 unscale、GradNormTracker、optimizer 和 scheduler 更新。不同 microbatch 的有效 token 数不等时，这等于四个 microbatch 均权平均，而不是将 32 个样本的有效 token 合并后一次求均值。LambdaLR 初始比例为 0.1，因此第一次更新实际 LR 为 LM 3e-5、mapper 6e-5。

GBS256 是相对于 Mizar Stage 1 GBS32 的 batch-size 消融，不是官方 batch 配置。沿用相同峰值学习率以单独考察 batch size；每 epoch 的样本曝光和 5% warmup 曝光量与 GBS32 接近，但 optimizer 更新次数减少约 8 倍，因此训练轨迹不会相同。单次 microbatch 从 4 增至 8 会提高每 GPU 激活显存；梯度累积逐个执行 microbatch，不会同时保留四份激活。能否在目标 GPU 上容纳 microbatch 8 仍须以正式作业的峰值显存和日志验证。其他明确差异为 HTSAT 架构、在线解码、过滤后的数据混合与原始 LibriTTS 音质。

## 远程执行

将本路线、外层脚本和现有 `code/RSmol/scripts/audit_mizar_s1_raw_audio.py` 经 GitHub 同步到服务器后，从远程仓库的 `code/RSmol/` 目录执行。默认复用 `mellow_c8204d8` conda 环境与既有 5090 容器。预训练目录默认：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2
/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt
```

如实际目录不同，用 `--text-model-dir` 和 `--htsat-root` 指定。manifest 路径固定，不提供替换入口。

直接提交本路线的 3 epochs formal 训练。使用独立入口：

```bash
bash run_mizar_s1_gbs256_5090.sh \
  --stage-start-steps 64 --stage-workers 16 \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_5090/formal_async_v1
```

本路线入口仅提交 formal，不接受 smoke 参数，也不读取历史 GBS32 smoke 或 checkpoint。仍记录本次配置、权重 SHA256 和软件版本，核对数据映射、tokenizer、参数组、精度、HTSAT 冻结和 staging 就绪状态。预拷贝步数/复制线程数只控制 I/O，不进入训练语义合同。

恢复时使用 `--resume-checkpoint ...`，仅接受本路线、同一代码/数据/配方的完整 formal checkpoint。checkpoint 在 optimizer update 边界保存，并记录 optimizer 游标、optimizer、scheduler、scaler、GradNorm 与每个 rank RNG；GBS32 checkpoint 因 route identity 不同而被拒绝。

## 审计产物与验收

训练产物写入 `<output-root>/formal/`：

- `staging_report.json`：数据身份、映射、启动档位预算、计划/实际复制数、容量与耗时。预拷贝完成为 `TRAINING_READY`，全量完成后原子更新为 `PASS`。
- `staging_progress.json`：后台 phase、已复制文件数/字节、连续就绪 batch、吞吐和剩余 cgroup/tmpfs 容量。节点本地另有 `STAGING.json` 供训练读取；异常阶段为 `FAILED`。
- `runtime.json`：完整配置、资产 SHA256、软件版本、训练更新数、已知差异。
- `config_audit.json`、`trainable_params.txt`：实际参数组、LR、PAD/EOS、精度、冻结状态、每来源 GPU 前向探针。
- `metrics.jsonl`：实际更新数、loss、使用的/下一步 LR、梯度范数和缩放、显存、采样行，以及累计 staging 等待时间/复制是否全部完成。
- `training_report.json`：最终成功报告；失败则尽可能写入 `failure_rank<N>.json` 并使任务失败。
- `checkpoints/`：完整状态 checkpoint。

本路线保留 `runtime_checks.py` 作为需要时手动执行的远程环境诊断工具；本次不运行 GPU 检查。CPU/stdlib 合同测试覆盖累积配置、八卡采样优先级、staging 边界、formal 参数生成和提交入口。

GBS32 原路线曾因 PAD 预检和全量 staging 启动时间遇到问题；这些运行历史不作为本 GBS256 路线的验收证据。新路线复用已经通过远程 smoke 的模型/数据处理基础，但 microbatch 8、四步累积和新的异步 staging 游标尚未进行真实 GPU 验证。

预拷贝就绪并启动 trainer 后，每来源 1 行通过训练数据加载器在线解码并经过真实 HTSAT→mapper→LM→CE。正式训练在全部 4 个 microbatch 完成后才更新梯度裁剪器、optimizer 和 scheduler。

用户日志中 8 项测试的主体耗时为 0.034 秒，跳过它们对长时间 GPU 空闲的改善有限。索引完整 manifest 和复制原始音频仍需要准备时间；日志与 `staging_report.json.phase_seconds` 分别记录加载映射、建立索引/选择文件及复制耗时。

formal 在 2,561 / 5,122 / 7,683 / 10,244 / 12,805 / 15,366 步保存，最终文件为 `formal/checkpoints/model--epo-3.ckpt`。

这不是全量音频解码审计；未抽到的坏文件仍可能在正式训练中暴露。训练期间解码失败会带原始行、来源和路径中止，不会默默跳过。

本地验证记录：本路线 CPU/stdlib 测试覆盖 GBS256 epoch 几何、八卡连续 rank 分片和累积组 staging 优先级、resume update 游标、64/16-step staging 窗口、loss/4 累积及 DDP no_sync 结构、formal-only 提交入口。Python 语法、两份 Bash 启动脚本语法和 Git whitespace 检查通过。本地没有 PyTorch、数据、预训练权重或 CUDA；microbatch 8 的显存需求和真实 8 卡 DDP/optimizer 更新尚待远程 formal 作业验证。统一 evaluator 尚未适配此路线的新序列合同与 checkpoint。

## 许可证

模型基础保留 Mellow MIT `LICENSE`。移植的 Mizar 部分保留 `licenses/Mizar-ClearBSD.txt`，归属说明见 `NOTICE`。
