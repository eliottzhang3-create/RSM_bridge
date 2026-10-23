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

提交器先在登录节点调用纯标准库的 `prepare_index.py`，按审计 TSV 的 `(field, raw_path)` 查找成功映射。ReasonAQA 使用 `filepath1_original`，其他来源使用 `audio_path`；过滤后的行号不用于匹配原审计。映射失败、原文件缺失、空文件或来源计数变化都会中止，并且不会申请 GPU。

原 manifest 的字段和值保持不变，额外生成带 `s1_raw_paths`、`s1_original_paths` 的 JSONL/offset 索引、去重文件列表和 `index_report.json`。索引保存在共享存储 `index_cache/<signature>/`；签名包含路线代码 SHA256、manifest/审计 TSV 的大小与内容 SHA256，输入绝对路径只作溯源记录。时间戳变化或内容相同的路径别名不使索引失效。首次准备统计来源数、原始行 canonical SHA256、映射 SHA256 和每个原始音频的大小/mtime；后续同代码/输入的提交直接复用，不扫描全部音频。索引经临时目录原子发布，缺失/不完整产物报错。输入文件应保持只读不变；每次复用索引只重新计算 manifest/审计 TSV 两个输入文件的 SHA256，不重扫 333,510 个音频。

2026-10-23 远程作业曾在加载 CPU 索引时报告 `manifest/audit TSV changed since CPU index preparation`。旧实现将路径、大小、`mtime_ns` 整体比较，错误没有指出差异，因此现有日志不能确认究竟是哪一项变化。现已采用 schema 2 的内容指纹；若内容变化，报错会列出具体输入及预期/当前路径、大小、SHA256。旧索引不会直接复用，更新代码后正常提交会在申请 GPU 前自动建立新的缓存项，无需手动删除旧缓存。GPU 节点启动只顺序读取这两个 metadata 输入验证 SHA256，不重新扫描音频。

原始音频按实际文件去重并复制到独占的 `/dev/shm/mizar_s1_gbs256_<job-id>`。staging 只做路径映射、容量/文件大小检查和原始字节复制，不解码、重采样、裁剪、补零或生成音频特征。训练在线执行：

1. soundfile 解码为 float32，所有声道取均值，重采样至 32 kHz。
2. 单个原始文件先中心裁剪或右侧补零到 20 秒，再分为连续的两个 10 秒。
3. 真正双文件各自中心裁剪或右补零到 10 秒，再按顺序拼接。只有显式 `filepath2_original` / `audio_path2` 才表示第二个原始文件，并且同样要求成功审计。
4. 两个 10 秒分别进入现有 HTSAT。历史 `filepath1` / `filepath2` 是旧的处理后路径，不能据此判断原始双文件。

正式训练复制过滤后全部保留行引用的原始音频。同一文件被多行引用只复制一次，不会整目录复制数据集，也不会复制已排除的 MusicCaps / train-other-500 行专属音频。

当前使用“共享存储开始训练 → 后台全量复制 → 八卡一起切换”的流程：

1. `run_mizar_s1_gbs256_5090.sh` 在申请 GPU 前准备或复用上述 CPU 索引。首次索引扫描仍有耗时，但此时尚未提交 GPU 作业；无需提前激活 PyTorch 环境，登录节点 Python 3.10+ 即可。
2. 计算节点加载已准备的 metadata/文件列表，检查 tmpfs/cgroup 容量后发布 `TRAINING_READY`，此时复制文件数可以为 **0**。不再重建全量 metadata，不设预拷贝门槛。模型/权重加载和来源前向探针仍需正常启动时间。
3. trainer 从计算节点直接读取 `/hpc_stor03/...` 原始共享路径，在线解码并执行真实参数更新；后台默认 **4 个 CPU 复制线程**复制全部实际引用的原始音频。`--stage-workers N` 可调整并发。没有从登录节点运行训练或代理传输音频；训练与复制暂时竞争同一共享存储带宽。
4. 每个文件复制前后核对大小/mtime，先写 `.partial`，大小匹配后原子重命名；全部文件完成才发布 `COMPLETE`。复制错误、进程退出或心跳超过 120 秒均明确报错。GPU 训练每个 optimizer update 检查一次复制健康，不等待文件水位，跨 epoch 也可以继续读共享路径。
5. 完整 optimizer update 边界、加载下一个 microbatch 之前，各 rank 对全量完成标志做 MIN all-reduce。只有八卡都确认完成，才一起把整个数据读取模式切为本地。不会按文件混用路径，也不会在四个 microbatch 中间切换。日志发布 `AUDIO_READ_SWITCH`，记录已完成的 optimizer steps；采样顺序、模型、optimizer、scheduler、GradNorm/RNG 和在线 fixed20 处理不重置。
6. 已复制音频保留到作业结束；若训练先结束，最终成功报告仍要求后台复制完成。退出先终止/回收复制进程，再清理本作业独占的 tmpfs。共享 CPU 索引保留供后续提交复用。

后台只复制原始字节，不解码或占用 GPU，没有音频归档或额外完整音频副本。最终音频仍约 333,510 个文件 / 173.48 GiB；CPU 索引额外占少量共享磁盘，训练通过 mmap 读取。并行复制与训练共享申请的 32 CPU / 256 GiB 主存。这个方案移除了 GPU 作业中的全量映射扫描和预拷贝等待，不能保证共享存储读取速度或平台不再终止作业，之前三次终止的确切原因仍需平台日志确认。

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
  --stage-workers 4 \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_5090/formal_sharedstart_v2
```

本路线入口仅提交 formal，不接受 smoke 参数，也不读取历史 GBS32 smoke 或 checkpoint。仍记录本次配置、权重 SHA256 和软件版本，核对数据映射、tokenizer、参数组、精度、HTSAT 冻结和 staging 身份。`--stage-start-steps` 已移除；`--index-cache-root DIR` 可指定共享索引缓存根目录，`--stage-workers N` 仅控制 I/O 并发。默认缓存位于 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_5090/index_cache/`。

恢复时使用 `--resume-checkpoint ...`，仅接受本路线、同一代码/数据/配方的完整 formal checkpoint。checkpoint 在 optimizer update 边界保存，并记录 optimizer 游标、optimizer、scheduler、scaler、GradNorm 与每个 rank RNG。新分配节点从共享路径恢复训练并重新复制音频，完成后再切换；路径切换不进入采样游标。GBS32 checkpoint 被拒绝；本次改代码前 GBS256 checkpoint 也因源码 SHA 改变而被拒绝，失败且没有 checkpoint 的作业从头重提。

## 审计产物与验收

训练产物写入 `<output-root>/formal/`：

- `staging_report.json`：CPU 索引身份、映射、计划/实际复制数、容量与耗时。无需音频预拷贝即为 `TRAINING_READY`，全量完成后原子更新为 `PASS`。
- `staging_progress.json`：后台 phase、已复制文件数/字节、吞吐和剩余 cgroup/tmpfs 容量。节点本地另有 `STAGING.json` 供训练读取；异常阶段为 `FAILED`。旧水位字段保留供状态兼容，本路线不依赖水位等待。
- `runtime.json`：完整配置、资产 SHA256、软件版本、训练更新数、已知差异。
- `config_audit.json`、`trainable_params.txt`：实际参数组、LR、PAD/EOS、精度、冻结状态、每来源 GPU 前向探针。
- `metrics.jsonl`：实际更新数、loss、使用的/下一步 LR、梯度范数和缩放、显存、采样行，以及 `audio_read_mode`、原始/本地文件读取计数、`local_switch_step` 与复制完成标记。
- `training_report.json`：最终成功报告；失败则尽可能写入 `failure_rank<N>.json` 并使任务失败。
- `checkpoints/`：完整状态 checkpoint。

本路线保留 `runtime_checks.py` 作为需要时手动执行的远程环境诊断工具；本次不运行 GPU 检查。CPU/stdlib 合同测试覆盖累积配置、采样优先级、零预拷贝启动、原始/本地路径选择、后台健康检查、formal 参数生成和提交入口。

用户反馈 GBS256 路线连续三次未能启动训练，怀疑 staging 等待过长。本次只修改 GBS256 路线的数据访问/复制启动流程，GBS32 目录不改。新的共享读取与八卡同步切换尚待远程正式作业验证。

trainer 启动后，每来源 1 行通过训练数据加载器在线解码并经过真实 HTSAT→mapper→LM→CE；此时允许直接使用共享音频。正式训练在全部 4 个 microbatch 完成后才更新梯度裁剪器、optimizer 和 scheduler。

首次索引耗时记录在共享 `index_report.json.preparation_seconds`；GPU 节点加载索引和音频复制耗时写入 `staging_report.json.phase_seconds`，不会把 CPU 索引准备时间记为 GPU 训练时间。

formal 在 2,561 / 5,122 / 7,683 / 10,244 / 12,805 / 15,366 步保存，最终文件为 `formal/checkpoints/model--epo-3.ckpt`。

这不是全量音频解码审计；未抽到的坏文件仍可能在正式训练中暴露。训练期间解码失败会带原始行、来源和路径中止，不会默默跳过。

本地验证覆盖：CPU 索引原子发布/复用、仅时间戳变化和内容相同的路径别名复用、保留大小/时间戳的内容变化拒绝、输入变化的详细诊断、旧身份格式拒绝、首个文件仍在复制时启动、共享读取跨 epoch、不读取 partial 文件、显式切换后仅用本地路径、复制失败/进程死亡/心跳过期、全量计数校验、原始文件大小/mtime 改变、零已复制文件的 formal 配置，以及 optimizer 边界 collective 切换的代码结构。Python/Bash 语法与 Git whitespace 检查通过。本地没有 PyTorch、真实数据、权重或 CUDA，不能将这些 CPU 检查作为真实八卡同步切换或 microbatch 8 显存适配通过。统一 evaluator 尚未适配本路线。

## 许可证

模型基础保留 Mellow MIT `LICENSE`。移植的 Mizar 部分保留 `licenses/Mizar-ClearBSD.txt`，归属说明见 `NOTICE`。
