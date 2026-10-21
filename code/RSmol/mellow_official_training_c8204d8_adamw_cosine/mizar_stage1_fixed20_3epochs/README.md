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

原始音频按实际文件去重并复制到独占的 `/dev/shm/mizar_s1_<job-id>`，文件内容不重编码。训练在线执行：

1. soundfile 解码为 float32，所有声道取均值，重采样至 32 kHz。
2. 单个原始文件先中心裁剪或右侧补零到 20 秒，再分为连续的两个 10 秒。
3. 真正双文件各自中心裁剪或右补零到 10 秒，再按顺序拼接。只有显式 `filepath2_original` / `audio_path2` 才表示第二个原始文件，并且同样要求成功审计。
4. 两个 10 秒分别进入现有 HTSAT。历史 `filepath1` / `filepath2` 是旧的处理后路径，不能据此判断原始双文件。

smoke 只复制各 rank 实际前 8 批需要的音频，以及每个来源的 3 个解码探针；正式训练复制全部保留行所需的原始音频。两种模式都读取完整 manifest 以保持相同采样顺序和数据身份。

复制前检查 tmpfs 和可用 cgroup 内存，额外保留 32 GiB；不足则在训练前明确退出。默认任务申请 256 GiB 主存，是否容得下全部原始音频以 `staging_report.json` 的实际字节数为准。清理只允许带本任务所有权标记的直接子目录，退出时自动执行。

## 序列与可训练参数

```text
HTSAT(10s)[129] + HTSAT(10s)[129] + SEP[1] + prompt[256] + answer[300]
                        audio 258          prefix 515       total 815
```

保留两段各自的 clip token 和原有平均池化，删除两段音频之间的 SEP，只在完整音频 prefix 后保留一个 SEP。该长度来自现有 HTSAT 架构，与官方 CED 的 126 个音频 token 不同。

HTSAT 主干固定 `eval()`、`no_grad()`；原有 c2l（527→768）、残差 projection（768→576，dropout 0.5）和完整 SmolLM2 参与训练。没有新增循环模块或 LoRA。

prompt 和 answer 各自在分词前追加 `" <|endoftext|>"`，然后分别截断/补齐到 256 / 300；若 EOS 被截断，不强制插回。PAD 为已有词表的 `!`（ID 0），SEP 同为 ID 0。LM 接收 `attention_mask=None`、`labels=None`、`use_cache=False`，固定 padding 位置参与因果注意力。

外部 next-token CE 只预测 answer，忽略 ID 0 的目标；prompt / audio 不计 loss。先求每个 rank 有效 answer token 的均值，再由 DDP 平均梯度，与旧路线的全局 token 加权均值不同。

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

先提交独立 8 步 smoke：

```bash
bash run_mizar_s1_5090.sh --mode smoke \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_5090/smoke_v1
```

成功报告位于 `smoke_v1/smoke/training_report.json`。确认后，在新目录提交 3 epochs：

```bash
bash run_mizar_s1_5090.sh --mode formal \
  --smoke-report /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_5090/smoke_v1/smoke/training_report.json \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_5090/formal_v1
```

也可以提交一个任务，先通过 smoke 门禁，再自动从原始预训练权重开始 3 epochs：

```bash
bash run_mizar_s1_5090.sh --mode smoke-then-formal \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_5090/first_s1_run
```

正式训练不接着 smoke 权重继续更新。门禁核对全部 8 ranks 的 8 次更新、梯度、权重变化、HTSAT 冻结、序列长度、完整 checkpoint，以及同一代码、数据、预训练权重与软件版本。代码或配置发生变化后，需要重跑 smoke。

只有本路线同一完整配方的 formal checkpoint 可以用 `--mode formal --smoke-report ... --resume-checkpoint ...` 恢复。保存 optimizer、scheduler、scaler、GradNorm、每个 rank RNG 和 epoch/batch 游标；smoke checkpoint 不接受用于 formal resume。

## 审计产物与验收

每种模式在 `<output-root>/smoke/` 或 `<output-root>/formal/` 写入：

- `staging_report.json`：数据身份、映射、文件数/大小、容量、各来源原始/复制后波形一致性。
- `runtime.json`：完整配置、资产 SHA256、软件版本、训练更新数、已知差异。
- `config_audit.json`、`trainable_params.txt`：实际参数组、LR、PAD/EOS、精度、冻结状态、每来源 GPU 前向探针。
- `metrics.jsonl`：实际更新数、loss、使用的/下一步 LR、梯度范数和缩放、显存、采样行。
- `training_report.json`：最终成功报告；失败则尽可能写入 `failure_rank<N>.json` 并使任务失败。
- `checkpoints/`：完整状态 checkpoint。

启动时先在现有 GPU 环境运行 8 项小型张量测试，检查 fixed20、解码/重采样、采样/恢复、文本、序列和 loss shift、GradNorm、优化器及下一步恢复一致性。随后每来源解码 3 行，并让每来源 1 行经过真实 HTSAT→mapper→LM→CE。smoke 再让每卡实际训练 8 批，共 256 次样本使用，并检查第 1/8 步三个可训练组件的非零有限梯度、权重变化及 HTSAT 完全不变。

smoke checkpoint 为 `checkpoints/model--step-8.ckpt`，保存后重新读取并核对模型张量及完整训练状态。formal 在 20,490 / 40,979 / 61,468 / 81,958 / 102,448 / 122,937 步保存，最终文件为 `formal/checkpoints/model--epo-3.ckpt`。

这不是全量音频解码审计；未抽到的坏文件仍可能在正式训练中暴露。训练期间解码失败会带原始行、来源和路径中止，不会默默跳过。

本地验证记录：新路线标准库测试 11 项、已有过滤/映射测试 11 项通过；Python 语法及两份 Bash 启动脚本语法检查通过。官方函数对照覆盖 1,344 次文本与随机状态比较、8 次 tokenizer 参数比较、122,938 个 LR 位置、10,000 次 GradNorm 标量比较。本地没有 PyTorch、数据、预训练权重或 CUDA，真实张量测试、8 卡 smoke 和正式训练必须以远程报告为准。现有统一 evaluator 尚未适配此路线的新序列合同与 checkpoint。

## 许可证

模型基础保留 Mellow MIT `LICENSE`。移植的 Mizar 部分保留 `licenses/Mizar-ClearBSD.txt`，归属说明见 `NOTICE`。
