# RSM_bridge：Recursive SmolLM / Audio MeSH 项目交接

> 最后同步：2026-09-09
> 本文件是后续 Codex 会话的首要交接依据。任何新会话必须先完整阅读本文件，再查看对应代码、测试和远程日志。若 README、口头历史与当前代码冲突，以当前代码行为和最新远程报告为准，并把差异补回本文。

## 0. 一页结论：现在做到哪里

项目最初在 SmolLM2-135M 上验证文本循环模型，现在已经进入音频模态阶段。当前唯一主线是：

```text
ReasonAQA 两段音频
  -> 冻结的 HTSAT AudioSet 编码器
  -> Mellow c2l + 严格按 Mellow 实现的投影/8 倍下采样 mapper
  -> 两段音频共 260 个 prefix 位置
  -> 5-10x2-5 MeSH 文本模型
  -> 只对 answer token 计算 causal-LM loss
```

当前主线的文本初始化模型已切换为第二轮低学习率 MeSH checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244
```

当前代码中的正式音频训练合同是：8 卡、每卡 micro-batch 8、GA=4、effective batch=256、3 epochs、max LR=1e-3、min LR=0、实际总 optimizer steps 的前 5% warmup、cosine decay、每 1000 步保存、最多保留最近 4 个完整 checkpoint。

重要状态：

- Stage 1 最终 manifest 已远程通过，共 1,243,942 条；12 条指向同一个缺失音频的 train 记录被明确丢弃。
- Stage 2 HTSAT 单音频审计已远程 PASS。
- 简化后的 Stage 3 manifest 审计已远程 PASS。
- Stage 4 曾在旧 mapper 上 PASS；随后 mapper 被改成严格 Mellow 结构，当前代码增加了结构、token 数、answer-only label 和梯度硬审计，**仍应使用最新代码重新跑 Stage 4**。
- 旧 Stage 7 `stage7_unified_padding_20260908_2/checkpoint-000010` 属于严格 Mellow mapper 修正前的产物，**不能作为当前正式训练的 resume checkpoint**。
- 当前严格 Mellow mapper 版本尚没有在本文中登记新的 Stage 7 PASS 或正式训练结果。独立审查代理在上下文中断前也没有提交最终报告，因此不得写成“已审查通过”。

## 1. 项目工作方式与不可违反的边界

本项目是本地代码、远程资产分离的工作流：

- 本地 Windows 工作区只保存核心 Python、Shell、测试和文档。
- 模型权重、数据集、manifest 产物、日志、训练输出和 checkpoint 均在远程 Linux。
- GitHub 只中转代码。不得提交权重、音频、parquet、checkpoint 或大日志。
- 本地负责阅读、修改、`py_compile`、静态测试、diff 审查；涉及真实数据或 CUDA 的审计必须在远程运行。
- 只用 CPU 的 Stage 0/1/3 可在远程终端 `conda activate rsmol` 后直接运行；所有 GPU 步骤必须走仓库提供的 `vc submit` 包装器，不能在登录节点直接跑 CUDA Python/`torchrun`。
- 各路线必须完全隔离。修改音频 MeSH 路线时，不得顺手改变旧 `5-10-5`、Parcae、直接 Poisson recursion 或 15R 文件。
- 不能因为本地静态测试通过就声称远程 PASS。远程状态必须有 report、日志或真实 checkpoint 作为依据。

状态用语约定：

- **已远程 PASS**：用户提供过明确 PASS report/日志。
- **代码就绪，待远程验证**：本地实现和静态检查存在，但当前版本没有远程 PASS。
- **历史/对照**：保留用于复现和比较，不是默认主线。

## 2. 远程环境和路径

### 2.1 工作区与环境

```text
本地仓库:
C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge

远程仓库:
/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM

远程 conda 环境:
/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/rsmol

输出根目录:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol
```

最近日志中看到的核心环境是 Python 3.10、PyTorch 2.11.0+cu128、Transformers 4.54.1。日志里的 `TRANSFORMERS_CACHE` FutureWarning 不是失败原因；后续可迁移到 `HF_HOME`，但不要为了消除 warning 改变训练逻辑。

GPU 包装器当前使用：

```text
queue: pdgpu-5090
image: docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1
formal resources: -c 32 -m 256G -g 8 -n 1
```

### 2.2 文本数据与模型

```text
原始 SmolLM2:
/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2

文本 parquet 的正确目录（shard 在 data 子目录，不是父目录）:
/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset/data
```

### 2.3 音频数据、外部代码与权重

```text
ReasonAQA metadata:
/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/train.json
/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/val.json
/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test.json

AudioCaps audio:
/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2/train
/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2/val
/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2/test

Clotho v2.1（普通 Clotho 任务可用；不是 ClothoAQA 的首选来源）:
/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1/development
/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1/validation
/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1/evaluation

ClothoAQA 官方音频包（ClothoAQA 记录直接从这里解析）:
/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_audio/audio_files

Mellow 官方代码:
/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main

HTSAT 官方代码:
/hpc_stor03/sjtu_home/jinwei.zhang/code/HTS-Audio-Transformer-main

HTSAT AudioSet checkpoint:
/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt
```

## 3. 版本历史总览

| 路线 | 精确定义 | 状态/用途 | 重要产物 |
|---|---|---|---|
| 原始 SmolLM2-135M | Hugging Face 原模型 | 基线与转换来源 | `/models/SmolLM2` |
| 15R recursive | 15 个物理层被逻辑重复执行 | 最早验证循环、梯度、DDP、checkpoint 的历史主线 | 历史输出 `/outputs/RSmol/stage4` |
| 固定 5-10-5 | prefix 5 + middle 10 + suffix 5，旧两次循环实现 | 历史固定深度对照；曾作为最初音频方案的文本 backbone，现已被 MeSH 替代 | `stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244` |
| 5-10-5 linear | 非递归线性对照 | 历史/对照 | 代码和评测仍保留 |
| 5-10x7-5 | 中间层固定运行 7 次 | 历史实验 | 代码保留，不是当前主线 |
| 5-10xr-5 | 直接 recursion，支持 Poisson 深度 | 与 Parcae 隔离的历史动态深度对照 | 代码保留 |
| 5-10xpoisson-parcae | Parcae 注入递归，截断 Poisson 深度 | 历史完成路线；Stage 3 eval 脚本存在 | `stage4_5_10xpoisson_parcae_lr8e-4_mb2_ga64_20260903_172047/checkpoint-009244` |
| 5-10x2-5-mesh | 20 个物理层、30 个逻辑层、5 个 memory slots、固定两轮 MeSH | 当前文本 backbone | 第一轮和第二轮 checkpoint 均存在 |
| 音频 5-10-5 + Mellow preflight | ReasonAQA manifest 与 HTSAT 预审 | 仍负责 Stage 0/1/2；名称保留是历史原因 | 最终 manifest 和 Stage 2 PASS report |
| 音频 5-10x2-5-mesh + Mellow | HTSAT/Mellow 音频 prefix 接 MeSH LM | **当前主线** | 当前严格 mapper 版本待重新完成 Stage 4/7/FORMAL |

最重要的隔离规则：

- Parcae 的 `A_bar(h)+B_bar(u)`、特殊 `h0` 和 Poisson 深度不能复制进 MeSH。
- 直接 recursion 的 `h0=e` 也不能冒充 Parcae。
- 当前音频路线必须使用 `recursive_model_5_10x2_5_mesh.py`，不是旧 `recursive_model_5_10_5.py`。
- 文件名中保留的 `audio_5_10_5_mellow` 仅用于早期 manifest/HTSAT preflight，不代表正式音频训练仍使用 5-10-5 文本模型。

### 3.1 历史循环路线的必要语义

固定 `5-10-5` 使用前 5 层输出初始化 hidden，中间 10 个共享物理层按固定次数循环，再进入后 5 层。它建立了当前 prefix/shared-core/suffix 的层映射基础，但没有 MeSH memory/router。

直接动态 recursion `5-10xr-5` 的核心是：

```text
e = Prefix(x)
h0 = e
h_{t+1} = Middle(h_t)
y = Suffix(h_T)
```

Parcae 路线则是：

```text
e = Prefix(x)
u = PreludeNorm(e)
h0 = 每次 forward、每条序列生成的初始化状态
h_{t+1} = Middle(A_bar(h_t) + B_bar(u))
y = Suffix(h_T)
```

两条动态路线都曾使用 `Poisson(lambda=7)` 截断到 `T in {4..10}` 的 per-sequence 深度，单 rank 执行到本地 `Tmax`，已完成样本在剩余轮次 no-op；推理默认 T=7，参数梯度窗口为最后 4 次循环，同时保留早期 hidden-state autograd 路径。历史 Poisson 正式合同为 8 卡、micro 2、GA 64、9244 steps、LR `8e-4 -> 8e-5`。这些定义只用于复现旧实验，**不能带入固定两轮 MeSH 音频主线**。

## 4. 当前文本 backbone：5-10x2-5 MeSH

核心实现：`code/RSmol/recursive_model_5_10x2_5_mesh.py`。

### 4.1 结构合同

- 物理层 20：prefix 5 + shared middle 10 + suffix 5。
- 逻辑层 30：prefix 5 + middle 10 × 2 次 + suffix 5。
- 固定循环次数 2；本版本不是 Poisson 动态深度。
- memory 形状是 `[batch, 5, sequence_length, hidden_size]`，共有 5 个槽位。
- 有 3 个 write router 和 3 个 read router：pre-transition 一组，每个循环各一组，共 6 个独立 `nn.Linear` router 模块、12 个参数张量（weight/bias）。
- transition router 的 query 是 prefix output。
- router 输出对 slot 维做 softmax；router 参数属于模型参数，参加优化并随 `save_pretrained` 保存进 checkpoint。
- memory 只在当前 forward 内存在，不是跨样本、跨 batch 持久状态。
- 不启用 MeSH 官方实验中的 `sqrt(d)` embedding scale；`embedding_scale="disabled"`。
- 架构合同：`logical_30_physical_20_5_10x2_5_mesh`；音频复合模型合同在其后加 `_audio_mellow`。

### 4.2 router 初始化与训练期检查

router 由 `_init_router` 显式初始化：bias 清零，weight 按代码中的有限值策略初始化，尺度为 `sqrt(2 / (5 * in_features))`。初始化函数已兼容 Transformers 的 meta-device 构造；遇到 meta tensor 时跳过数据读取，避免过去的 `Tensor.item() cannot be called on meta tensors`。

训练中的 router collapse 检查现在只做 warning/diagnostic，不再使某一 rank 单独抛异常。过去在 step 520 观察到的 NCCL broadcast timeout，实际链路是某 rank 因 `write_1` collapse 硬失败提前退出，其他 rank 才在 collective 等待超时；不是已经证明的网络故障。当前策略仍记录各 rank router stats，但不会因此中止模型自行学习。

### 4.3 已有 checkpoint

第一轮/早期正式训练完成产物：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/
formal_resume_000500_nonfatal_router_20260907_115533/checkpoint-009244
```

第二轮从上面 checkpoint 权重初始化，使用 `max_lr=2e-4, min_lr=2e-5` 完成；**这是当前音频路线默认初始化模型**：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/
formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244
```

旧文本正式合同为 8 卡、micro 8、GA 16、9244 optimizer steps。第一轮学习率为 `8e-4 -> 8e-5`，第二轮为 `2e-4 -> 2e-5`。这是文本预训练历史，不能与当前音频训练的 `1e-3 -> 0` 混淆。

Stage 3 文本评测入口：

```text
code/RSmol/scripts/evaluate_stage3_5_10x2_5_mesh.py
code/RSmol/run_stage3_eval_5_10x2_5_mesh_5090.sh
```

早期该评测曾有 `recursive_model` 局部变量未赋值错误，当前代码已修复。

## 5. 当前音频模型：HTSAT + 严格 Mellow mapper + MeSH

核心实现：

```text
code/RSmol/audio_5_10x2_5_mesh_mellow/
  data.py
  model.py
  stage3.py
  README.md
```

### 5.1 音频预处理

- 每条样本最多有 `filepath1` 和 `filepath2` 两段音频。
- 输入转成 mono、32 kHz、固定 10 秒。
- 超过 10 秒直接裁剪前 10 秒；不足 10 秒右侧补零；采样率不同则重采样。
- 如果 `filepath2` 为空，不另采样音频，直接复用 `filepath1`，即一段音频占两个输入位置。
- collate 会通过 `audio2_reused_mask` 标记复用。若可复用，HTSAT 编码结果也复用，避免无意义的第二次编码；但最终序列里仍有两份 audio prefix。

### 5.2 HTSAT 与 c2l

- 使用远程 Mellow 的 `mellow.model.htsat.HTSATWrapper` 和 AudioSet checkpoint。
- HTSAT backbone 冻结且永久保持 eval mode，不更新其参数或 dropout/batch statistics。
- HTSAT/Mellow 输出包含一个 `[B,1,768]` 的 latent/global/CLS 等价特征，以及 `[B,1024,527]` 的逐帧 event-presence map（以已审计的当前实现形状为合同）。
- `c2l = Linear(527,768)` 不是 HTSAT AudioSet checkpoint 中的固有已训练输出层，而是 Mellow 用来把 527 类事件图映射到 768 latent 空间的 adapter；它随机初始化、可训练，并单独保存。
- 优先使用 Mellow wrapper 已经产生的 `latent + c2l(framewise)` embedding，避免对 c2l 漏执行或重复执行；兼容 fallback 仅在 wrapper 没暴露 embedding 时自行拼接。

### 5.3 严格 Mellow mapper

当前 mapper 不是早期简化 MLP，而是按 Mellow 论文/官方代码落实：

```text
x: [B, 1025, 768] = concat(CLS, c2l(framewise))
p1 = Linear(768 -> 576, bias=False)(x)
p2 = Dropout(0.5)(Linear(576 -> 576, bias=False)(GELU(p1)))
p  = LayerNorm(p1 + p2)

CLS = p[:, :1]                         # 保留，不池化
frames = AvgPool2d(kernel=(8,1), stride=(8,1))(p[:, 1:])
audio_prefix = concat(CLS, frames)     # 1 + 1024/8 = 129 tokens
```

- 两个 Linear 权重均 Xavier uniform 随机初始化，无 bias。
- LayerNorm 初始化 weight=1、bias=0。
- `c2l` 和整个 projection bridge 都在音频训练开始时随机初始化；它们不是从文本 MeSH checkpoint 继承的。
- DDP 构造时各 rank 的模型参数会由 DDP 同步，不能把每 rank 的 seed 差异理解成八套独立 mapper。
- 每段音频输出 129 tokens；两段音频合计 258 tokens。

### 5.4 实际送入语言模型的序列

```text
[audio1:129] [separator:1] [audio2:129] [separator:1]
[真实 prompt tokens] [真实 answer tokens] [仅 batch 尾部 padding]
```

- 音频 prefix 总长固定 260。
- 不新增 `begin_of_audio`、`end_of_audio` 或专用 audio special token。
- separator 从现有 tokenizer/model 词表解析：优先 tokenizer 的 sep token，其次 `!`，再 fallback 到配置/eos/bos。SmolLM2 当前路线通常使用 `!`，但代码不硬编码词表 id。
- separator 不是新增参数；它的 label 为 `-100`，不会被直接监督，但其输入 embedding 可通过后续 answer loss 获得间接梯度。

### 5.5 prompt、padding、answer-only loss 与 NTP

这是必须保持的正确实现：

1. prompt 单独 tokenize：`add_special_tokens=True`、最多 129 tokens、不 padding。
2. answer 单独 tokenize：`add_special_tokens=False`、最多 250 tokens、不 padding。
3. 对每个样本先拼成真实的 `prompt + answer`。
4. 再在 batch 内把完整文本序列统一右 padding 到该 batch 最长长度。
5. 模型依据每条样本的 `prompt_lengths` 和 `answer_lengths` 构造 labels。

labels 规则：

- 两段音频、两个 separator、全部 prompt、所有尾部 padding 都是 `-100`。
- 只有精确的真实 answer 区间等于相应 `text_ids`，参与 loss。
- `build_labels` 会断言有效 label 数严格等于 `sum(answer_lengths)`。
- HF `ForCausalLMLoss` 内部执行标准 one-token causal shift。因此第一个 answer token 由它前面的最后一个真实 prompt token 预测，不存在把 answer 自身喂给自己或在 prompt/answer 中间插 pad 的 NTP 错位。

早期实现曾分别 padding prompt 和 answer 后再拼接，短 prompt 的 pad 会插到 prompt 与 answer 之间；该错误已经改成“先真实拼接、再统一 padding”。不要回退。

### 5.6 上下文长度

```text
audio prefix:       260
max prompt:         129
max answer:         250
最大实际使用长度:   639
runtime hard limit: 768
```

训练采用 batch 内最长的动态右 padding，不会把每个 batch 都填到 768。Stage 3 按要求只做轻量 manifest/路径审计与 768 合同检查，不逐条读取 wav、不重采样、不统计 tokenizer 长度。

### 5.7 哪些参数训练

| 模块 | 状态 |
|---|---|
| HTSAT AudioSet backbone | 冻结，eval |
| Mellow `c2l(527->768)` | 训练 |
| bias-free 768->576->576 projection、LayerNorm | 训练 |
| 完整 MeSH LM（embedding、prefix/middle/suffix、norm、lm_head） | 训练 |
| 6 个 MeSH router | 训练并保存 |

## 6. ReasonAQA manifest 的最终状态

Stage 1 解析规则：

- AudioCaps 记录从 `/data/audiocaps_v2/{train,val,test}` 解析。
- 普通 Clotho 可使用 Clotho v2.1 对应 split。
- `ClothoAQA\audio_files\...` 记录直接从官方 `/data/clotho_aqa_audio/audio_files` 解析，不再强行映射到 Clotho v2.1。
- `filepath2` 为空时令 `audio2_path=audio1_path`，并记录 `audio2_reused=true`、`audio2_source=filepath1_duplicate`。
- 最后仍有 12 条 train 记录指向同一个实际不存在的音频。用户已明确决定使用 `--drop-unresolved` 丢弃，不用空路径继续训练。

最终 manifest：

```text
DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12

TRAIN=$DIR/reasonaqa_train.jsonl
VAL=$DIR/reasonaqa_val.jsonl
TEST=$DIR/reasonaqa_test.jsonl
```

远程通过后的行数：

| split | records |
|---|---:|
| train | 968,059 |
| val | 114,188 |
| test | 161,695 |
| total | 1,243,942 |

无需为当前训练重新生成 manifest，除非原始 metadata/音频目录变化，或再次改变 unresolved policy。

## 7. 音频阶段化审计与当前真实状态

| Stage | 设备 | 内容 | 当前状态 |
|---|---|---|---|
| Stage 0 | CPU | 环境、依赖、外部代码目录和 checkpoint 基础检查 | 早期已使用；以最新 Stage 2 环境为准 |
| Stage 1 | CPU | 解析并审计 ReasonAQA manifest，不读 waveform | **PASS（drop 12 后）** |
| Stage 2 | 1 GPU | 构造 Mellow/HTSAT、严格加载 AudioSet checkpoint、单音频前向和输出 shape | **PASS** |
| Stage 3 | CPU | 快速检查最终 manifest 路径、answer、split overlap 和 768 上下文合同；不逐条读 wav/tokenizer | **PASS** |
| Stage 4 | 1 GPU | 真实 batch 前向/反向、严格 mapper 结构、129/260 token、answer-only labels、c2l/bridge 有限梯度、MeSH 可训练、HTSAT frozen | 旧 mapper 曾 PASS；**严格 mapper 版本待重跑** |
| Stage 5 | 8 GPU | DDP/topology 短 smoke（默认 2 optimizer steps） | 当前严格 mapper 版本待确认 |
| Stage 7 | 8 GPU | 10 optimizer steps，保存复合 checkpoint，真实 reload，恢复 optimizer/scheduler/RNG 后再前反向审计 | 旧 mapper 曾 PASS；**严格 mapper 版本待重跑** |
| FORMAL | 8 GPU | 3 epochs 正式训练 | 代码就绪，当前结果未登记 |

已知 PASS report：

```text
Stage 2:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage2_v2/htsat_single_audio_audit.json

Stage 3:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/stage3_manifest_audit_drop12.json
```

旧 Stage 7 产物，仅用于理解历史，不能续当前严格 mapper：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/stage7_unified_padding_20260908_2/checkpoint-000010
```

原因：此后 mapper 的 bias、投影顺序、dropout、初始化和 8 倍池化合同已被改正；旧 `audio_bridge.pt` 不应静默加载进新架构。

## 8. 正式音频训练合同

### 8.1 数据分片与 shuffle

- 8 个 rank 都各自创建 DataLoader，不是 rank 0 统一读取。
- `DistributedSampler(dataset, num_replicas=8, rank=rank, shuffle=True, drop_last=True)` 给每个 rank 分配互不重复的 index 子集。
- 每个 epoch 调用 `sampler.set_epoch(epoch)`，所以每个 epoch 的全局 permutation 都变化，同时同一 seed/epoch 下仍可确定性复现。
- 各 rank 同时打开不同音频文件不会造成数据语义冲突；存储 I/O 吞吐可能成为性能瓶颈，但不等于读取重复或互相覆盖。
- `num_workers` 默认是每 rank 0，即读取发生在 8 个 rank 主进程内，不是全局只有 rank 0。若设 N，则总 worker 数约为 `8*N`。
- 当前 `--val-manifest` 被记录/传入，但训练循环没有周期性 validation；不要把传入 val path 理解成已经计算 val loss。

### 8.2 batch、步数和尾部

正式配置：

```text
world size:                    8
micro-batch per GPU:           8
samples per micro-step:        64
gradient accumulation:         4
effective samples/optimizer:   8 * 8 * 4 = 256
epochs:                        3
```

对 train=968,059：

```text
DistributedSampler 每 rank samples: floor(968059/8) = 121007
DataLoader 每 rank microbatches:     ceil(121007/8) = 15126
完整 GA windows/epoch:               floor(15126/4) = 3781 optimizer steps
总 optimizer steps:                 3781 * 3 = 11343
warmup:                              ceil(11343 * 0.05) = 568
```

代码只把完整的 4-microbatch accumulation window 变成 optimizer step。每 epoch 最后 2 个 microbatch 不优化，下一 epoch 会重新 shuffle；这样每一步始终保持 effective batch 256。`[audio-train] ... batch=X/15126` 中的 batch 指当前 rank 在本 epoch 已消费的 **microbatch 游标**，不是 optimizer step，也不是单个样本序号。

### 8.3 optimizer、scheduler 与精度

```text
optimizer:       AdamW
betas:           (0.9, 0.95)
weight decay:    0.1
max LR:          1e-3
min LR:          0
schedule:        568-step linear warmup + cosine decay to 0
gradient clip:   global norm 0.5
autocast:        BF16
```

GA/DDP 语义：前 3 个 micro-step 使用 DDP `no_sync()`，第 4 个同步；每个 microbatch 的 loss 除以 4 后 backward；随后 DDP 对各 rank 梯度求平均、clip 一次、optimizer step 一次、scheduler step 一次。按样本数计算的 effective batch=256 正确。

需要保留的精确解释：HF loss 默认对当前 microbatch 中所有非 `-100` answer tokens 求 mean。当前实现累积的是 32 个 local microbatch token-mean 的等权平均（8 ranks × 4 microsteps），并不是把全局所有 answer token 先求总和再除以全局 answer-token 总数。因此样本 batch 合同正确，但在 answer 长度差异较大时，它不等价于严格 global token-weighted mean。除非用户明确要求改变目标函数，不要在不说明的情况下重写 reduction。

### 8.4 日志

rank 0 每 10 optimizer steps 打印：

- step/total、progress、epoch；
- 当前 epoch microbatch cursor；
- loss、LR、grad norm；
- step time、samples/s、audio seconds/s；
- answer token 计数；
- GPU allocated/reserved/max allocated/max reserved；
- router stats。

注意：当前日志中的 `loss` 是 rank 0 最后一个 microbatch 的 `output.loss`，不是 4 个 microstep、8 个 rank 聚合后的 optimizer-step loss；`effective_answer_tokens` 也只是最后一个 microbatch经 rank 汇总的计数，不覆盖整个 GA window。它们可用于趋势观察，但不能被描述成严格全局 step loss。loss 约为 1 本身不能证明 prompt 被错误加入；answer-only mask 已有硬断言。若要判断模型是否真正利用音频，应做正确音频、置零音频、跨样本打乱音频的对照，而不是只看训练 loss。

### 8.5 checkpoint 保存、清理和 resume

正式训练每 1000 optimizer steps 保存，并只保留当前 output dir 内最近 4 个带完整 marker 的 checkpoint；final step 总会保存。按 11,343 步预计最终保留：

```text
checkpoint-009000
checkpoint-010000
checkpoint-011000
checkpoint-011343
```

每个复合 checkpoint 必须包含：

```text
mesh_model/                 # 完整 MeSH LM，含 routers
tokenizer/
audio_bridge.pt             # bridge + c2l
training_state.pt           # optimizer, scheduler, global_step,
                            # epoch, batch_in_epoch, per-rank RNG
audio_mesh_config.json      # 架构、mapper、数据 hash、训练配置
checkpoint_complete.json    # 完成 marker
```

冻结的 HTSAT 不重复保存；`audio_mesh_config.json` 记录外部 HTSAT checkpoint 路径，reload 时必须仍可访问该文件及 Mellow 代码。

- **从文本 checkpoint 开始新的音频训练**：传 `--mesh-checkpoint`，不要传 `--resume-from`。此时 c2l/mapper 随机初始化，optimizer/scheduler/step 从零开始。
- **恢复音频训练**：传 `--resume-from /.../checkpoint-NNNNNN`。程序从其中的 `mesh_model`、`tokenizer`、`audio_bridge.pt`、`training_state.pt` 恢复，并校验 batch cursor 在 GA 边界。
- 不要把已完成 9244 步的文本 checkpoint 当成音频 resume；它只能是 `--mesh-checkpoint`。
- 不要把旧 mapper 的 Stage 7 checkpoint 当成新 mapper resume。

独立 checkpoint reload 审计入口：

```text
code/RSmol/scripts/audit_audio_checkpoint_5_10x2_5_mesh_mellow.py
code/RSmol/run_audio_checkpoint_audit_5_10x2_5_mesh_mellow_5090.sh
```

## 9. 当前推荐的远程执行顺序

在远程仓库根目录：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull
```

### 9.1 先重跑严格 mapper 的 Stage 4

Stage 4 是 1 GPU 作业。以 wrapper `--help`/当前脚本参数为准；核心输入应使用当前第二轮 MeSH checkpoint、最终 train manifest、Mellow root 和 HTSAT checkpoint。不要引用旧 Stage 4 PASS 代替这一步。

```bash
bash code/RSmol/run_audio_stage4_5_10x2_5_mesh_mellow_5090.sh \
  --mesh-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244 \
  --htsat-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt \
  --mellow-root /hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main \
  --manifest /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl \
  --report-path /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/stage4_mellow_exact_mapper_20260909.json
```

如果 Stage 4 wrapper 的底层参数名与上面不一致，先运行 `bash ... --help` 或直接查看 wrapper/对应 `.sh`，不要凭旧聊天命令猜测。

### 9.2 再跑 Stage 5 和 Stage 7

Stage 5/7 都是 8 GPU。Stage 7 必须产生新的严格 mapper `checkpoint-000010` 并通过内置保存/重载/前反向审计，才能作为当前 checkpoint 完整性证据。输出目录必须使用新名字，绝不能覆盖旧 `stage7_unified_padding_20260908_2`。

入口：

```text
code/RSmol/run_audio_stage5_5_10x2_5_mesh_mellow_5090.sh
code/RSmol/run_audio_stage7_5_10x2_5_mesh_mellow_5090.sh
```

当前正式配置对应的 smoke 命令：

```bash
bash code/RSmol/run_audio_stage5_5_10x2_5_mesh_mellow_5090.sh \
  --mesh-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244 \
  --htsat-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt \
  --mellow-root /hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main \
  --train-manifest /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/stage5_mellow_exact_mapper_20260909 \
  --micro-batch-size 8 \
  --gradient-accumulation-steps 4
```

Stage 5 PASS 后再提交：

```bash
bash code/RSmol/run_audio_stage7_5_10x2_5_mesh_mellow_5090.sh \
  --mesh-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244 \
  --htsat-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt \
  --mellow-root /hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main \
  --train-manifest /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/stage7_mellow_exact_mapper_bs256_20260909 \
  --micro-batch-size 8 \
  --gradient-accumulation-steps 4
```

### 9.3 正式训练（当前 canonical 命令）

```bash
bash code/RSmol/run_audio_formal_5_10x2_5_mesh_mellow_5090.sh \
  --mesh-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244 \
  --htsat-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt \
  --mellow-root /hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main \
  --train-manifest /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl \
  --val-manifest /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_val.jsonl \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/formal_mellow_exact_mapper_round2_bs256_20260909 \
  --micro-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --epochs 3 \
  --max-lr 1e-3 \
  --min-lr 0 \
  --save-every 1000 \
  --checkpoint-retention 4
```

底层 formal shell 已注入 micro 8、GA 4、save 1000、retention 4；命令中再次显式写出是为了让实验合同自说明。后置 CLI 参数会覆盖默认值。

## 10. 文件地图

### 10.1 当前 MeSH 文本路线

```text
code/RSmol/recursive_model_5_10x2_5_mesh.py
code/RSmol/scripts/convert_stepwise_5_10x2_5_mesh.py
code/RSmol/scripts/audit_stage1_5_10x2_5_mesh.py
code/RSmol/scripts/train_stage4_5_10x2_5_mesh_ddp.py
code/RSmol/scripts/evaluate_stage3_5_10x2_5_mesh.py
code/RSmol/run_*5_10x2_5_mesh*.sh
tests/test_5_10x2_5_mesh_static.py
tests/test_stage3_5_10x2_5_mesh_static.py
```

### 10.2 音频 preflight 与当前正式路线

```text
# Stage 0/1/2 和 manifest（历史名称 5_10_5_mellow）
code/RSmol/audio_5_10_5_mellow/manifest.py
code/RSmol/scripts/audit_audio_stage0_5_10_5_mellow.py
code/RSmol/scripts/prepare_reasonaqa_manifest_5_10_5_mellow.py
code/RSmol/scripts/audit_audio_stage2_htsat_5_10_5_mellow.py

# 当前 MeSH 音频复合模型
code/RSmol/audio_5_10x2_5_mesh_mellow/data.py
code/RSmol/audio_5_10x2_5_mesh_mellow/model.py
code/RSmol/audio_5_10x2_5_mesh_mellow/stage3.py
code/RSmol/scripts/audit_audio_stage3_5_10x2_5_mesh_mellow.py
code/RSmol/scripts/audit_audio_stage4_5_10x2_5_mesh_mellow.py
code/RSmol/scripts/train_audio_5_10x2_5_mesh_mellow_ddp.py
code/RSmol/scripts/audit_audio_checkpoint_5_10x2_5_mesh_mellow.py
code/RSmol/run_audio_*5_10x2_5_mesh_mellow_5090.sh
tests/test_audio_5_10x2_5_mesh_mellow_static.py
```

### 10.3 历史模型与评测

```text
code/RSmol/recursive_model.py
code/RSmol/recursive_model_5_10_5.py
code/RSmol/recursive_model_5_10_5_linear.py
code/RSmol/recursive_model_5_10x7_5.py
code/RSmol/recursive_model_5_10xr_5.py
code/RSmol/recursive_model_5_10xpoisson_parcae.py
```

Parcae 完成 checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
stage4_5_10xpoisson_parcae_lr8e-4_mb2_ga64_20260903_172047/checkpoint-009244
```

Parcae Stage 3 eval：

```text
code/RSmol/scripts/evaluate_stage3_5_10xpoisson_parcae.py
code/RSmol/run_stage3_eval_5_10xpoisson_parcae_5090.sh
```

## 11. 已遇到并已处理的问题

这些历史故障对新会话很重要，避免重复误诊：

1. MeSH Stage 0 转换报缺 `lm_head.weight`：SmolLM2 tied embedding 导致 state dict 表现不同，转换器已按 tied-weight 语义处理。
2. MeSH Stage 1 加载报 meta tensor `.item()`：router 初始化改成 meta-safe。
3. 文本 Gate D 找不到 parquet：真正 shard 在 `.../SmolLM2-135M-10Bsubset/data`。
4. Gate E 输入 E 却运行 D：旧环境变量/launcher 传递问题已处理；任何 gate 仍应看 report 内实际 `gate`，不能只看提交命令。
5. NCCL 在 step 520 broadcast timeout：某 rank 的 router collapse 硬检查先退出导致集体通信失配；router collapse 已改为 non-fatal warning。
6. Parcae eval 报 `B is not identity initialized`：对已训练 checkpoint 强制检查初始 identity 不合理，评测逻辑已修正。
7. MeSH eval `UnboundLocalError: recursive_model`：已修复局部变量初始化。
8. ReasonAQA 大量 missing：AudioCaps/Clotho/ClothoAQA 数据包与路径需要分别解析；最终只余 12 条同源缺失记录并按用户决定 drop。
9. Stage 2 缺 `importlib_resources`/`torchlibrosa`：属于依赖缺失；安装并修正 Mellow/official HTSAT 导入与 checkpoint key 映射后已 PASS。
10. Stage 2 一度选择 official fallback 且 200 个 unexpected/201 个 missing：实现来源和 checkpoint target 对错了；当前加载器明确走 Mellow wrapper 严格匹配 HTSAT child。
11. prompt/answer 分别 padding：会把 pad 插到边界中间；已改为每样本先真实拼接，再 batch padding。
12. 早期 audio mapper 与论文不完全一致：当前已修正为 c2l、CLS+frames、bias-free 768->576->576 residual、dropout 0.5、LayerNorm、CLS-preserving avgpool8，并加强 Stage 4 静态/运行时硬审计。

## 12. 当前仍需关注的风险与待办

1. **先拿到严格 mapper 的新 Stage 4/5/7 PASS**，再开始或确认 FORMAL；旧 PASS 不能自动继承。
2. 正式训练的 `val_manifest` 目前不执行周期验证；若需要模型选择或过拟合监控，要单独设计 validation，但不要未经授权改变正式合同。
3. 日志 loss 不是全局 GA 聚合 loss；如要严谨比较实验，需要新增只读聚合指标或明确改变 reduction 的方案。
4. answer-only loss 能保证监督区间正确，但不能保证模型真正使用音频。后续应做 zero/shuffle/correct audio 对照和 ReasonAQA evaluation。
5. HTSAT 未写入复合 checkpoint，远程清理外部代码或 AudioSet checkpoint 会使 resume 失败。
6. 训练数据每 epoch shuffle，但由于 `drop_last` 和只取完整 GA window，每 epoch 有极少量尾部样本不参与该 epoch；下一 epoch permutation 会变化。
7. 独立音频训练代码审查曾被请求，但旧会话中没有收到最终审查报告。新会话如需要“独立审查已通过”的结论，必须重新执行，不能沿用不存在的结果。
8. 根级 `run_audio_formal_5_10x2_5_mesh_mellow_5090.sh` 顶部注释仍写着旧的 micro 4/GA 1；实际执行合同由内层 `scripts/train_audio_formal_5_10x2_5_mesh_mellow_ddp.sh` 和后置 CLI 决定，当前是 micro 8/GA 4。不要依据那行旧注释提交实验。

## 13. 本地质量检查

修改当前音频路线后至少运行：

```powershell
python -m py_compile code/RSmol/recursive_model_5_10x2_5_mesh.py
python -m py_compile code/RSmol/audio_5_10x2_5_mesh_mellow/data.py
python -m py_compile code/RSmol/audio_5_10x2_5_mesh_mellow/model.py
python -m py_compile code/RSmol/audio_5_10x2_5_mesh_mellow/stage3.py
python -m py_compile code/RSmol/scripts/audit_audio_stage4_5_10x2_5_mesh_mellow.py
python -m py_compile code/RSmol/scripts/train_audio_5_10x2_5_mesh_mellow_ddp.py
python -m pytest -q tests/test_5_10x2_5_mesh_static.py tests/test_audio_5_10x2_5_mesh_mellow_static.py
git diff --check
```

本地缺少 torch、Mellow、HTSAT 或远程权重时，可以记录测试因环境未运行，但不能把它写成 PASS。GPU 作业 report 是最终运行证据。

## 14. 给下一位 Codex 的接手清单

1. 完整阅读本 README。
2. 运行 `git status --short`，保护用户已有修改，不做 reset/checkout 覆盖。
3. 明确本次任务属于哪条路线；当前默认是音频 `5-10x2-5-mesh-mellow`。
4. 同时阅读 `audio_5_10x2_5_mesh_mellow/{data.py,model.py}`、训练脚本、对应 launcher 和静态测试，不只看 README 猜实现。
5. 涉及 manifest 时使用最终 `stage1_with_clotho_aqa_v2_drop12` 路径。
6. 涉及新的音频训练初始化时使用第二轮 MeSH `formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244`。
7. 区分 `--mesh-checkpoint`（只取文本权重、从零开始音频训练）与 `--resume-from`（恢复完整音频 checkpoint）。
8. 不恢复旧 `stage7_unified_padding_20260908_2/checkpoint-000010` 到严格 mapper。
9. 所有 GPU 步骤通过 5090 `vc submit` wrapper；CPU preflight 才直接运行。
10. 每次交付列出本地改动、静态检查、尚未远程验证项，并给出逐步远程命令。
11. 远程返回失败时先找最早的 Python exception/rank，不要只分析末尾 NCCL `ChildFailedError`。
12. 新的远程 PASS、输出目录、checkpoint 或训练超参数确认后，及时更新本 README，保持它可独立完成下一次交接。

## 15. 参考资料

- MeSH 论文：<https://arxiv.org/pdf/2510.07739>
- MeSH 官方仓库：<https://github.com/LivingFutureLab/MeSH/>
- Mellow 论文：<https://arxiv.org/pdf/2503.08540>
- HTSAT 官方仓库：<https://github.com/RetroCirce/HTS-Audio-Transformer>
- Parcae 论文：<https://arxiv.org/pdf/2604.12946>
- Parcae 参考实现：<https://github.com/sandyresearch/parcae>

外部论文解释设计，当前仓库代码和最新远程 report 决定实际实验合同。
