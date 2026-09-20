# RSM_bridge：Recursive SmolLM / Audio MeSH 项目交接

> 最后同步：2026-09-18
> 本文件是新 Codex 会话的首要交接依据。新会话必须先完整阅读本文，再阅读“当前主线文件”中列出的代码与最新远程 report。若本文、旧聊天和代码冲突，以当前代码行为与最新远程证据为准，并及时把差异补回本文。

## 0. 当前状态：先读这一节

项目最初在 SmolLM2-135M 上研究循环语言模型，目前唯一默认主线是 ReasonAQA 音频训练：

```text
已解码 waveform（mono、32 kHz、10 s、float32）
  -> 冻结的 HTSAT AudioSet backbone
  -> 可训练的 Mellow c2l(527->768)
  -> 可训练的 Mellow bridge(768->576->576, avgpool8)
  -> 单音频 130-token / 双音频 260-token prefix
  -> 5-10x2-5 MeSH / SmolLM2-135M
  -> 只对 answer token 计算 causal-LM loss
```

当前正式训练不再在线随机读取原始 wav，也不再使用早期 64-shard mmap 方案。它使用已经完成并审计通过的六个连通分量 waveform partition：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/
rsmol_reasonaqa_train_component_partitions6_32k_10s_f32_v2
```

每次只加载一个 partition。每个 DDP rank 将该 partition 的全部唯一 waveform 克隆到自己的匿名 CPU RAM，训练完该 partition 后严格释放并核对每 rank RSS 与 cgroup anon 回落，通过后才加载下一 partition。

当前关键状态：

- 六分区物化：远程 `PASS`，共 968,059 条 QA、54,046 条唯一音频、64.42785263061523 GiB waveform；跨 partition 音频复制为 0。
- partition PERF20：p0、p2 均已远程跑通；完整预加载后稳态训练约 1.6–1.7 秒/step。
- 正式训练前的旧 v1（答案未追加 EOS）smoke：用户已确认两次均 `PASS`：
  - p2 加载 → 10 步 → 释放；p0 加载 → 10 步 → 释放；保存 step-20 checkpoint。
  - 从 step-20 checkpoint resume；p1 加载 → 2 步 → 释放；保存 step-22 checkpoint。
- 2026-09-18 已把训练合同升级为 `component_partitions6_rank_ram_compact_audio_answer_eos_v2`：每条 answer 最后恰好有一个受监督的 `<|endoftext|>`。由于输入/标签合同发生改变，旧 v1 smoke 不能放行 v2 正式训练，必须按相同 20+2 流程重跑。
- 正式训练：准备采用 10 epochs；EOS v2 smoke 和正式作业均尚未登记远程完成。
- 正式训练入口会重新读取两个 smoke report，验证训练合同、step 游标、resume 关联、MeSH 30 层轨迹和全部 8 rank 内存释放；验证不通过则拒绝启动。
- 原始 SmolLM2-135M 音频对比线已在本地改造成同一套六分区、compact-prefix、answer-EOS-v2、10-epoch 训练合同，训练 wrapper 使用 `pdgpu-3090`；当前状态为代码就绪，新的 20+2 smoke 与正式训练均尚未登记远程 PASS。
- 固定 5-10-5 recursive 音频对比线也已改造成六分区、compact-prefix、answer-EOS-v2、10-epoch 合同；保持 20 个物理层和精确 `5-10-10-5` 逻辑轨迹、无 MeSH router/memory，训练 wrapper 使用 `pdgpu-5090`。当前仅代码就绪，新的 20+2 smoke 与正式训练尚未登记远程 PASS。

文本模型默认初始化自第二轮低学习率 MeSH checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244
```

初始化边界必须说清楚：

- MeSH/SmolLM2 文本模型、routers 和 memory 参数：从上述 checkpoint 加载。
- HTSAT：从 AudioSet checkpoint 加载并完全冻结。
- `c2l(527->768)`：随机初始化并训练。
- bridge `768->576->576`：随机初始化并训练；线性层 Xavier uniform，LayerNorm weight=1/bias=0。
- `--mesh-checkpoint` 表示从文本模型开始一次新的音频训练。
- `--resume-from` 只接受本分区音频训练器生成的完整复合 checkpoint，会恢复 LLM、c2l、bridge、optimizer、scheduler、游标和所有 rank RNG，不会重新随机初始化 mapper。

## 1. 工作方式和不可违反的边界

本项目采用“本地代码、远程资产”工作流：

- 本地 Windows 工作区只保存 Python、Shell、测试和文档。
- 模型权重、原始数据、manifest、waveform store、日志和 checkpoint 都在远程 Linux。
- GitHub 只中转代码，不提交音频、权重、parquet、checkpoint 或大日志。
- 本地可做代码修改、`py_compile`、静态测试和 diff 审查；真实 CUDA、HTSAT、数据与内存释放必须以远程 report 为证据。
- 所有 GPU 作业必须通过仓库中的 `vc submit` wrapper；不要在登录节点直接运行 CUDA Python 或 `torchrun`。
- 保留各实验路线隔离。不要把 MeSH、固定 recursive、Parcae、Poisson recursion 或原始 SmolLM2 基线的模型/checkpoint/脚本互相替换。
- 不得因本地测试通过就声称远程 `PASS`。

状态术语：

- **远程 PASS**：用户提供过明确 PASS report、日志或真实 checkpoint。
- **代码就绪**：本地实现和检查通过，但远程尚未确认。
- **历史/对照**：保留用于复现或比较，不是默认主线。

## 2. 环境和远程路径

### 2.1 工作区、环境和资源

```text
本地仓库:
C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge

远程仓库:
/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM

项目代码目录（提交命令通常在这里执行）:
/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

conda 环境:
/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/rsmol

输出根目录:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol
```

已知远程环境：Python 3.10、PyTorch 2.11.0+cu128、Transformers 4.54.1。

当前正式 wrapper 资源：

```text
queue: pdgpu-5090
image: docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1
resources: -c 32 -m 256G -g 8 -n 1
```

### 2.2 模型与外部代码

```text
原始 SmolLM2:
/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2

当前文本 MeSH 初始化 checkpoint:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/
formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244

Mellow 官方代码:
/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main

HTSAT 官方代码:
/hpc_stor03/sjtu_home/jinwei.zhang/code/HTS-Audio-Transformer-main

HTSAT AudioSet checkpoint:
/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt
```

### 2.3 数据、manifest 和当前 partition store

```text
ReasonAQA metadata:
/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/{train,val,test}.json

AudioCaps:
/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2/{train,val,test}

Clotho v2.1:
/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1/{development,validation,evaluation}

ClothoAQA audio:
/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_audio/audio_files

最终 drop12 train manifest:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/
stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl

当前六分区 waveform store:
/hpc_stor03/sjtu_home/jinwei.zhang/data/
rsmol_reasonaqa_train_component_partitions6_32k_10s_f32_v2
```

## 3. 当前文本 backbone：5-10x2-5 MeSH

核心实现：

```text
code/RSmol/recursive_model_5_10x2_5_mesh.py
```

结构合同：

- SmolLM2-135M hidden size 576。
- 20 个物理 decoder layers：prefix 5 + shared middle 10 + suffix 5。
- 30 个逻辑执行位置：`5 + 10 + 10 + 5`。
- 中间 10 个物理层固定循环两次；不是 Poisson 动态深度。
- 5 个 memory slots。
- 三个阶段分别有 3 个 write routers 和 3 个 read routers，共 6 个独立线性 router。
- memory 是单次 forward 内的状态，不跨样本、batch 或 optimizer step 持久化。
- 训练 forward 使用 `use_cache=False`。

首个真实 optimizer step 会审计：

- 逻辑层轨迹精确匹配 `5-10-10-5`。
- 两次 middle loop 输入/输出都有有限梯度。
- 6 个 router 都有有限梯度。
- 10 个 shared middle 物理层都获得有限梯度。

PERF20 和当前 partition trainer 为避免 router 统计中的额外 GPU→CPU 同步，关闭逐 forward router 数值统计，但仍保留首步 router 参数梯度与完整逻辑轨迹审计。router/memory 参数本身仍正常训练。

## 4. 当前音频模型与精确序列合同

核心文件：

```text
code/RSmol/audio_5_10x2_5_mesh_mellow/model.py
code/RSmol/audio_5_10x2_5_mesh_mellow/data.py
```

### 4.1 Waveform 和 HTSAT

当前 partition 中保存的不是 HTSAT tokens，而是已经完成以下预处理的 waveform：

- mono；
- 32 kHz；
- 取前 10 秒，过短右侧补零；
- 固定 `[1, 320000]`；
- little-endian float32；
- 每条 waveform 1,280,000 bytes。

训练时 waveform 仍会进入 HTSAT。HTSAT backbone 全部 `requires_grad=False` 且保持 `eval()`；不能把 waveform store 描述成“已经离线算好 HTSAT feature”。

HTSAT/Mellow 路径输出 latent/CLS 等价特征和 527 类 framewise event-presence，Mellow `c2l` 将 527 维映射到 768 维。

### 4.2 Mapper

可训练 mapper 包括：

```text
c2l: Linear(527 -> 768)
bridge:
  Linear(768 -> 576, bias=False)
  + GELU
  + Dropout(0.5)
  + Linear(576 -> 576, bias=False)
  + residual
  + LayerNorm
  + CLS-preserving temporal avgpool8
```

每段音频输出精确 129 tokens：1 个 CLS/global token + 128 个下采样 frame tokens。

### 4.3 单/双音频 prefix

新的 partition 训练路线启用 `compact_single_audio_prefix=True`：

```text
真实单音频样本:
[audio1:129] [separator:1] [prompt] [answer]
prefix = 130

双音频样本:
[audio1:129] [separator:1] [audio2:129] [separator:1] [prompt] [answer]
prefix = 260
```

单槽/双槽按 manifest 的结构字段判断，不仅看两个路径是否相同：

- manifest 缺少第二音频、`audio2_reused=true`：真实单槽，prefix 130。
- manifest 明确存在两个槽，即使路径相同：仍是双槽，prefix 260；允许复用相同 waveform/embedding 计算，但不删除第二槽。

separator 从 tokenizer/model 词表解析，SmolLM2 当前通常使用 `!`，代码不硬编码词表 ID。

### 4.4 Prompt、answer、padding 和 loss

```text
prompt/question: 最多 129 tokens，add_special_tokens=True
answer 正文:     最多 249 tokens，add_special_tokens=False
answer 终止:     手动追加且只追加一个 <|endoftext|>
answer 总长:     最多 250 tokens（包含终止符）
```

代码要求 tokenizer 的 `<|endoftext|>` ID 与 `eos_token_id` 完全相同。若原始 answer 尾部已经含一个或多个 EOS，会先去掉尾部 EOS，再追加一个，保证每条 answer 恰好以一个 EOS 结束。EOS 位于 answer 区间内，`attention=1` 且 label 是真实 token ID，因此参与 causal-LM loss。即使 tokenizer 用 EOS 兼任 pad token，右侧 batch padding 仍为 `attention=0, label=-100`，不会进入 loss。

上下文上限：

```text
单音频最大: 130 + 129 + 250 = 509
双音频最大: 260 + 129 + 250 = 639
模型保护上限: 768
```

每条样本先独立构造真实 `prompt + answer`，之后才做 batch 右侧 padding。单/双音频可自然混在同一个 batch；每条样本先按自己的 130/260 prefix 拼接，再将完整多模态序列右侧补齐。

直接参与 loss 的区域严格如下：

| 区域 | attention | label |
|---|---:|---:|
| audio prefix | 1 | `-100` |
| separator | 1 | `-100` |
| prompt/question | 1 | `-100` |
| answer 正文及末尾 `<|endoftext|>` | 1 | 真实 token ID |
| batch padding | 0 | `-100` |

底层 Hugging Face causal-LM loss 做正常 one-token shift：第一个 answer token 由最后一个 prompt token预测，之后逐 token 自回归。audio 与 prompt 不作为监督目标，但它们是预测 answer 的上下文，因此 answer loss 会经由它们向 mapper 和 LLM 传播梯度；这是预期语义。

### 4.5 冻结和可训练参数

冻结：

- HTSAT backbone 全部参数；
- Mellow wrapper 中除 `c2l` 外的全部参数。

训练：

- `c2l(527->768)`；
- 768→576→576 bridge；
- 完整 MeSH/SmolLM2，包括 embeddings、20 个物理层、routers、memory 参数和 LM head。

optimizer 只接收 `requires_grad=True` 的参数。启动时 `trainable_parameter_audit()` 会硬检查 HTSAT frozen、wrapper 只有 c2l 可训练、c2l/bridge/MeSH 均处于训练模式。

## 5. ReasonAQA manifest 和六分区数据

最终用于分区规划的 train manifest SHA256：

```text
14cd324e4c78ba289c1d6b2ac26ba702f70d86ffc069536327bbd7cbd935c055
```

相关统计：

```text
QA rows:                    968,059
same-audio rows:            764,494（早期路径相同统计；结构单槽以 manifest 字段为准）
distinct two-audio rows:    203,565
unique audio:               54,046
total QA audio incidences:  1,171,624
```

构建方法：

1. 统计双音频关联图。
2. 对音频图做 union-find，完整连通分量不能拆开。
3. 将连通分量装箱到 6 个 partition。
4. 每条 QA 只属于一个 partition；其引用的所有音频都在本区。
5. 每个唯一音频只物化一次，跨 partition 复制为 0。
6. 从原唯一 waveform store 顺序复制成六个自包含 fixed-stride store。
7. 校验 manifest/index/waveform SHA256、payload 总字节数和抽样字节一致性。

远程 v2 物化结果：

```text
status: PASS
duplicated_audio: 0
total waveform: 64.42785263061523 GiB
source_payload_sha256_reverified: true

partition 0: 278,680 QA, 3,839 audio, 约 4.576 GiB
partition 1-5: 各约 137,876 QA、约 10,040 audio、约 11.97 GiB
```

partition 0 对应不可拆分的大型 Clotho 连通分量，因此在零复制条件下六份 QA 数量不可能严格均衡；这是已知且由用户接受的结果，不是物化错误。

规划与物化脚本保留用于审计/复现：

```text
code/RSmol/scripts/prepare_unique_audio_waveform_store.py
code/RSmol/scripts/plan_reasonaqa_component_partitions.py
code/RSmol/scripts/materialize_reasonaqa_component_partitions.py
```

## 6. 当前正式训练合同

唯一当前正式入口：

```text
code/RSmol/scripts/train_audio_partitioned_5_10x2_5_mesh_mellow_ddp.py
code/RSmol/scripts/train_audio_partition_formal_5_10x2_5_mesh_mellow_ddp.sh
code/RSmol/run_audio_partition_formal_5_10x2_5_mesh_mellow_5090.sh
```

旧的 `train_audio_5_10x2_5_mesh_mellow_ddp.py --gate FORMAL` 是历史在线/固定 prefix 路线，不是当前推荐正式入口。

### 6.1 Batch、优化器与调度器

```text
world size:                  8
micro batch per GPU:         8
gradient accumulation:       4
effective global batch:      8 * 8 * 4 = 256
precision:                   BF16 autocast
optimizer:                   AdamW
betas:                       (0.9, 0.95)
weight decay:                0.1
max LR:                      1e-3
min LR:                      0
warmup:                      ceil(total optimizer steps * 5%)
after warmup:                cosine decay to min LR
gradient clipping:           0.5, error_if_nonfinite=True
DDP broadcast_buffers:       False
DDP find_unused_parameters:  False
NCCL timeout:                30 minutes
NCCL async error handling:   enabled in shell wrapper
```

当前计划的 10 epochs：

```text
optimizer steps/epoch:  floor(968059 / 256) = 3781
total optimizer steps:  3781 * 10 = 37810
warmup steps:           ceil(37810 * 0.05) = 1891
```

如果显式传 `--warmup-steps`，必须精确等于上述公式，否则硬失败。

当前 loss reduction 需要准确描述：每个 rank 的每个 microbatch 由 HF 对所有非 `-100` answer tokens 求 mean，再除以 GA=4 反传；DDP 对 rank 梯度平均。因此目标是各 microbatch token-mean 的等权组合，不是把整个 256 样本的全部 answer token 先求总和再除以全局 token 数。除非用户明确决定改变目标函数，不要擅自重写 reduction。

### 6.2 Epoch、partition 顺序和分区内 shuffle

每 epoch 固定 3781 optimizer steps：

- partition 0：`floor(278680/256)=1088` 步，即 278,528 条；shuffle 后丢弃 152 条，不重复。
- partition 1–5：合计 2693 步；用累计平衡分配 538/539 步。
- 如果某个小 partition 的配额比实际 QA 多，缺额只从该 partition 本 epoch 已打乱序列的开头补齐；绝不跨 partition。
- 每个 partition 内所有 QA 统一 shuffle；单槽和双槽自然混合，不分池。
- 每个 epoch 重新生成确定性 partition 顺序和分区内 shuffle。
- partition 0 只能出现在位置 0–4，绝不放在最后。
- shuffle plan 会计算 SHA256；中途 resume 必须重新生成相同 plan 并匹配 hash。

默认 `seed=0` 时前三个 epoch 的分区顺序：

```text
epoch 1: [2, 1, 4, 3, 0, 5]
epoch 2: [0, 2, 1, 5, 4, 3]
epoch 3: [4, 5, 3, 0, 2, 1]
```

### 6.3 一个 partition 的完整生命周期

1. 验证根 materialization report 和当前 partition metadata。
2. 验证 `PASS`、零复制、ID、manifest/index SHA256、waveform 文件字节数。
3. 每个 rank 打开当前 partition store。
4. 每个 rank 按本地 audio ID 顺序将所有 waveform `.clone()` 到本进程匿名 CPU RAM。
5. 检查 resident audio 数、总字节、miss 数且 eviction=0。
6. 8 rank barrier；打印 `[partition-train] loaded ...`。
7. 构造该 epoch/partition 的确定性全局 QA shuffle 和 optimizer-window plan。
8. 每个 global step 的 256 条按 rank 切成 32 条；每 rank 再运行 4 个 microbatch，每个 8 条。
9. BF16 forward、answer-only loss、GA backward、clip、AdamW、scheduler。
10. 分区训练期间 waveform cache 的 miss/clone/resident 数必须不变，即不再读 store。
11. 分区结束 CUDA synchronize，删除 batch/plan/cache/dataset/tensor 引用，关闭 mmap。
12. 执行 `gc.collect()` 和 Linux `malloc_trim(0)`。
13. barrier 后最多轮询 120 秒。
14. 每 rank RSS 至少下降 `partition waveform bytes * 70%`。
15. cgroup anon 至少下降 `partition waveform bytes * 8 * 70%`。
16. 任一释放检查失败即硬失败，绝不加载下一 partition。

### 6.4 日志

rank 0 在以下时机打印 step 日志：

- 全局 step 是 10 的倍数；
- 当前 partition 最后一步（即使不是 10 的倍数）。

格式：

```text
[partition-train] step=180/37810 p2 180/538 loss=1.234567 lr=0.000095238 1.993s
```

其中：

- `loss` 是 rank 0 上当前 optimizer step 四个 microbatch loss 的算术平均，不是 8 rank 聚合的全局 loss。
- `lr` 是该 optimizer step 在 `optimizer.step()` 时实际使用的 LR。
- JSON report 仍保存每一步的 loss、LR、耗时和单/双/混合槽型标记。
- partition 加载完成和释放完成另有日志；完整内存/cgroup 数据写入 `partition_training_report.json`。

### 6.5 Checkpoint、保留与 resume

```text
save_every:           500 optimizer steps
checkpoint_retention: 4
final step:           always saved
```

checkpoint 使用临时目录完整写入并检查 required files 后原子发布。保存内容：

- 完整 MeSH/SmolLM2；
- c2l 与 bridge；
- optimizer、scheduler；
- global step 和 segment/partition-step 游标；
- 当前 plan hash；
- 8 个 rank 的 Python/Torch/CUDA RNG；
- partition inventory、hash、完整 schedule；
- prefix、batch、LR、warmup、timeout、释放阈值；
- HTSAT/Mellow 路径与 Mellow 源码 SHA256。

可在 partition 中间 resume。resume 必须使用新的空 output dir，并保持 epochs、seed、batch、LR、保存策略、timeout、释放阈值、partition inventory 和 Mellow provenance 完全一致。旧固定-260 prefix 音频 checkpoint、旧 Stage 7 checkpoint、文本 checkpoint 均不能作为本训练器的 `--resume-from`。

### 6.6 正式训练前 smoke gate

Smoke 入口：

```text
code/RSmol/scripts/train_audio_partitioned_5_10x2_5_mesh_mellow_ddp.py --mode smoke
code/RSmol/run_audio_partition_smoke20_5_10x2_5_mesh_mellow_5090.sh
```

固定 smoke schedule：

```text
p2: preload -> train 10 -> strict release
p0: preload -> train 10 -> strict release
save checkpoint-000020

new job/output dir, resume checkpoint-000020
p1: preload -> train 2 -> strict release
save checkpoint-000022
```

旧 v1 的两次 smoke 已由用户确认远程通过，但它们没有 EOS 标签，不能用于当前 v2 正式 gate。必须用当前代码重新运行上述两次 smoke，并从新的输出目录传入 v2 report：

```text
--smoke20-report .../partition_training_report.json
--smoke-resume-report .../partition_training_report.json
```

正式 gate 会核对：`training_contract=component_partitions6_rank_ram_compact_audio_answer_eos_v2`、status、mode、inventory、seed、0→20 与 20→22 游标、segment 步数、30 层轨迹、每段 8-rank RSS/cgroup release，以及 resume report 是否确实指向第一次的 checkpoint-000020。

### 6.7 10 epochs 正式提交模板

必须将 smoke report 替换为远程真实 PASS 路径：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

SMOKE20_REPORT=/actual/eos_v2_smoke20/output/partition_training_report.json
SMOKE_RESUME_REPORT=/actual/eos_v2_resume2/output/partition_training_report.json
FORMAL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/partition_formal_eos_v2_10epochs_20260918

bash run_audio_partition_formal_5_10x2_5_mesh_mellow_5090.sh \
  --output-dir "$FORMAL_DIR" \
  --epochs 10 \
  --smoke20-report "$SMOKE20_REPORT" \
  --smoke-resume-report "$SMOKE_RESUME_REPORT"
```

默认 seed 为 0。如果 smoke 使用了其他 seed，正式命令必须传同一 `--seed`。

## 7. 性能优化实验：保留的结论

详细历史 report 位于远程输出目录和旧聊天，本文只保留对当前设计有用的因果结论。

### 7.1 为什么不继续在线读取

原始在线读取具有严重且随机的 rank 长尾：data/forward/backward 的慢 rank 在不同 step/rank 间漂移，整个 DDP step 被最慢 rank 拖住。DataLoader `num_workers=2/rank`、64 CPU 核没有改善，反而受共享存储并发与调度影响。

四组 warm/cold 对照表明，主要损失来自 waveform 及时供应，不是文本 tokenization/collate：

- 在线路径稳态约 18.9 秒/step。
- waveform/full preload 可到约 1.6–1.7 秒/step。
- waveform preload 与 full preload 仅约 0.7 秒差异，优先解决 waveform I/O 是正确方向。

### 7.2 被放弃或仅保留复现的方案

- 64 个随机 waveform shard + mmap：约 27.5 秒/step，跨 shard 随机访问/page fault 很差；不用于当前正式训练。
- 一个 64.4 GiB 共享 mmap + rank0 page-cache warm：仍产生大量 major/minor faults与 rank 长尾；不用于正式训练。
- 小容量在线 LRU/prefetch：音频命中率低且 producer/consumer 争用，曾约 60 秒/step；不用于正式训练。
- 完整全量 waveform 每 rank 预加载：训练快但 8 rank 内存总量不可接受。

### 7.3 当前方案的依据

双音频关联图的完整连通分量允许把全部 64.4 GiB waveform 零复制切成六份。每个 rank 同时只持有一个 partition：

- p0 约 4.576 GiB/rank；
- 最大 partition 约 11.97 GiB/rank；
- 8 rank waveform 总驻留最大约 95.8 GiB，加模型/进程/caches 后在 256 GiB cgroup 中已通过 smoke。

partition PERF20 观测：第一步包含初始化开销，第二步下降，第三步起进入约 1.x 秒稳态。p0/p2 都证明完整分区加载后训练速度接近全量 preload 下限，因此转为当前正式生命周期设计。

### 7.4 PERF20 仍可用于诊断

历史性能入口仍保留：

```text
code/RSmol/scripts/train_audio_5_10x2_5_mesh_mellow_ddp.py --gate PERF20
code/RSmol/scripts/train_audio_perf20_5_10x2_5_mesh_mellow_ddp.sh
code/RSmol/run_audio_perf20_5_10x2_5_mesh_mellow_5090.sh
```

它支持 online、warm、waveform/full preload、shared store、rank RAM preload/prefetch、partition rank RAM 等隔离模式及 torch.profiler。它是诊断工具，不是当前正式训练入口。

## 8. 评测与生成

### 8.1 ReasonAQA samples generation 与 router CSV

MeSH 音频 checkpoint 的 ReasonAQA generation 入口：

```text
code/RSmol/scripts/generate_audio_checkpoint_reasonaqa.py
code/RSmol/scripts/generate_audio_checkpoint_reasonaqa.sh
code/RSmol/run_audio_checkpoint_reasonaqa_generation_3090.sh
```

支持默认前五个或 `--sample-indices` 指定 zero-based manifest 行号，并可导出所有 token 位置的 5 个 memory slots × 3 组 × write/read router 权重到 CSV。该入口现已兼容 partition-v2 checkpoint：严格核对 `component_partitions6_rank_ram_compact_audio_answer_eos_v2`、completion marker、130/260 prefix 和 answer-EOS 合同；结构单音频使用 `audio1 + separator1` 的 130-token prefix，显式双音频使用 260-token prefix，router CSV 也按每条样本的真实 prefix 标注 token region。当前默认 checkpoint 为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/partition_formal_answer_eos_v2_10epochs_20260918/checkpoint-037810
```

### 8.2 MMAU test mini

MeSH 音频 MMAU test mini 已升级到当前 partition-v2 checkpoint，并调用官方 `evaluation.py --input ...`。入口在运行前严格审计 `MMAU-v05.15.25` 的 1000 条 metadata、ID/任务/难度分布、metadata 与 scorer SHA256；模型侧严格核对 compact 130-token 单音频 prefix、768 context、checkpoint completion marker 与 Mellow/HTSAT provenance。正式结果固定保留官方 1000 条分母，任何缺音频、字段错配或推理 skip 都会阻止官方计分。

answer-EOS v2 模型的正常输出是 `c) It is plausible<|endoftext|>`；解码后的 `generated_text` 是 `c) It is plausible`。MMAU/MMAR prompt 已与 ReasonAQA 对齐为 `问题 a) ... b) ... c) ...`，不添加 `Choices:`；MMAR 的五、六选项自动使用 `e)`、`f)`。本地不再做任何选项识别或答案预解析：完整 `generated_text` 原样写入 MMAU 的 `model_output` 或 MMAR 的 `answer_prediction`，全部匹配与评分交给官方 `evaluation.py`；带特殊 token 的 `generated_text_raw` 只保存在 append-only 审计 JSONL 中。

当前 MeSH 文件：

```text
code/RSmol/scripts/evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.py
code/RSmol/scripts/evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.sh
code/RSmol/run_mmau_test_mini_5_10x2_5_mesh_mellow_5090.sh
```

默认 checkpoint 是本节 8.1 的 `checkpoint-037810`，队列为 `pdgpu-5090`。smoke 与正式评测必须复用同一 output dir；第二次提交会从前 5 条继续，绝不重复推理已完成样本：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
EVAL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/mmau_test_mini_checkpoint_037810_reasonaqa_prompt_raw_v1

RSMOL_MMAU_MODE=smoke RSMOL_MMAU_OUTPUT_DIR="$EVAL_DIR" \
  bash run_mmau_test_mini_5_10x2_5_mesh_mellow_5090.sh

# smoke PASS 后，以同一目录续跑全部 1000 条并执行官方计分
RSMOL_MMAU_MODE=full RSMOL_MMAU_OUTPUT_DIR="$EVAL_DIR" \
  bash run_mmau_test_mini_5_10x2_5_mesh_mellow_5090.sh
```

### 8.3 MMAR

MMAR 使用 Hugging Face 下载的 `MMAR-meta.json`、已解压的 `mmar-audio/audio/*.wav` 和下载包内官方 `code/evaluation.py`。入口逐条按官方 metadata 顺序生成，在运行前审计 1000 个唯一 ID、核心字段 canonical SHA256、modality/category 分布、全部音频存在性和官方 scorer SHA256。音频统一为 32 kHz，短音频补零、长音频取开头 10 秒；选择顺序不打乱。输出保持官方完整记录并新增 `answer_prediction`，由官方 scorer 原样计分。

```text
code/RSmol/scripts/evaluate_mmar_5_10x2_5_mesh_mellow.py
code/RSmol/scripts/evaluate_mmar_5_10x2_5_mesh_mellow.sh
code/RSmol/run_mmar_5_10x2_5_mesh_mellow_5090.sh
```

同样使用 `pdgpu-5090`，smoke→full 复用同一个目录完成续跑测试和正式评测：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
EVAL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/mmar_checkpoint_037810_reasonaqa_prompt_raw_v1

RSMOL_MMAR_MODE=smoke RSMOL_MMAR_OUTPUT_DIR="$EVAL_DIR" \
  bash run_mmar_5_10x2_5_mesh_mellow_5090.sh

# smoke PASS 后，以同一目录续跑全部 1000 条并执行官方计分
RSMOL_MMAR_MODE=full RSMOL_MMAR_OUTPUT_DIR="$EVAL_DIR" \
  bash run_mmar_5_10x2_5_mesh_mellow_5090.sh
```

两套评测均写出 `run_config.json`、`progress.jsonl`、`raw_generations.jsonl`、`skipped.jsonl`、`smoke_first5.jsonl`、`official_evaluation.txt` 与 `evaluation_report.json`。MMAU 的官方输入是 `predictions_fixed_order.json`；MMAR 的官方输入是 `predictions_answer_prediction.json`。新会话仍应先用以下命令核实文件没有改名：

```bash
rg --files code/RSmol | rg 'mmau|MMAU|mmar|MMAR'
```

### 8.4 原始 SmolLM2 音频基线

当前新训练入口（代码就绪，尚无远程 PASS）：

```text
code/RSmol/scripts/train_audio_partitioned_smollm2_135m_mellow_ddp.py
code/RSmol/run_audio_smollm2_135m_mellow_smoke20_3090.sh
code/RSmol/run_audio_smollm2_135m_mellow_resume2_3090.sh
code/RSmol/run_audio_smollm2_135m_mellow_formal_3090.sh
```

训练合同为 `smollm2_component_partitions6_rank_ram_compact_audio_answer_eos_v2`。除文本 backbone 是原始 30 个独立物理层的 SmolLM2-135M、没有 MeSH router/memory 外，六分区数据生命周期、130/260 compact prefix、answer EOS、batch/优化器/LR、20+2 smoke、严格内存释放、checkpoint/resume 和 10-epoch 正式配置均与当前 MeSH 主线一致。正式训练同样必须由本路线两个真实 PASS smoke report 放行。

旧的三 epoch、固定 260-prefix checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/
formal_20260911_v1/checkpoint-011343
```

该 checkpoint 只用于历史生成/评测，不能作为新 partition-v2 trainer 的 `--resume-from`。它有独立 generation 与 MMAU 脚本，不能和 MeSH checkpoint 混用：

```text
code/RSmol/scripts/generate_audio_smollm2_checkpoint_reasonaqa.py
code/RSmol/scripts/evaluate_mmau_test_mini_audio_smollm2.py
code/RSmol/run_audio_smollm2_checkpoint_reasonaqa_generation_3090.sh
code/RSmol/run_mmau_test_mini_audio_smollm2_5090.sh
```

## 9. 历史与对照路线

以下内容保留为必要背景，但不是当前默认工作目标。

| 路线 | 定义 | 当前用途 |
|---|---|---|
| 原始 SmolLM2-135M | Hugging Face 原模型 | 音频基线与转换来源 |
| 15R recursive | 早期循环层实验 | 历史 |
| 固定 5-10-5 | 20 物理层，中间 10 层循环两次，无 MeSH | 独立音频对照 |
| 5-10-5 linear | 非递归线性对照 | 历史 |
| 5-10x7-5 | 中间层固定运行 7 次 | 历史 |
| 5-10xr-5 | 直接 recursion，可用 Poisson 深度 | 历史动态深度对照 |
| 5-10xpoisson-parcae | Parcae 注入递归 | 已完成历史路线 |
| 5-10x2-5 MeSH | 20 物理/30 逻辑、5 memory slots | 当前文本 backbone |
| Audio SmolLM2 | 不循环文本模型 + 同类音频 mapper | 当前对比基线 |

关键隔离规则：

- Parcae 的 `A_bar(h)+B_bar(u)`、特殊 `h0` 和 Poisson 深度不能带入 MeSH。
- 直接 recursion 的 `h0=e` 不能冒充 Parcae。
- 当前 MeSH 音频路线必须使用 `recursive_model_5_10x2_5_mesh.py`。
- 固定 5-10-5 音频对照使用 `recursive_model_5_10_5.py`，无 router/memory。
- 旧 `audio_5_10_5_mellow` 名称主要承载 manifest/HTSAT preflight，不代表当前 MeSH 正式 backbone 是固定 5-10-5。

固定 5-10-5 音频对照曾从以下 checkpoint 初始化并继续训练：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/
formal-epoch2-continue-20260902_184936/checkpoint-step-009244
```

当前该路线使用新的 partition-v2 trainer，但仍从上述文本 checkpoint 初始化：

```text
code/RSmol/scripts/train_audio_partitioned_5_10_5_recursive_mellow_ddp.py
code/RSmol/run_audio_5_10_5_recursive_mellow_smoke20_5090.sh
code/RSmol/run_audio_5_10_5_recursive_mellow_resume2_5090.sh
code/RSmol/run_audio_5_10_5_recursive_mellow_formal_5090.sh
```

训练合同为 `recursive_5_10_5_component_partitions6_rank_ram_compact_audio_answer_eos_v2`。它必须先完成本路线自己的 EOS-v2 20+2 smoke；旧三轮 composite checkpoint、MeSH/SmolLM2 smoke report 或文本初始化 checkpoint 都不能放行正式训练或作为新训练器的 resume source。

Parcae 历史完成 checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
stage4_5_10xpoisson_parcae_lr8e-4_mb2_ga64_20260903_172047/checkpoint-009244
```

Parcae 的历史运维合同仍被静态测试引用，保留以下最小信息：提交入口是 `run_stage4_5_10xpoisson_parcae_3090.sh`；转换时需要显式设置类似 `RSMOL_5_10XPOISSON_PARCAE_SOURCE_CHECKPOINT=/hpc_stor03/...` 的来源变量；Shell 批量审计范围包括 `scripts/*.sh`。正式训练 report 每个 optimizer step 保存 compact scalar metric per optimizer step，并用 `total_grad_norm_finite_nonzero` 记录梯度范数有限且非零。这些字段属于历史 Parcae 合同，不应复制进当前 partition 音频训练。

## 10. 当前主线文件地图

### 10.1 必读实现

```text
README.md
code/RSmol/recursive_model_5_10x2_5_mesh.py
code/RSmol/audio_5_10x2_5_mesh_mellow/model.py
code/RSmol/audio_5_10x2_5_mesh_mellow/data.py
code/RSmol/scripts/train_audio_partitioned_5_10x2_5_mesh_mellow_ddp.py
code/RSmol/scripts/train_audio_partition_smoke20_5_10x2_5_mesh_mellow_ddp.sh
code/RSmol/scripts/train_audio_partition_formal_5_10x2_5_mesh_mellow_ddp.sh
code/RSmol/run_audio_partition_smoke20_5_10x2_5_mesh_mellow_5090.sh
code/RSmol/run_audio_partition_formal_5_10x2_5_mesh_mellow_5090.sh
tests/test_audio_partition_training_static.py
tests/test_audio_5_10x2_5_mesh_mellow_static.py
```

### 10.2 数据和性能工具

```text
code/RSmol/scripts/prepare_unique_audio_waveform_store.py
code/RSmol/scripts/plan_reasonaqa_component_partitions.py
code/RSmol/scripts/materialize_reasonaqa_component_partitions.py
code/RSmol/scripts/train_audio_5_10x2_5_mesh_mellow_ddp.py
code/RSmol/scripts/train_audio_perf20_5_10x2_5_mesh_mellow_ddp.sh
code/RSmol/run_audio_perf20_5_10x2_5_mesh_mellow_5090.sh
tests/test_audio_perf20_static.py
tests/test_audio_waveform_cache_perf20_static.py
tests/test_reasonaqa_component_partitions.py
tests/test_reasonaqa_partition_materialization.py
```

### 10.3 原始 SmolLM2 partition 对比线

```text
code/RSmol/audio_smollm2_135m_mellow/model.py
code/RSmol/audio_smollm2_135m_mellow/data.py
code/RSmol/scripts/train_audio_partitioned_smollm2_135m_mellow_ddp.py
code/RSmol/scripts/train_audio_smollm2_135m_mellow_smoke20_ddp.sh
code/RSmol/scripts/train_audio_smollm2_135m_mellow_resume2_ddp.sh
code/RSmol/scripts/train_audio_smollm2_135m_mellow_formal_ddp.sh
code/RSmol/run_audio_smollm2_135m_mellow_smoke20_3090.sh
code/RSmol/run_audio_smollm2_135m_mellow_resume2_3090.sh
code/RSmol/run_audio_smollm2_135m_mellow_formal_3090.sh
tests/test_audio_smollm2_partition_training_static.py
tests/test_audio_smollm2_135m_mellow_static.py
```

### 10.4 固定 5-10-5 recursive partition 对比线

```text
code/RSmol/audio_5_10_5_recursive_mellow/model.py
code/RSmol/audio_5_10_5_recursive_mellow/data.py
code/RSmol/scripts/train_audio_partitioned_5_10_5_recursive_mellow_ddp.py
code/RSmol/scripts/train_audio_5_10_5_recursive_mellow_smoke20_ddp.sh
code/RSmol/scripts/train_audio_5_10_5_recursive_mellow_resume2_ddp.sh
code/RSmol/scripts/train_audio_5_10_5_recursive_mellow_formal_ddp.sh
code/RSmol/run_audio_5_10_5_recursive_mellow_smoke20_5090.sh
code/RSmol/run_audio_5_10_5_recursive_mellow_resume2_5090.sh
code/RSmol/run_audio_5_10_5_recursive_mellow_formal_5090.sh
tests/test_audio_5_10_5_recursive_partition_training_static.py
tests/test_audio_5_10_5_recursive_mellow_static.py
```

## 11. 已处理故障与不要重复的误诊

1. SmolLM2 tied embedding 导致转换时缺 `lm_head.weight`：转换器已按 tied-weight 语义处理。
2. MeSH meta tensor `.item()`：router 初始化已改为 meta-safe。
3. 文本 parquet 真正 shard 在 `SmolLM2-135M-10Bsubset/data`。
4. 某次 step 520 NCCL timeout 的根因是单 rank router-collapse 硬检查先退出，不是已经证明的网络故障；collapse 已改 diagnostic/warning。
5. prompt 和 answer 分别 padding 会把 pad 插到二者之间；当前已改为每条真实拼接后再 batch padding。
6. HTSAT checkpoint 必须加载 Mellow wrapper 对应的 child；曾经 official fallback 产生大量 missing/unexpected，当前严格拒绝 mismatch。
7. HTSAT frozen 不等于整条 wrapper `no_grad`：c2l 必须获得梯度，因此 forward 不能用全 wrapper `torch.no_grad()`。
8. 旧 mapper 不是严格 Mellow；当前已使用 c2l、CLS+frames、bias-free residual bridge、dropout 0.5、LayerNorm、CLS-preserving avgpool8。
9. 64-shard mmap 慢不是因为单纯 shard 数量，而是跨文件随机访问/page faults/共享存储长尾；当前正式路线已弃用。
10. shared mmap 被 rank0 warm 不等于每 rank 匿名 RAM；后者才在 PERF20 中达到理想稳态。
11. cgroup 路径可能是 v1/v2 混合；当前代码使用已有 hybrid-safe 探测，正式释放审计要求取得 anon 指标，否则拒绝下一 partition。
12. NCCL 日志停在 `Connected all rings` 不一定挂死：旧 smoke 在首次完整分区加载前没有进度打印，加载最大 partition 可持续数分钟；应先查作业状态与最早 Python exception，不要只看末尾 socket shutdown warning。
13. `loss≈1` 不能证明 prompt 被计入 loss；标签 mask 才是判断依据。模型是否真正利用音频应做正确/置零/打乱音频对照。

## 12. 当前风险与后续注意事项

- 旧 v1 两个 smoke 已由用户确认 PASS；当前 answer-EOS v2 合同必须重新跑 20+2 smoke，旧 report 会被正式 gate 拒绝。
- 原始 SmolLM2 partition 对比线同样必须先在 `pdgpu-3090` 重跑本路线 EOS-v2 的 20+2 smoke；旧三 epoch checkpoint、旧 STAGE7 report 或 MeSH smoke report 都不能放行该路线正式训练。
- 固定 5-10-5 recursive partition 对比线必须在 `pdgpu-5090` 完成本路线自己的 EOS-v2 20+2 smoke；旧三轮 composite checkpoint、文本初始化 checkpoint 或其他路线 report 都不能放行它的正式训练。
- 正式 10-epoch 训练尚未登记启动/完成。看到远程结果后及时写入 job、output dir、最终 checkpoint 和 report 状态。
- 当前日志 loss 只是 rank 0 的四个 microbatch 平均，不是 8 rank 全局聚合 loss。
- 当前训练没有周期性 validation；模型选择需要另行设计只读评测，不要把 train report 当验证集表现。
- ReasonAQA samples generation、MMAU test-mini 与 MMAR 均已兼容 partition-v2 compact 130-token 单音频 prefix；远程正式分数仍以各自 `evaluation_report.json` 和 `official_evaluation.txt` 为准。
- HTSAT checkpoint 和 Mellow 源码不打包进复合 checkpoint；远程清理外部文件会导致 resume 失败。
- 10 epochs 很长；resume 时必须使用相同 epochs=10 和全部训练合同，否则 checkpoint 校验会拒绝。
- output dir 必须是新的空目录；脚本拒绝覆盖已有输出。
- 若正式训练某个分区 release 失败，不要放宽阈值后直接继续；先检查该 rank 是否仍持有 batch/cache/tensor 引用以及 cgroup anon/RSS 证据。

## 13. 本地质量检查

修改当前主线后至少执行：

```powershell
python -m py_compile code/RSmol/recursive_model_5_10x2_5_mesh.py
python -m py_compile code/RSmol/audio_5_10x2_5_mesh_mellow/data.py
python -m py_compile code/RSmol/audio_5_10x2_5_mesh_mellow/model.py
python -m py_compile code/RSmol/scripts/train_audio_partitioned_5_10x2_5_mesh_mellow_ddp.py
python -m unittest discover -s tests -p "test_audio_partition_training_static.py"
python -m unittest discover -s tests -p "test_audio_5_10x2_5_mesh_mellow_static.py"
git diff --check
```

本地缺少 Torch/Mellow/HTSAT 或远程权重时，只能说明相关运行测试未执行，不能写成 GPU PASS。

## 14. 新 Codex 会话接手清单

1. 完整阅读本文。
2. 执行 `git status --short`，保护用户已有修改；不要 reset/checkout 覆盖。
3. 阅读第 10.1 节的当前主线实现和测试，不要只根据 README 猜代码。
4. 明确任务属于当前 partition 正式训练、历史 PERF20、评测，还是固定 5-10-5/SmolLM2 对照。
5. 当前音频新训练默认文本初始化必须是第二轮 MeSH checkpoint-009244。
6. 区分 `--mesh-checkpoint` 与 `--resume-from`。
7. 正式训练必须使用 partitioned trainer，不要误用旧 `--gate FORMAL`。
8. 任何正式提交都要核对两个 smoke report 的实际绝对路径、seed 和 `PASS`。
9. 远程失败时先找最早的 Python traceback/rank；末尾 NCCL `ChildFailedError` 通常只是连带结果。
10. 新的远程 PASS、job、output dir、checkpoint 或超参数确认后立即更新本文。

## 15. 参考资料

- MeSH：<https://arxiv.org/pdf/2510.07739>
- MeSH 官方仓库：<https://github.com/LivingFutureLab/MeSH/>
- Mellow：<https://arxiv.org/pdf/2503.08540>
- HTSAT：<https://github.com/RetroCirce/HTS-Audio-Transformer>
- Parcae：<https://arxiv.org/pdf/2604.12946>

外部论文用于解释设计；当前仓库代码、checkpoint config 和最新远程 report 决定实际实验合同。
