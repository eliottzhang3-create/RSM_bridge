# RSM_bridge

这是 Recursive SALM（暂称 RSM）实验的代码同步仓库。仓库的主要作用是在本地 Windows 机器上由 Codex
协助编写、检查和维护代码，再通过 GitHub 将代码同步到远程 Linux HPC 服务器执行模型转换、预训练、up training
和音频推理训练。

本 README 是项目的长期背景记录和操作约定。后续新的 Codex 窗口应先阅读本文件，再开始修改代码或设计远程实验。
其中“已验证”表示有实际日志、报告或检查结果支持；“计划”表示当前研究路线；“待确认”表示不能自行猜测，
需要由用户提供远程信息或实验结果。

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
Stage 3  使用 SmolLM2-135M-10B 的文本 up training
Stage 4  递归文本模型 benchmark 与 checkpoint 验证
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
`C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge` 与未来确认的远程 Linux 根目录；脚本中也必须根据运行平台使用对应路径。

### 11. 建议的远程目录约定

在远程根目录确认后，优先采用下面的逻辑分区；实际目录名可根据后续项目设计调整：

```text
<REMOTE_REPO_ROOT>/
  README.md                 # 与 GitHub 同步
  .gitignore                # 与 GitHub 同步
  code/ 或 src/             # 模型、数据、训练和评估代码，同步
  scripts/                  # 可复用运行脚本，同步
  configs/                  # 小型配置，同步
  tests/                    # 本地/远程审计脚本，同步
  data/                     # 远程数据、manifest、索引或缓存，不提交大文件
  models/                   # 远程模型快照或转换权重，不提交权重
  outputs/                  # 训练与评估结果，不提交
  logs/                     # 远程日志，不提交
```

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
- 新项目实际使用的 queue、container、conda environment、GPU 型号和资源请求仍需用户确认后写入本 README。

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

已确定：

- 本地项目路径为 `C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge`；
- GitHub 中转仓库为 `eliottzhang3-create/RSM_bridge`；
- 远程 Git checkout 路径为 `/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM`；
- 远程模型路径为 `/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2`；
- 远程文本数据路径为 `/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset`；
- 研究主线是 SmolLM2-135M 的 Stepwise 半层递归化、文本 up training 和 Mellow 风格音频推理训练；
- 文本 up training 数据集为 `EleutherAI/SmolLM2-135M-10B`；
- 递归目标暂按 15 个唯一层循环两次记录；
- 当前 README 背景和工程规则已建立。

待用户后续补充或确认：

- conda environment、container、queue、GPU 型号和资源申请；
- 实际代码目录和 runtime/submit 脚本目录；
- Stepwise 层映射的最终算法和转换 checkpoint 命名；
- 是否首先实现严格 Recursive baseline，何时加入 Relaxed layer-wise LoRA；
- relaxed LoRA 的作用模块、rank、SVD 初始化策略和 trainability；
- 音频编码器、mapper、输入协议、音频推理数据和评估 benchmark；
- up training 与音频训练的 batch、学习率、token budget、checkpoint retention 和 resume 方案。

在上述远程信息和训练细节确认前，任何 Codex 窗口都不应擅自补写具体 Linux 路径、环境名、checkpoint 路径、音频模型
权重位置或正式训练超参数。

## 参考资料

- [Mellow paper](https://arxiv.org/abs/2503.08540)
- [Mellow reference implementation](https://github.com/soham97/Mellow)
- [Relaxed Recursive Transformers paper](https://arxiv.org/abs/2410.20672)
- [SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [SmolLM2-135M-10B dataset card](https://huggingface.co/datasets/EleutherAI/SmolLM2-135M-10B)
