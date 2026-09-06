# RSM_bridge

## 0. 先读这里：项目交接与文档约定

本文件是本仓库面向后续 Codex 会话的权威交接文档，最后更新于 2026-09-06。它优先描述当前代码和最近的实验状态；后面的历史说明只用于理解项目演化，不能覆盖前面的当前配置。

本项目的特殊工作方式必须牢记：

- 本地 Windows 工作区只保存核心 Python、Shell、测试和文档代码。
- 模型权重、数据集、转换产物、训练输出和 checkpoint 全部在远程 Linux 服务器。
- GitHub 只用于中转代码，不能把权重、parquet、日志或 checkpoint 提交进仓库。
- 本地可以做静态检查、py_compile、单元测试和代码审查；真正的 GPU 审计、数据预审和正式训练必须在远程环境完成。
- 任何新会话都应先读本文件，再阅读对应的 code/RSmol 脚本和测试，确认当前实验分支，不要把一个版本的修改带到另一个版本。

项目目标是先在 SmolLM2-135M 上验证“前缀层 + 共享循环层 + 后缀层”的文本循环模型训练流程，随后把同一训练和梯度审计框架迁移到音频循环模型。当前主工作仍是文本模型阶段。

状态标签含义：

- 已远程通过：有用户提供的远程 PASS 日志或已确认的 checkpoint。
- 代码已落地、待远程验证：本地代码和静态检查完成，但不能据此宣称远程通过。
- 历史/对照：用于复现实验或对比，不是当前默认主线。

## 1. 当前实验总览

| 版本 | 循环形式 | 当前用途与状态 | 远程模型/输出位置 |
|---|---|---|---|
| 原始 SmolLM2-135M | 原模型 | 基础推理 smoke 已通过 | /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2 |
| 15R recursive | 15 个物理层逻辑执行两次 | 历史主线，Stage 1/2/4 及正式训练已通过 | /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4 |
| 5-10-5 recursive | 前 5 层 + 10 层共享循环 + 后 5 层；h0=e | 历史固定深度对照，正式训练已通过 | /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5 |
| 5-10-5 linear | 非递归线性对照 | 代码和评测流程存在；历史正式训练曾在中途停止，不能标记为完整通过 | 远程输出目录以旧日志为准 |
| 5-10xpoisson-parcae | Parcae 注入递归；每序列截断 Poisson 深度 | 当前已完成正式 9244 步，lr 8e-4/8e-5 的 checkpoint 已存在；正在准备从该 checkpoint 初始化第二轮 | /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10xpoisson-parcae |
| 5-10xr-5-poisson | 直接递归 h_{t+1}=Block(h_t)；每序列截断 Poisson 深度 | 当前独立对照版本；最新训练代码已加入进度日志和启动错误传播修复，仍需远程重新验证 | /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10xr-5-poisson |

最重要的隔离规则：Parcae 和直接递归是两个不同模型，不得互相替换循环公式、初始化、注入结构或审计脚本。修改动态直接递归版本时，只改 *_5_10xr_5_* 文件；修改 Parcae 版本时，只改 *_5_10xpoisson_parcae_* 文件。

## 2. 两个当前实验版本的精确定义

### 2.1 5-10xpoisson-parcae

物理结构是 5 个 prefix 层、10 个共享 middle 层、5 个 suffix 层，对应逻辑深度为 5 + 10T + 5，其中每个样本的 T 单独采样。

对输入序列，前 5 层输出记为 e。循环初始化和更新采用 Parcae 风格：

~~~text
e = Prefix(x)
u = PreludeNorm(e)
h0 = per-forward, per-sequence like-init state
h_{t+1} = MiddleBlock( A_bar(h_t) + B_bar(u) )
y = Suffix(h_T)
~~~

A_bar(h_t) 和 B_bar(u) 是相加，不是拼接；前缀输出先经过 Prelude Norm，再作为固定注入输入；h0 不是一个可训练的全局 nn.Parameter，而是按当前 forward/序列生成、具有模型初始化尺度的状态。A_bar、B_bar、衰减参数、注入参数和 no-weight-decay 分组应以 recursive_model_5_10xpoisson_parcae.py 及对应审计脚本为准，设计来源是 Parcae 的 injection.py 与 parcae.py。

训练深度采样：

~~~text
K ~ Poisson(lambda=7), restricted to {4,5,6,7,8,9,10}
P_trunc(k) = P(K=k) / sum_{j=4}^{10} P(K=j)
~~~

每张卡上的每个序列独立采样 T_i；单卡内部取 Tmax=max_i(T_i)，所有序列执行到本卡 Tmax。已经达到自身深度的序列在剩余步执行 identity/no-op，保证张量形状和 DDP All-Reduce 一致。不同 rank 不要求共享同一 Tmax，也不广播深度。

推理默认固定 T=7；审计时可显式测试 T=4..10。训练反向传播的有效参数梯度窗口固定为最后 4 次循环；早期循环必须保留对 hidden state 的 autograd 路径，不能用 torch.no_grad() 切断前 5 层梯度。

### 2.2 5-10xr-5-poisson

这是与 Parcae 版本配对的直接递归对照，结构仍然是前 5 层、10 层共享中间层、后 5 层，但循环公式保持原始形式：

~~~text
e = Prefix(x)
h0 = e
h_{t+1} = Block(h_t)
y = Suffix(h_T)
~~~

这里不使用 Parcae 的 h0、Prelude Norm、A_bar/B_bar 或注入项。循环次数仍按截断 Poisson(lambda=7, 4..10) per-sequence 采样，推理默认 r=7，单卡本地 Tmax 对齐和 identity/no-op 规则相同。反向传播仍固定最后 4 次循环获得中间层参数梯度，同时保留整个 hidden-state 路径，让前 5 层能够得到梯度。

## 3. 当前正式训练合同

以下是两个当前 Poisson 版本正式训练共同采用的主配置；如某脚本有更严格的参数校验，以脚本校验为准：

- 数据：/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset
- 原始数据记录：85 个 parquet shard，约 10,058,156 行，约 25.6 GB；正式训练使用固定 shard 顺序、max_length=1024，短序列保留并在 microbatch 内动态 padding。
- DDP：8 卡/8 rank。
- 单卡 microbatch：2。
- 梯度累积：64。
- 全局 effective batch：8 × 2 × 64 = 1024 个样本。
- 优化器：AdamW；betas、weight decay 和参数分组以训练脚本为准。
- 最大/最小学习率：8e-4 / 8e-5。
- scheduler 总步数：9244；optimizer step：9244；warmup：463 步。
- 保存间隔：每 500 步；保留最近 3 个 checkpoint。
- 正式数据训练不再使用 4500 步；当前正式合同是完整 9244 步。
- 训练期间每 10 个 optimizer step 由 rank 0 打印并写入进度记录：loss、学习率、step 速度、循环次数统计、GPU allocated/reserved/peak memory 和 CPU RSS。动态版本输出 stage4_progress.jsonl。

训练中的深度统计应是跨 rank 汇总后的样本统计，而不是只看 rank 0 的本地 batch。显存统计使用各 rank 的最大值，不能把单卡局部值误认为全局峰值。

## 4. checkpoint 语义：继续训练与重新初始化必须区分

1. 真正 resume：设置 RESUME_FROM=/path/checkpoint。这会恢复模型、optimizer、scheduler、全局 step、数据游标和随机状态。若 checkpoint 已经是 009244，通常不会再跑一整轮 9244 步。
2. 从 checkpoint 权重开始新一轮：把 checkpoint 目录作为 MODEL_DIR，不要设置 RESUME_FROM，并使用新的输出目录。这样只读取模型配置、权重和 tokenizer，optimizer、scheduler、step 和数据游标从零开始。

用户当前指定的 Parcae 第二轮初始化 checkpoint：

~~~text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
stage4_5_10xpoisson_parcae_lr8e-4_mb2_ga64_20260903_172047/checkpoint-009244
~~~

canonical 提交形式如下；不要设置 RSMOL_5_10XPOISSON_PARCAE_RESUME_FROM：

~~~bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

INIT_CKPT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10xpoisson_parcae_lr8e-4_mb2_ga64_20260903_172047/checkpoint-009244
NEW_OUTPUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10xpoisson_parcae_lr8e-4_mb2_ga64_round2_$(date +%Y%m%d_%H%M%S)

RSMOL_5_10XPOISSON_PARCAE_STAGE4_GATE=FORMAL \
RSMOL_5_10XPOISSON_PARCAE_MODEL_DIR="$INIT_CKPT" \
RSMOL_5_10XPOISSON_PARCAE_TOKENIZER_PATH="$INIT_CKPT/tokenizer" \
RSMOL_5_10XPOISSON_PARCAE_OUTPUT_DIR="$NEW_OUTPUT" \
RSMOL_5_10XPOISSON_PARCAE_WORLD_SIZE=8 \
RSMOL_5_10XPOISSON_PARCAE_MICRO_BATCH_SIZE=2 \
RSMOL_5_10XPOISSON_PARCAE_GRADIENT_ACCUMULATION_STEPS=64 \
RSMOL_5_10XPOISSON_PARCAE_MAX_LR=8e-4 \
RSMOL_5_10XPOISSON_PARCAE_MIN_LR=8e-5 \
RSMOL_5_10XPOISSON_PARCAE_SEED=0 \
RSMOL_5_10XPOISSON_PARCAE_JOB_NAME=parcae-lr8-round2-$(date +%m%d%H%M) \
bash run_stage4_5_10xpoisson_parcae_3090.sh
~~~

## 5. 远程路径、环境与代码中转

本地：C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge

远程 checkout：/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM

远程环境与重要目录：

~~~text
Python/env: /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/rsmol
原始模型:   /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2
数据集:     /hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset
输出根目录: /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol
~~~

最后记录的远程环境为 Python 3.10、PyTorch 2.11.0+cu128、Transformers 4.54.1、Datasets 3.6.0、PyArrow 23.0.1；作业日志是环境版本的最终依据。

常用队列/容器：

- 当前 3090 版本：队列 pdgpu-3090，容器 docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1，典型资源 -c 32 -m 256G -g 8。
- 旧 15R 主线使用过 5090 队列；不要把 5090 launcher 的资源参数复制到 3090 Poisson 版本。

推荐工作流：

1. 本地修改并做 py_compile、静态测试和 git diff --check。
2. 提交/推送代码到 GitHub。
3. 远程 checkout 拉取指定 commit。
4. 在远程先做转换、Stage 1 和 Stage 4 gate，再提交正式训练。
5. 只把远程日志、report 路径、checkpoint 路径回填到 README 或实验记录，不把大文件拉回本地。

## 6. 代码入口与文件地图

核心模型：

- code/RSmol/recursive_model.py：早期/15R 递归模型。
- code/RSmol/recursive_model_5_10_5.py：固定 5-10-5 递归模型。
- code/RSmol/recursive_model_5_10_5_linear.py：5-10-5 线性对照。
- code/RSmol/recursive_model_5_10xr_5.py：直接递归 Poisson 对照的模型实现。
- code/RSmol/recursive_model_5_10xpoisson_parcae.py：Parcae 注入 Poisson 版本。

Parcae 版本入口：

- scripts/convert_stepwise_5_10xpoisson_parcae.py：从原始 SmolLM2 转换模型。
- scripts/audit_stage1_5_10xpoisson_parcae.py：单卡 Stage 1 审计。
- scripts/smoke_recursive_5_10xpoisson_parcae.py：Parcae smoke/契约审计。
- scripts/train_stage4_5_10xpoisson_parcae_ddp.py：8 卡 Stage 4 和正式训练。
- code/RSmol/run_convert_stepwise_5_10xpoisson_parcae_3090.sh、code/RSmol/run_audit_stage1_5_10xpoisson_parcae_3090.sh、code/RSmol/run_stage4_5_10xpoisson_parcae_3090.sh：远程提交包装器。

直接递归版本入口：

- scripts/convert_stepwise_5_10xr_5.py：模型转换。
- scripts/smoke_recursive_5_10xr_5.py：Stage 1 单卡审计。
- scripts/train_stage4_5_10xr_5_ddp.py：Stage 4 和正式训练。
- code/RSmol/run_convert_stepwise_5_10xr_5_3090.sh、code/RSmol/run_smoke_recursive_5_10xr_5_3090.sh、code/RSmol/run_stage4_5_10xr_5_3090.sh：远程提交包装器。

测试入口在 tests/。当前动态版本的静态测试是 tests/test_5_10xr_5_static.py；其余测试文件按版本命名，修改某个版本后优先运行对应测试。

## 7. 阶段化验证流程

### Stage 0：代码与转换前检查

检查模型配置、层映射、参数数量、输出目录和 tokenizer 路径。模型转换产物必须在远程模型目录，不得进入 Git。

### Stage 1：单卡审计

至少覆盖：

- 模型可加载、输出 shape/dtype/finite。
- scalar 推理深度 r/T=4..10，默认推理深度 7。
- 每序列 Poisson 采样支持、单卡本地 Tmax/no-op 对齐。
- 反向图覆盖：前 5 层、后 5 层、最后 4 个循环的参数梯度；早期循环不能用 no_grad 破坏 hidden-state 路径。
- Parcae 版本额外检查 PN(e)、A+B 注入、h0 初始化、注入参数和 no-weight-decay 分组。
- cache、generation、save/reload 契约。BF16 增量 logits 的数值差异可能来自精度/缓存路径；如果只是语义 warning，不应替代硬性的 finite、shape、slot-length 和 reload 完整性检查。

Stage 1 是远程 GPU 审计，不要因为本地没有权重而尝试本地运行完整模型。

### Stage 4：DDP gate

- Gate A：合成数据和梯度路径。
- Gate D：真实 parquet 数据短跑/预审，通常用于确认 8 rank、padding、深度统计、梯度非零和显存。
- Gate E：短跑 checkpoint 保存、重载和 resume 语义。
- FORMAL：完整 9244 optimizer steps。

正式训练前必须确认数据预审 report、模型路径、tokenizer 路径、输出目录和 job 配置均来自同一版本。若 rank 0 在数据预审阶段抛异常，所有 rank 应收到原始异常摘要；过去的 Connection closed by peer 只是 rank0 先退出后其他 rank 被动失败的表象。

## 8. 训练数据与 checkpoint 合同

数据读取使用固定 shard manifest、固定行顺序和确定性 cursor。每个样本按 tokenizer 的 input_ids 截断到 1024；短样本保留，microbatch 内动态 padding，并用 valid-token mask 计算 loss。跨 shard/epoch 的未完成 batch 和 cursor 必须写进 checkpoint，resume 不能重复大段数据。

训练报告应至少记录：loss、学习率、有效 token 数、深度/Tmax 直方图、总梯度范数及 finite/nonzero 状态。详细审计点要逐 rank 检查 prefix/middle/suffix、Parcae 注入组（如适用）和 trainable parameters。

checkpoint 保存必须包含：模型权重、配置、tokenizer、optimizer/scheduler、训练 step、数据 cursor、随机状态和 manifest/配置摘要。保存后应检查所有文件存在且能离线 reload；不要只看目录名。

## 9. 当前已知问题和处理原则

1. 数据路径错误：no parquet shards 是远程数据路径/挂载问题，不是显存不足。先检查目录、权限和 shard manifest。
2. DDP Connection closed by peer：通常是某一 rank（多见 rank0）在 collective 之前发生异常。先找最早的 rank 日志，不要只看最后一个 NCCL 错误。
3. 显存 OOM：优先降低单卡 microbatch 或检查 cache/activation 是否意外保留；effective batch 可以用增大梯度累积维持，但不能在未核对合同的情况下随意改变全局 batch。
4. Transformers cache/pad warning：key_cache/value_cache 旧属性、缺少 attention mask、pad=eos 的 warning 不等于模型失败，但正式 generation 审计必须显式传 attention_mask 和 pad_token_id，并在代码迁移时逐步适配新 cache API。
5. BF16 cache logits 不完全相等：先报告 max/mean diff、cosine、argmax；不要把合理的低精度差异误判为结构错误，也不能因此跳过 save/reload、shape、finite 和 cache slot 硬审计。
6. 工作树未推送：动态 5-10xr-5 最新训练日志和 rank0 预审错误传播修复曾在本地完成静态检查，但在没有远程重新运行前只能标为“待远程验证”。

## 10. 本地质量检查

在提交代码前，至少执行：

~~~powershell
python -m py_compile code/RSmol/recursive_model_5_10xr_5.py
python -m py_compile code/RSmol/scripts/train_stage4_5_10xr_5_ddp.py
python -m pytest -q tests/test_5_10xr_5_static.py
git diff --check
~~~

如果修改的是 Parcae 版本，替换为对应 Parcae 模型、训练脚本和测试；不要因为本地缺少 torch/权重而宣称 Stage 1 或 Stage 4 通过。远程通过的依据必须是作业 report/log 或真实 checkpoint。

## 11. 历史背景与对照结果

最初的 15R 版本用于验证：共享物理层多次逻辑执行、训练梯度路径、数据 cursor、DDP 梯度同步和 checkpoint resume。随后固定 5-10-5 版本把 30 个逻辑层拆成 5+10+5，用于更清楚地观察循环中间层的共享参数梯度。线性版本作为非递归对照存在。

曾经出现过 5-10x7-5、幂律 r^2、4500 步等方案草稿；它们已被当前 Poisson(lambda=7, support=4..10) 和正式 9244 步合同取代。除非用户明确要求复现实验，不要把这些旧配置写回当前 launcher 或 README 主状态。

Stage 3 离线 benchmark、旧 15R/固定 5-10-5 checkpoint 和历史 5090 作业仍可作为回归对照，但不得与当前 3090 Poisson 正式训练结果混用。

## 12. 音频模型后续路线

文本循环模型阶段的目标是先固定循环深度采样、每序列对齐、反向传播窗口、梯度审计、数据 cursor 和 DDP 训练合同。音频阶段应复用这些工程契约，再替换输入/输出表征和数据管线；不能在文本版本尚未完成远程验证时同时引入音频数据、不同缓存语义或新的深度采样策略。

## 13. 给未来 Codex 会话的最短交接规则

1. 先读本 README，再读当前任务指定版本的模型、训练脚本、launcher 和测试。
2. 先确认是在 Parcae 还是直接递归版本，确认是 conversion、Stage 1、Stage 4 gate、formal 还是 checkpoint round2。
3. 不读取本地不存在的权重/数据；需要远程验证时给出远程提交命令并区分“代码完成”和“远程通过”。
4. 不修改其他版本；尤其不要把 Parcae 注入逻辑复制到 5-10xr-5，也不要把直接递归的 h0=e 复制到 Parcae。
5. 修改完成后做本地静态检查，记录未提交/未远程验证状态；只有拿到远程 PASS 或真实 checkpoint 才更新状态为已通过。

## 14. 参考资料

- Parcae 论文：https://arxiv.org/pdf/2604.12946
- Parcae 参考实现：https://github.com/sandyresearch/parcae
- 本仓库的实现、审计和 launcher 是实验合同的最终实现来源；论文和外部仓库只用于解释设计，不替代当前代码行为。

## 15. 最新任务：隔离的 `5-10x2-5-mesh`（2026-09-06）

当前正在新增一套完全隔离的 MeSH 对照版本，目标是从现有两次循环的
`5-10-5` 初始化模型出发，只改变循环部分为 MeSH 风格的读写路由与槽位
机制。此版本尚未远程转换、Stage 1 或 Stage 4 验证；在拿到远程 PASS
之前只能标记为“实现进行中/待远程验证”。

### 15.1 不可变的实验边界

- 新版本标识：`5-10x2-5-mesh`；代码、转换脚本、审计脚本、训练脚本、
  launcher、测试和输出目录都必须使用独立文件名/目录。
- 不得修改或复用修改现有 `5-10-5`、`5-10xr-5-poisson`、
  `5-10xpoisson-parcae`、15R 等版本的运行时代码；可以只读参考它们的
  映射、数据管线和阶段化审计框架。
- 初始化来源是当前远程两次循环 `5-10-5` 模型（前 5 层 + 10 层共享中间层
  执行 2 次 + 后 5 层），具体 checkpoint/model 目录以远程转换时确认的
  `5-10-5` 产物为准。
- **不启用官方实验的 `sqrt(d)` embedding scale**。Embedding 输出保持现有
  SmolLM2/`5-10-5` 的尺度；不能在新模型中偷偷加入全局 `sqrt(hidden_size)`
  乘法。

### 15.2 MeSH 结构合同

严格参照 MeSH 论文（https://arxiv.org/pdf/2510.07739）和官方实现（https://github.com/LivingFutureLab/MeSH/）：

- 循环次数固定为 2（推理也默认 2；本版本不是 Poisson 动态深度）。
- 前 5 层输出作为初始 embedding/外部输入 `e`；循环层每一步接收当前
  hidden state 和读路由得到的 memory 内容。
- 槽位 memory 的形状为 **`[batch, 5, sequence_length, hidden_size]`**，即
  `5 × L × d`；5 个槽位对应 MeSH 的 memory slots，而不是把槽位拼进
  hidden dimension。
- 使用 MeSH 的读路由、写路由和 slot mixing 逻辑；路由权重必须按官方实现
  的维度、归一化和读/写时序实现，不能改成 Parcae 的 `A_bar h + B_bar e`
  注入，也不能改成简单 concat。
- 循环主体仍以现有 SmolLM2 的中间 `Block` 为计算单元；前缀/后缀层保持
  `5-10-5` 的层映射和输出接口，除 MeSH memory/routing 外不改变模型骨架。
- 训练、梯度、cache、save/reload、generation 的审计必须覆盖 memory slot
  长度、读写路由概率和两次循环状态更新；不能把旧版只检查 `h_t` 的审计
  原样当作通过标准。

### 15.3 实现状态与交接规则

本版本由专门子代理负责落代码，主代理负责最终审查。子代理交付后必须
完成以下检查再提交远程：

1. `git diff --check`、对应 Python 文件 `py_compile` 和隔离版本静态测试。
2. 核对文件名中均明确包含 `5_10x2_5_mesh` 或等价 `5-10x2-5-mesh` 标识，
   没有覆盖旧版本文件。
3. 静态确认没有 `sqrt(d)` scale、没有 Parcae 的 `A_bar/B_bar`、没有
   Poisson 深度采样，且 memory 真实为 `[B,5,L,d]`。
4. 远程按“转换 → Stage 1 单卡审计 → Stage 4 Gate A/D/E → 正式训练”顺序
   验证；每一步都记录远程命令、report 路径和 checkpoint 路径。

