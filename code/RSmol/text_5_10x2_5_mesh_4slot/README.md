# 两次循环、四记忆槽 MeSH：隔离文本转换与 BF16 完整一轮训练

完整复制隔离六槽 `text_5_10x2_5_mesh_6slot` 的模型、转换器、完整一轮文本训练、
保存恢复和提交入口。槽数改为四，训练采用历史五槽的 BF16 autocast 方式，
保留 FP32 参数/AdamW 状态、全局有效 token 加权梯度和可恢复 Parquet 读取。
本路线拥有独立模型类、源码、架构/精度合同、环境变量、模型目录、训练输出与日志。
本地只维护代码，经 GitHub 同步；模型权重、Parquet 与实际训练均在远程 Linux。

## 模型与转换

- hidden size 576；20 个物理 decoder 层，30 次逻辑执行，轨迹为
  `0..14, 5..14, 15..19`，即 `5 + 10×2 + 5`。
- 四个临时 memory slots，形状 `[B,4,L,576]`。每次 forward 初始第 0 槽保存
  原始 input embeddings，另外三槽为零；memory 不跨 forward 持久化，也不是参数。
- 三组独立 write/read routers，共六个 `Linear(576, 4, bias=True)`，
  共十二个权重/偏置 tensor、13,848 个 router 参数。
- 保留母本的读写公式、query 来源、两次共享循环与 30 个逻辑 KV cache 槽。
- router 权重沿用原初始化公式 `std=sqrt(2/(slots×hidden_size))`，此处
  为 `sqrt(2/(4×576))`；采样后截到 ±3 std，bias 为零，默认转换 seed 为 0。
- 架构合同为 `logical_30_physical_20_5_10x2_5_mesh_4slot_3router`。

prelude 的 query/value 均取 prefix output；两轮循环的 query 取循环输入，
write value 取循环输出。仍执行三次 write/read，中间十层参数在两轮共享。

默认转换输入是干净的、仅转换过的远程目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5
```

也接受原始 30 层 `/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2`。
30→20 的源层映射保持
`0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29`；
若输入已是合法的仅转换 20 层模型，则直接复制全部 20 层，不再次抽层。
原有 MeSH artifact 和带训练状态的 checkpoint 默认拒绝作为来源。

转换器严格核对全部继承 tensor，处理 tied embedding/LM head，并将逐 tensor
相等检查结果、router 来源、层映射、seed 和环境写入 `mesh_conversion_metadata.json`。
新增随机参数仅为六个 routers；保留源 tokenizer。模型产物不能写入 Git checkout，
默认拒绝覆盖已有模型目录。

本地提交并推送代码到 GitHub 后，在远程登录节点进行 CPU-only 转换：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/scripts/convert_stepwise_5_10x2_5_mesh_4slot.sh
```

默认产物为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x2-5-mesh-4slot
```

转换 wrapper 自动激活远程 `rsmol`，设置 HF 离线模式并清空 `CUDA_VISIBLE_DEVICES`。
若需要从原始 30 层模型转换：

```bash
RSMOL_5_10X2_5_MESH_4SLOT_SOURCE_CHECKPOINT=/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2 \
bash code/RSmol/scripts/convert_stepwise_5_10x2_5_mesh_4slot.sh
```

可通过 `RSMOL_5_10X2_5_MESH_4SLOT_OUTPUT_DIR` 临时指定另一转换输出。
训练时用 `RSMOL_5_10X2_5_MESH_4SLOT_MODEL_DIR` 指向该模型，并为训练选择独立输出目录。

## 正式训练合同

| 项目 | 配置 |
|---|---|
| 文本初始化 | 新转换的四槽模型和 tokenizer；新 optimizer/scheduler/RNG/数据游标 |
| GPU / batch | 8 GPU × microbatch 8 × accumulation 16 = GBS1024 |
| 精度 | FP32 参数/AdamW 状态，模型前向 BF16 autocast，FP32 loss/router softmax；无 GradScaler，TF32 关闭 |
| Context | 1024；沿用原截断、batch padding 和 attention mask |
| 数据 | `data/SmolLM2-135M-10Bsubset/data/*.parquet`，直接读取 persistent 存储 |
| 分配与顺序 | 排序后 shard index modulo 8；首轮 shard 和 shard 内行顺序不变 |
| 训练长度 | 项目定义的完整一轮：9,244 次 optimizer updates；不进入第二轮 |
| 学习率 | max `8e-4`，min `0.1×max=8e-5` |
| 调度 | 前 `ceil(9244×0.05)=463` 步 warmup；其后沿用原 step-level cosine |
| 优化器 | AdamW，betas `(0.9, 0.95)`，eps `1e-8`，原 decay/no-decay 分组 |
| Weight decay | 矩阵参数 `0.1`；bias、norm 及低维参数 `0` |
| Loss | 全文本 causal next-token CE；只计 mask 有效目标；按 8 rank 整个累积窗口的有效 token 总数归一化 |
| 梯度裁剪 | 全局 L2 范数 `1.0`；非有限梯度报错；router collapse 保留非致命诊断 |
| 保存 | 每 500 optimizer steps 和最终步；保留最新三个完整 checkpoint |
| Seed | 默认 0，各 rank 沿用 seed+rank |

这里是纯文本预训练，无音频、prompt/answer 分区或 answer-only loss。
9,244 是原路线的固定一轮合同，训练器不会重新根据远程 Parquet 行数推导步数。
学习率公式的步索引和端点行为与五槽母本一致。

训练合同为 `text_mesh_x2_4slot_bf16_full_epoch_v1`。启动时必须确认 CUDA 支持 BF16；
不支持时明确失败，正式/短验证训练不会静默回退 FP32 或 FP16。CPU 只允许 Gate A 诊断，
报告会注明 autocast 未开启。模型转换沿用源权重存储精度，训练显式加载 FP32 参数，
checkpoint 的 `config.torch_dtype=float32` 与 BF16 autocast 训练相符，并不代表全 FP32 前向。
独立 metadata、training state 和训练报告均记录完整 precision policy。

autocast 仅覆盖模型前向，CE/归一化及 backward 在上下文外执行；保留原 `.float()`
router softmax。FP32 embedding/residual/memory 语义保持，因此 BF16 不表示所有激活都减半。

直接提交正式训练，无 smoke 报告前置门槛：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_stage4_5_10x2_5_mesh_4slot_formal_3090.sh
```

该入口通过 `vc submit` 提交到 `pdgpu-3090 / 64 CPU / 256G / 8 GPU / 1 node`，
使用原容器 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1` 和 `rsmol` 环境。
无需修改或升级环境。job 名、日志和输出均包含独立时间戳。

默认训练输出为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh_4slot/formal_1epoch_bf16_<timestamp>/
```

最终模型位于其下 `checkpoint-009244/`；完整训练审计为
`stage4_gate_FORMAL_audit.json`，诊断为 `ddp_diagnostics/`，提交日志在
`code/RSmol/log/text_x2_4s_formal.<timestamp>.JOB.log`。
训练 report 的 loss 是 rank 0 的有效 token 均值；梯度归一化使用全局有效 token 数。

显式指定模型与新输出目录：

```bash
RSMOL_5_10X2_5_MESH_4SLOT_MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x2-5-mesh-4slot \
RSMOL_5_10X2_5_MESH_4SLOT_OUTPUT_DIR="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh_4slot/formal_1epoch_bf16_$(date +%Y%m%d_%H%M%S%N)" \
bash run_stage4_5_10x2_5_mesh_4slot_formal_3090.sh
```

## 中断恢复

checkpoint 保存模型/config/tokenizer、AdamW 状态、scheduler 步数/合同、
全局步数、每 rank 的 Parquet 游标、manifest 和 Python/Torch CPU/CUDA RNG。
恢复时实际跳过已消费行；校验 rank 分片、游标边界、manifest、seed、精度、
batch、LR、9,244-step horizon 和 463-step warmup。

只允许从本四槽 BF16 路线相同训练合同及完整 precision policy 的 formal checkpoint 续训。
五/六/七槽、其他精度或短 smoke checkpoint 均不能作为正式恢复来源。
fresh 初始化还会校验四槽 conversion provenance、继承权重一致性与十二个新 router tensor。
fresh formal wrapper 会清空通用 `RESUME_FROM`，避免继承先前诊断恢复路径；
正式恢复须显式指定 `FORMAL_RESUME_FROM`，同时使用新的空输出目录：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
RSMOL_5_10X2_5_MESH_4SLOT_FORMAL_RESUME_FROM=/absolute/path/to/formal/checkpoint-005000 \
RSMOL_5_10X2_5_MESH_4SLOT_OUTPUT_DIR="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh_4slot/resume_$(date +%Y%m%d_%H%M%S%N)" \
bash run_stage4_5_10x2_5_mesh_4slot_formal_3090.sh
```

恢复后目标仍是累计第 9,244 步，不是额外再训练一轮。

## 可选短验证

保留与母本相同的 Gate D：0→10、Gate E：10→12 诊断入口。
短验证采用独立 10-step scheduler/1-step warmup；正式训练从转换模型重新开始。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_stage4_5_10x2_5_mesh_4slot_smoke_3090.sh

# 将路径替换为上一个 smoke 的实际输出目录。
RSMOL_5_10X2_5_MESH_4SLOT_RESUME_FROM=/absolute/path/to/smoke/checkpoint-000010 \
bash run_stage4_5_10x2_5_mesh_4slot_resume_3090.sh
```

## 文件与验证状态

- 模型：`../recursive_model_5_10x2_5_mesh_4slot.py`
- 转换：`../scripts/convert_stepwise_5_10x2_5_mesh_4slot.{py,sh}`
- 训练：`../scripts/train_stage4_5_10x2_5_mesh_4slot_ddp.{py,sh}`
- 通用提交：`../run_stage4_5_10x2_5_mesh_4slot_3090.sh`
- 正式/短验证/短恢复：`../run_stage4_5_10x2_5_mesh_4slot_{formal,smoke,resume}_3090.sh`
- 测试：仓库根目录 `tests/test_text_5_10x2_5_mesh_4slot_*.py`
- 本地统一验证：仓库根目录 `tests/validate_text_mesh_4slot_bf16.py`

本地依赖轻量检查覆盖调度、转换来源/词表 tied alias、数据 cursor、恢复合同、
各 rank RNG、与母本的核心 AST 等价和真实 Bash mock 提交/参数转发。
本地没有 PyTorch、Transformers、真实权重和数据，不据此声明远程转换或 GPU 训练 PASS。
本地统一验证命令（四槽与原六槽回归）为：

```bash
python -B tests/validate_text_mesh_4slot_bf16.py
```

当前 50 项实际检查通过：四槽 27 项、六槽母本回归 23 项；8 项真实 tensor 检查
因本地缺依赖跳过，其中四槽 5 项、六槽 3 项。另核对 8 个 Python 文件语法、
6 个四槽 Bash 文件语法/LF 和 14 个母本原文件 SHA；未执行远程生产转换或真实 GPU 训练。
报告为仓库根目录 `audit_reports/text_mesh_4slot_bf16_validation.json`。
`--require-tensors` 会将任何 tensor skip（含 CUDA）视为失败；需要完整 ML 环境与 BF16 GPU。
Windows 沙箱首次运行遇测试临时目录访问限制，实际通过结果来自沙箱外同一验证流程。

可在远程登录节点的 `rsmol` 中手动运行小模型 CPU tensor 检查，无生产权重，禁用 GPU：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/rsmol
CUDA_VISIBLE_DEVICES="" python -B -m unittest discover -s tests -p 'test_text_5_10x2_5_mesh_4slot_tensor.py'
```

CPU 检查验证真实转换/保存重载、四槽初态与 30 层轨迹、两次循环/router 梯度、
CPU BF16 autocast 下 query/write 来源和 FP32 memory/AdamW 状态，以及恢复后下一次更新一致性。
另有 CUDA BF16 检查，CPU-only 命令会明确 skip；需要时在 `vc submit` 分配的计算节点
手动运行同一测试且不清空 `CUDA_VISIBLE_DEVICES`。检查不自动插入正式提交。
正式完成仍以远程 `checkpoint-009244` 和 `stage4_gate_FORMAL_audit.json` 为准。
