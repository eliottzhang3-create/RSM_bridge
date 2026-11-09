# 纯文本 SmolLM2：5＋20×2＋5，GBS512

本地 Windows 只维护核心代码，经 GitHub 同步到远程 Linux。原始权重、转换产物、85 片文本数据和训练输出均留在服务器。本路线独立复制原 `run_stage4_5_10_5_5090.sh` 对应的模型/训练实现；原 `run_stage4_5_10_5_training_5090.sh` 只是该脚本的兼容别名。原路线文件的转换前 SHA256 保存在 `source_guard.json`，本次不修改这些母本。

## 架构和继承

从 `/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2` 的原始 30 层 Llama/SmolLM2 初始化。保留源层 0..29，一一对应 30 个物理模块；执行顺序严格为：

```text
prefix:  0..4
middle:  5..24
middle:  5..24  （复用同一批参数对象）
suffix: 25..29
```

共 50 次层执行。配置 `num_hidden_layers=50` 表示逻辑执行深度，`recursive_layer_count=30` 表示实际存储层数。模型保存原 embedding、final norm、lm_head、权重绑定、RoPE/attention 配置及 tokenizer；全模型训练，无 MeSH/router/memory/LoRA/音频模块。KV cache 分别使用 50 个逻辑槽，中间两次执行不共享 KV 状态。

转换器使用显式 `LlamaForCausalLM` 读取源模型、显式 `RecursiveLlama5_20x2_5ForCausalLM` 创建目标，训练也显式加载新类，不修改全局 AutoModel 注册。独立 Python 模型实现位于 `../recursive_model_5_20x2_5.py`，加载已转换/已训练模型时必须使用此类，不能用普通 50 层 Llama 替代。

转换在 CPU 离线运行，默认 8 线程。拒绝递归/适配/带训练状态的源模型以及已经存在的目标目录，不提供覆盖选项。逐 tensor 检查键、形状、dtype 和数值；验证临时单遍 30 层前向与原模型一致、实际 50 次执行顺序正确、logits 有限，以及保存重载后的权重/绑定/logits/tokenizer 一致。通过后发布临时同级目录；失败的 staging 和错误 JSON 保留供排查。生产转换结果以远程实际执行产生的 `recursive_conversion_metadata.json` 为准。

## 固定正式配方

| 项目 | 本路线 |
|---|---|
| GPU / micro batch / GA | 8 / 4 / 16 |
| 全局 batch | 512（原 1024 的一半） |
| 更新数 / scheduler domain | 18,488 / 18,488 |
| warmup | 925（向上取整的 5%） |
| 样本预算 | 9,465,856 = 18,488×512 = 9,244×1,024 |
| 每 rank 样本 / microbatch 数 | 1,183,232 / 295,808 |
| 最大 / 最小 LR | 8e-4 / 8e-5 |
| 参数 / 前向 / CE | FP32 / CUDA BF16 autocast / FP32 |
| context | 1024，动态 padding，不拼接文档 |
| AdamW | 单参数组；betas=(0.9,0.95)，eps=1e-8，WD=0.1，amsgrad=false |
| 全局梯度裁剪 | 1.0 |
| 保存 / 保留 | 每 500 次更新及最后一次；最新 3 个完整 checkpoint |
| 数据读取 | 85 shards，seed=0，buffer=4096，workers=0，pin/persistent=false |
| 环境 / 资源 | rsmol；pdgpu-5090，32CPU/256G/8GPU/1node |
| 容器 | docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 |

沿用原线性 warmup＋cosine 公式：step0 为最小 LR，step925 到最大 LR，step18488 回到最小 LR。正式入口直接选择 `FORMAL`，不自动提交 Gate A/D/E 或额外 GPU smoke。

数据分片、片内 bounded shuffle、tokenize、padding、有效 token 计数、DDP 梯度平均、更新及保存循环沿用母本。每次更新先取得全部 16 个 microbatch 的跨卡有效 token 总数，再以 `world_size / global_window_tokens` 缩放各自 CE sum 后 backward；没有额外 GA 除法，不使用音频训练的 microbatch-mean loss。每片末尾不足一个 microbatch 的文档仍丢弃；micro 从 8 改为 4 后片尾丢弃数可能改变。因此这里保持的是原样本预算与读取规则，不承诺两条路线的逐样本位置或实际 token 数完全相同。

## 远程执行：先转换，再正式训练

将本地代码通过 GitHub 同步到远程后，在 Linux 登录节点运行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol

# CPU 转换；原模型和输出目录都有独立默认值。
bash scripts/convert_stepwise_5_20x2_5.sh

# 直接提交一个正式八卡作业，不再先运行 CPU 数据检查。
bash run_stage4_5_20x2_5_training_5090.sh
```

转换输出默认：`/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-20x2-5`。

训练默认输出：`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_5_20x2_5_gbs512/formal_<timestamp>_<pid>/formal/`。

正式日志：远程代码目录的 `log/text_5_20x2_5_gbs512/formal_<timestamp>_<pid>.JOB.log`。提交前打印具体 OUTPUT_DIR；训练中的 `stage4_progress.jsonl`、`stage4_lifecycle_rank*.jsonl`、监控和报告保存在该输出目录。

正式入口按原文本提交方式，直接调用 `vc submit`；不再运行 `prepare_text_5_20x2_5.py` 或自动 Gate B，不扫描三个 shard 的内容、不预先分词。训练仍使用母本的 `discover_parquet_files` 和 `assign_shards`，从当前数据目录（支持其 `data/` 子目录）发现 85 个 shard，按 seed0 分给八卡，然后由原 streaming loader 读取训练。

已有转换模型可直接复用，不必重新转换。用户本次失败发生在 `forward_trace_done` 之后、第一步训练之前；该失败 run 没有可恢复的训练更新，重新提交到默认新输出目录即可。

`--audit-report` 仅为旧命令保留，可选附带已有 Gate B 报告：

```bash
bash run_stage4_5_20x2_5_training_5090.sh \
  --audit-report /absolute/path/to/stage4_gate_B_audit.json
```

历史报告仅写入 `dataset_audit.historical_report`，不参与当前路径、tokenizer 文件或容量匹配，也不提供训练 shard 路径。报告缺失或无法解析会打印说明后继续；未提供则完全不读取报告。当前数据分配始终来自当前训练数据目录，而非旧报告内的登录节点/挂载路径。报告如实记录 `status=NOT_RUN`、`dataset_audit_performed=false`，不会将历史检查冒充本次审计。

路径选项：`--model-dir`、`--data-dir`、`--output-dir`、`--audit-report`、`--resume-from`。对应环境变量前缀统一为 `RSMOL_5_20X2_5_`，不读取旧 `RSMOL_STAGE4_*` / `RSMOL_5_10_5_*` 配方。输出须为新的空外部目录。短名 `run_stage4_5_20x2_5_5090.sh` 转发到同一个正式入口。

`--dry-run` 只检查/写出固定正式配置报告，不读取权重/数据、不审计、不提交作业。转换仅需要显式路径时可用 `--source-checkpoint` / `--output-dir` 覆盖默认路径。

## 保存和恢复

本 run 最终产物为 `formal/checkpoint-step-018488/`，包含模型、config、tokenizer、`training_state.pt` 和 `checkpoint_complete.json`。正常完成后保留 step17500、18000、18488；`latest_complete_checkpoint.json` 指向最近完整保存。

恢复例子（路径替换为实际 run 与 checkpoint，输出另选新目录）：

```bash
bash run_stage4_5_20x2_5_training_5090.sh \
  --resume-from /absolute/old-run/formal/checkpoint-step-018000 \
  --output-dir /absolute/new-resume-run/formal
```

使用本路线 FORMAL checkpoint 的模型、优化器、scheduler、累计计数、manifest 和各 rank cursor，继续到总目标 18,488，已完成部分不会再加一个完整训练周期。恢复前检查架构/路线、8×4×16、LR/AdamW、seed、context、shuffle buffer、数据路径、原 scheduler domain 和 warmup。旧 5＋10×2＋5、MeSH、普通 SmolLM2、Gate D/E checkpoint 均不能作为本路线完整恢复来源。

恢复语义继承原文本训练的限制：每卡数据 cursor 和 bounded-shuffle RNG 有保存，但没有保存 shuffle buffer 内容；仅保存 rank0 的 Python/Torch/CUDA RNG，并非八卡各自独立 RNG 快照。恢复不保证 bitwise exact，也不承诺严格连续的逐样本顺序。报告保留 `bounded_shuffle_bitwise_exact=false`。

## 本地验证范围

```bash
python -B tests/validate_text_5_20x2_5.py
# 远程具备 Torch/Transformers 的环境中可手动要求真实 CPU tensor 用例不可跳过：
python -B tests/validate_text_5_20x2_5.py --require-tensors
```

验证包含母本 SHA、完整更新主体/读取/损失/attention/cache 算法 AST 对照、实际 schedule/cache-slot helper、全部 LR 位置、正式配方/跨路线恢复拒绝、模拟 Parquet 的真实 micro4 迭代，以及真实 Bash＋假 vc/conda/torchrun 的转换/直接正式/恢复流程。针对本次故障，实际运行不带报告和带不同挂载路径旧报告的 FORMAL 启动控制流，替换模型/GPU执行后确认从 `forward_trace_done` 进入 `run_training`；另检查八 rank manifest 广播、旧报告缺失/损坏不阻断、默认无 CPU preflight/扫描/分词，以及 85 片分配与母本一致。可选真实张量用例覆盖源模型单遍一致性、两轮共享梯度求和、50 槽 cache、绑定/非绑定保存重载、完整转换、16 次累积的 token 加权梯度及 checkpoint optimizer/scheduler/rank cursor 恢复。

本地 Windows 当前缺 Torch/Transformers，真实张量检查会明确跳过；结果写入仓库 `audit_reports/text_5_20x2_5_validation.json`。本地检查不代表远程生产转换、八卡训练或八卡恢复已通过；这些须由实际资产和远程作业给出证据。
