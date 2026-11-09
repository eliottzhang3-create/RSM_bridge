# Mizar 最终文本权重 → Huginn 类 5＋10×2＋5，GBS1024 单 epoch 训练

本地 Windows 仅编辑核心代码，经 GitHub 同步到远程 Linux。全部模型、数据和训练输出留在
`/hpc_stor03/sjtu_home/jinwei.zhang`；CPU 转换在登录节点执行，八卡训练通过 `vc submit`。
原 5＋10×2＋5、5＋20×2＋5 和音频路线均保持原样。

## 权重与结构

源根目录为 `models/Mizar159M`，实际使用
`checkpoints/final/stage3_q4_seed20260905.ckpt` 中的 `caption_decoder.lm.*`。
转换器按既有 release manifest 核验最终 checkpoint 大小、SHA256、完整 tensor inventory，
以及文本 config/tokenizer 文件大小和 SHA256。`assets/smollm2-135m/model.safetensors`
不参与权重加载；文本 assets 仅提供配置、tokenizer 和 generation config。
不构造音频编码器、mapper 或音频模型。

原 30 层按以下一基索引继承，不平均、不重新初始化所选权重：

```
前五层： 1 2 3 4 5
中十层： 6 8 10 12 14 16 18 20 22 24
后五层： 26 27 28 29 30
```

20 个物理模块，执行 `前5 → 中10 → 中10 → 后5`，总共 30 次；两次中间执行共享参数。
新增且只新增一个共享 `Linear(1152,576,bias=False)`，共 663,552 参数：

```text
E  = Prelude(Embedding(tokens))
S0 = TruncNormal(mean=0, std=sqrt(2/5), bounds=±3std), shape=[B,L,576]
S1 = Core(A(cat([S0,E], dim=-1)))
S2 = Core(A(cat([S1,E], dim=-1)))
logits = LMHead(FinalNorm(Coda(S2)))
```

同一个 A 调用两次，拼接顺序均为 `[state, prelude]`。S0 每次完整 forward 只采样一次，
各样本/token/channel 独立，第二次循环不重新采样。E 与 S1 均不 detach，两个循环完整反传，
全部继承参数和投影参数参与同一 AdamW 组训练。S0 是临时激活，不是可训练参数或跨 forward 的持久状态。
高斯底层标准差为 0.6324555320336759，截断界限为 ±1.8973665961010275；截断后实际方差略小。
只对新增 A 使用 σ=√(2/(5×576))≈0.0263523、±3σ 截断正态，CPU seed0 独立 RNG 上下文初始化，
不改变继承 tensor 或调用者 RNG。BF16 autocast 后的投影结果转回原 residual dtype，保持 FP32 residual。

参考 [Huginn 论文](https://arxiv.org/html/2502.05171v2) 与
[官方模型实现](https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/raven_modeling_minimal.py)。
本路线在预训练 Mizar 上引入其共享状态投影接口；沿用 Mizar/Llama 的 norm/attention/RoPE，
固定两轮、完整反传，不引入原工作中的可变深度、sandwich norm 或截断 BPTT。

使用独立 `recursive_model_mizar_text_5_10x2_5_huginn.MizarHuginnForCausalLM` 显式加载，KV cache 仍为 30 个逻辑槽，
训练关闭 cache；不调用全局 AutoModel 注册。全量文本参数训练，无 MeSH/router/LoRA。
继承 embedding、final norm、lm_head、权重绑定、tokenizer、RoPE 和 attention 配置。

转换在 CPU 离线运行，默认 8 线程；严格加载最终 LM、检查绑定 head 的原始值一致、
逐项核对选中 tensor 和唯一新增投影、hook 验证 30 次执行及两次共享 A，
再保存并重载验证全部 tensor（含 A）、绑定、固定同一噪声实现的 logits 和 tokenizer。
模型 eval 默认仍然随机；开发测试可传显式 `initial_state` 固定噪声，缓存/非缓存比较须对齐各 token 的 S0。
原完整 30 层与抽层循环后的 logits 不作相等声明。只有全部检查成功才发布完整目录；
拒绝覆盖已存在输出和写入源码/源模型目录，失败 staging 保留在输出同级。

## 单次训练合同

每次提交只训练一个项目约定的 epoch：`8 ranks × micro8 × GA16 = GBS1024`，
`9244 updates / scheduler domain 9244 / warmup 463`。
名义样本额度为 `9244×1024=9,465,856`，不声明恰好遍历所有有效文本一次。
每个 epoch 都从 step0 重新线性 warmup（min_lr→max_lr），再 cosine 回 min_lr。

| 手动提交 | max_lr | min_lr | 初始化 |
|---|---:|---:|---|
| epoch 1 | 8e-4 | 8e-5 | 本路线转换结果 |
| epoch 2 | 2e-4 | 2e-5 | epoch 1 的完整 `checkpoint-step-009244`，仅模型权重 |

`--max-lr`、`--min-lr` 可显式传入；省略 min_lr 时取 `0.1×max_lr`，保持原比例。
`--model-dir` 是新 epoch 的权重初始化：优化器、scheduler、RNG、计数和数据游标全部重新开始，
不读取 `training_state.pt`。所有权重包含训练后的 A，都严格重载，不重新初始化 A。
从训练结果开始新 epoch 时要求本路线完整 step9244；
epoch 2 再次使用 seed0、同样的原始数据流。

沿用原纯文本 5＋10×2＋5 的数据和更新方法：

- FP32 参数及 AdamW 状态、仅 forward 使用 BF16、FP32 CE，无 GradScaler/TF32。
- 单组 AdamW（包括 norm），betas=(0.9,0.95)、eps=1e-8、WD=0.1、amsgrad=false、clip=1。
- context1024、动态 padding、不 packing、`add_special_tokens=False`、不额外补 EOS。
- 当前 data-dir 或其 `data/` 下发现 85 个 shard；seed0 一次 shuffle，再连续切给 8 rank，
  前五 rank 各 11 片、后三 rank 各 10 片。不是较新 MeSH 路线的 index%8 分配。
- 每片 bounded shuffle buffer4096，片尾不足 micro8 丢弃；workers=0。
- loss/梯度按跨 8 ranks 和 16 次累积的全局有效 token 数归一化，无额外 GA 除法。
- 每 500 步及最终保存；保留最后三个完整 checkpoint：8500、9000、9244。

正式入口直接提交 FORMAL，无 CPU 数据集扫描、Gate B 或 smoke 前置要求。
容器、rsmol 环境及资源沿用原入口：pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node。
专用 `RSMOL_MIZAR_TEXT_5_10X2_5_HUGINN_*` 环境变量、输出、日志和 job 名与旧路线隔离。
转换输出环境变量为 `..._CONVERT_OUTPUT_DIR`，训练输出为 `..._OUTPUT_DIR`。

## 远程执行与提交

先完成本地 Git 提交、推送，再在远程登录节点同步并执行一次转换：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/scripts/convert_mizar_text_5_10x2_5_huginn.sh
```

默认转换输出为 `models/Mizar159M-text-5-10x2-5-huginn`。如需其他路径：

```bash
bash code/RSmol/scripts/convert_mizar_text_5_10x2_5_huginn.sh \
  --source-root /hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-text-5-10x2-5-huginn
```

转换成功后，提交 epoch 1。下例使用固定输出名，必须是新的空外部目录；重复实验请换名字。
`--output-dir` 即实际 checkpoint 父目录，脚本不会再自动追加 `formal/`：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN1=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_huginn_gbs1024/epoch1
bash code/RSmol/run_stage4_mizar_text_5_10x2_5_huginn_training_5090.sh \
  --model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-text-5-10x2-5-huginn \
  --output-dir "$RUN1" --max-lr 8e-4 --min-lr 8e-5
```

由用户确认 epoch 1 已完成、生成完整 step9244 后，手动执行 epoch 2；不同 shell 会话需重设 RUN1：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN1=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_huginn_gbs1024/epoch1
RUN2=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_huginn_gbs1024/epoch2
bash code/RSmol/run_stage4_mizar_text_5_10x2_5_huginn_training_5090.sh \
  --model-dir "$RUN1/checkpoint-step-009244" \
  --output-dir "$RUN2" --max-lr 2e-4 --min-lr 2e-5
```

默认数据目录为 `data/SmolLM2-135M-10Bsubset`，可加 `--data-dir PATH`。
省略 output-dir 会使用隔离的 `formal_<timestamp>_<pid>/formal`。
日志位于远程 `code/RSmol/log/mizar_text_5_10x2_5_huginn_gbs1024/`，每 10 步显示实际 loss/lr/速度。
加 `--dry-run` 只输出配方报告，不读模型/数据、不初始化 CUDA、不提交 vc。

同一个 epoch 中断后的完整恢复使用 `--resume-from`，LR 必须与该 checkpoint 一致，
输出使用新空目录，累计训练到本 epoch 的 step9244：

```bash
bash code/RSmol/run_stage4_mizar_text_5_10x2_5_huginn_training_5090.sh \
  --resume-from "$RUN1/checkpoint-step-009000" \
  --output-dir "${RUN1}_resume" --max-lr 8e-4 --min-lr 8e-5
```

模型 Torch CPU/CUDA 噪声流使用 `seed+1009×rank`，八 rank 独立；结构审计保留 RNG，避免消耗训练噪声。
完整 checkpoint 收集并保存每个 rank 的 Python/NumPy/Torch CPU/当前 CUDA RNG，恢复时按 rank 选择。
同时保存八 rank 数据游标与 bounded-shuffle RNG，但仍不保存 buffer 内容；
因此不保证数据逐样本连续或训练 bitwise exact，不把 RNG 恢复等同于数据流的精确恢复。
跨 epoch 请使用上面的 `--model-dir` 命令。

## 本地验证与提交代码

```powershell
cd C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
python -B tests/validate_mizar_text_5_10x2_5_huginn.py
```

报告：`audit_reports/mizar_text_5_10x2_5_huginn_validation.json`。
检查母本 SHA、原更新/读取/损失 AST、全 9245 点 LR、初始化合同及实际 Bash 模拟提交。
真实 CPU 张量检查覆盖共享 A、唯一采样、完整反传、分布、固定噪声数值、BF16 residual、
训练后 A 重载、各 rank RNG 与审计保护；缺少 torch/transformers/safetensors 时会明确跳过。
这不代表远程生产权重转换、八卡训练或恢复已经验证。
需要时可在远程 rsmol 环境运行以下合成 CPU 检查，它不是训练前置步骤：

```bash
python -B tests/validate_mizar_text_5_10x2_5_huginn.py --require-tensors
```

仅提交本次交付文件（下面命令在 PowerShell 和 Bash 都可执行）。工作区存在另一条 LoRA 路线改动，
不要使用 `git add .`。`RM.md` 按块仅选 Huginn 记录；若与其他新增记录合并在同一块，用 `e` 编辑选择：

```text
git add -- code/RSmol/mizar_text_5_10x2_5_huginn code/RSmol/recursive_model_mizar_text_5_10x2_5_huginn.py code/RSmol/scripts/convert_mizar_text_5_10x2_5_huginn.py code/RSmol/scripts/convert_mizar_text_5_10x2_5_huginn.sh code/RSmol/scripts/train_stage4_mizar_text_5_10x2_5_huginn_ddp.py code/RSmol/scripts/train_stage4_mizar_text_5_10x2_5_huginn_ddp.sh code/RSmol/run_stage4_mizar_text_5_10x2_5_huginn_training_5090.sh tests/test_mizar_text_5_10x2_5_huginn.py tests/test_mizar_text_5_10x2_5_huginn_shell.py tests/test_mizar_text_5_10x2_5_huginn_tensors.py tests/validate_mizar_text_5_10x2_5_huginn.py audit_reports/mizar_text_5_10x2_5_huginn_validation.json
git add -p -- RM.md
git diff --cached --stat
git commit -m "Add isolated Huginn-style Mizar conversion and epoch training"
git push
```

未自动 commit/push，未访问远程生产资产，也未提交真实训练作业。
