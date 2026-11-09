# Mizar 文本 5＋10×2＋5：中间两轮独立 LoRA，rank8 / alpha16

在原 `run_stage4_mizar_text_5_10x2_5_training_5090.sh` 的基础上复制隔离。
本地 Windows 维护核心代码，经 GitHub 同步；模型、数据、转换结果和训练输出都在远程 Linux。
原模型、转换器、trainer、提交脚本保持不变，18 份参考文件的归一化 LF SHA256
保存在本目录的 `source_guard.json`，验证器会核对。

## 实现与模型来源

| 用途 | 隔离实现（仓库相对路径） |
|---|---|
| 模型与逻辑缓存 | `code/RSmol/recursive_model_mizar_text_5_10x2_5_loop_lora_r8.py` |
| 两轮 LoRA | `code/RSmol/mizar_text_5_10x2_5_loop_lora_r8/loop_lora.py` |
| 架构、来源、初始化合同 | 本目录 `contract.py` |
| 转换、继承和重载检查 | 本目录 `conversion.py`、`publication.py` |
| CPU 转换入口 | `code/RSmol/scripts/convert_mizar_text_5_10x2_5_loop_lora_r8.sh` |
| DDP trainer | `code/RSmol/scripts/train_stage4_mizar_text_5_10x2_5_loop_lora_r8_ddp.py` |
| 作业内 torchrun 入口 | 同名 `.sh` |
| 5090 正式提交入口 | `code/RSmol/run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh` |

转换源为 `/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M` 下的
`checkpoints/final/stage3_q4_seed20260905.ckpt`，提取 `caption_decoder.lm.*`。
读取既有 `mizar_official_eval/release_manifest.json`，核验最终 checkpoint 和文本 assets 的
大小、SHA256、完整 tensor inventory。文本 assets 提供 config/tokenizer/generation config；
权重来自最终 Mizar checkpoint。不构造音频编码器或 mapper。

选层沿用原方案，以下为**一基**源索引：

```text
前五层： 1 2 3 4 5
中十层： 6 8 10 12 14 16 18 20 22 24
后五层： 26 27 28 29 30
```

20 个物理层，执行 `前5 → 中10（LoRA 0）→ 中10（LoRA 1）→ 后5`，总计30次。
中间物理层 5..14（零基）的基础参数在两轮共享；每层的每个投影各有两套独立 A/B：

```text
self_attn.q_proj / k_proj / v_proj / o_proj
mlp.gate_proj / up_proj / down_proj
y = Linear(x, W, bias) + (16/8) * Linear(Linear(x, A[loop]), B[loop])
```

rank8、alpha16、scale2、dropout0；A 为 Kaiming 初始化，B 为零。
共 `10层 × 7投影 × 2轮 = 140` 对 A/B、280个新增 tensor、1,628,160个新增参数。
各层、投影、轮次独立；前后五层没有 LoRA。基础参数和 LoRA 均训练，使用相同 LR、相同 WD。
初始化在 CPU RNG fork 中进行，不改变原模型的基础初始化和外部 RNG 状态。
原 W/bias Parameter 对象及 state_dict 名称保留，新增键如：
`model.layers.5.self_attn.q_proj.lora_A.0`、`lora_B.1`。

使用专用 `MizarTextLoopLoRAR8ForCausalLM` 显式加载；不注册全局 AutoModel。
保留原 attention、RoPE、embedding/head 绑定、30个逻辑 KV cache 槽；训练关闭 cache。
每次调用中间层时选择当前轮的 A/B，退出调用后清除选择；autograd 使用该次前向实际选中的 tensor。
继承原训练配置，关闭 decoder gradient checkpointing；模型会明确拒绝启用该功能。

转换在 CPU 离线执行、默认8线程、拒绝覆盖已有输出、失败保留 staging。
逐项验证继承权重、20层/30次 trace、LoRA 范围和参数数、全参数可训练，
并与使用相同继承权重的原循环模型逐值比较零 B 的 logits；再保存并重载检查全部 tensor
（包含 A/B）、logits、tokenizer。抽层循环模型与原完整30层模型不作 logits 相等声明。

## 训练配方和第二个 epoch

每次提交仅运行一个项目约定的 epoch：8 ranks × micro8 × GA16 = GBS1024，
9244 updates、scheduler domain9244、warmup463。名义样本额度为9,465,856；
不保证严格遍历全部有效文本恰好一次。两次提交的学习率分别如下：

| 提交 | max_lr | min_lr | 初始化 |
|---|---:|---:|---|
| epoch1 | 8e-4 | 8e-5 | 本路线转换结果 |
| epoch2 | 2e-4 | 2e-5 | epoch1 完整 `checkpoint-step-009244` 的全部模型权重 |

epoch2 由用户手动提交，`--model-dir` 会完整加载已训练的基础权重及两套 LoRA，
不读取 `training_state.pt`；optimizer、scheduler、RNG、计数和数据游标重新开始，seed0从头读取。
每次独立执行 min→max 线性 warmup，再 cosine 回 min；不合并为18488步 schedule。
省略 `--min-lr` 时取 `0.1×max_lr`。

其余配方保留原文本训练实现：

- FP32参数与 AdamW 状态，BF16仅 forward、FP32 CE，无 GradScaler/TF32。
- 单组 AdamW，基础权重、norm、A/B 均 WD0.1，betas=(0.9,0.95)、eps1e-8、clip1。
- context1024、动态 padding、不 packing、`add_special_tokens=False`、不额外补 EOS。
- 85个 shard 按原 seed0 shuffle 后连续切给8 rank，前五 rank各11片、后三各10片。
- 片内 bounded shuffle buffer4096，片尾不足 micro8 丢弃，workers0。
- 按全部 rank和累积窗口的全局有效 token 数归一化，无额外 GA 除法。
- 每500步及最终保存；最终保留8500、9000、9244三个完整 checkpoint，包含全部 A/B。

正式入口直接 `vc submit`，无新增 CPU 数据扫描、Gate B 或 smoke 前置门槛。
沿用 pdgpu-5090、32CPU、256G、8GPU、1node、原镜像和 rsmol 环境。
模型路径、输出、日志、job 和 `RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_*` 环境变量独立。
转换使用 `..._CONVERT_OUTPUT_DIR`，训练使用 `..._OUTPUT_DIR`。

## 远程执行命令

本地提交推送后，在远程登录节点同步并进行一次 CPU 转换：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/scripts/convert_mizar_text_5_10x2_5_loop_lora_r8.sh
```

默认生成 `/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-text-5-10x2-5-loop-lora-r8`。
可用转换参数 `--source-root PATH --output-dir NEW_PATH` 指定其他目录，已有输出会拒绝覆盖。

提交 epoch1。下例使用固定输出名，重复实验需换成新的空外部目录：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN1=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_loop_lora_r8_gbs1024/epoch1
bash code/RSmol/run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh \
  --model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-text-5-10x2-5-loop-lora-r8 \
  --output-dir "$RUN1" --max-lr 8e-4 --min-lr 8e-5
```

epoch1 完成后，用户另行手动提交 epoch2；下列命令可用于新的 shell 会话：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN1=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_loop_lora_r8_gbs1024/epoch1
RUN2=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_loop_lora_r8_gbs1024/epoch2
bash code/RSmol/run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh \
  --model-dir "$RUN1/checkpoint-step-009244" \
  --output-dir "$RUN2" --max-lr 2e-4 --min-lr 2e-5
```

默认数据为 `/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset`，可加 `--data-dir PATH`。
`--output-dir` 是实际 checkpoint 父目录，不会再追加 formal/。
省略时使用本路线 `formal_<timestamp>_<pid>/formal`。
日志在远程 `code/RSmol/log/mizar_text_5_10x2_5_loop_lora_r8_gbs1024/`，每10步输出 loss/lr/速度。
加 `--dry-run` 仅生成配方报告，不加载模型/数据或提交GPU作业。

同一 epoch 中断后用 `--resume-from` 恢复完整状态、保持原 LR，输出使用新空目录：

```bash
bash code/RSmol/run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh \
  --resume-from "$RUN1/checkpoint-step-009000" \
  --output-dir "${RUN1}_resume" --max-lr 8e-4 --min-lr 8e-5
```

同 epoch 恢复包含模型、optimizer、scheduler、累计计数、manifest 和八 rank 数据游标，累计到9244步。
保留母本限制：bounded shuffle buffer 内容未保存；模型 Python/Torch/CUDA RNG 仅 rank0 快照。
不保证逐样本连续、八 rank 独立 RNG 恢复或 bitwise exact。跨 epoch 使用 `--model-dir`。
旧路线 checkpoint 会因路线/LoRA 合同或缺失 A/B 被拒绝。

## 验证和 Git 提交命令

本地执行：

```powershell
cd C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
python -B tests/validate_mizar_text_5_10x2_5_loop_lora_r8.py
```

报告：`audit_reports/mizar_text_5_10x2_5_loop_lora_r8_validation.json`。
本机34项通过：母本SHA、完整训练更新和读取AST、两组各9245点LR、
epoch2实际启动路径不反序列化训练状态、LoRA NumPy前向/有限差分/分支清理、
真实 Bash vc→torchrun 参数/资源/转义/失败传递检查。
10项真实 CPU tensor 检查因本机缺 torch/transformers/safetensors 跳过；
包括真实任务梯度、权重继承及初始化 RNG、非零 A/B 完整保存重载、缺失 A/B 拒绝、
30槽 cache，以及 checkpoint 中 optimizer moments/RNG/指定 rank游标的恢复。
NumPy检查不代表 PyTorch autograd 已验证；远程转换、八卡训练和八rank恢复均未运行。

远程 rsmol 环境可选执行下列合成 CPU 检查，不需要生产模型/数据，也不是训练前置门槛：

```bash
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/rsmol
python -B tests/validate_mizar_text_5_10x2_5_loop_lora_r8.py --require-tensors
```

本地仅提交本次隔离实现和记录，以下命令适用于 PowerShell，未由助手自动执行：

```powershell
cd C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
git add -- RM.md code/RSmol/mizar_text_5_10x2_5_loop_lora_r8 code/RSmol/recursive_model_mizar_text_5_10x2_5_loop_lora_r8.py code/RSmol/scripts/convert_mizar_text_5_10x2_5_loop_lora_r8.py code/RSmol/scripts/convert_mizar_text_5_10x2_5_loop_lora_r8.sh code/RSmol/scripts/train_stage4_mizar_text_5_10x2_5_loop_lora_r8_ddp.py code/RSmol/scripts/train_stage4_mizar_text_5_10x2_5_loop_lora_r8_ddp.sh code/RSmol/run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh tests/test_mizar_text_5_10x2_5_loop_lora_r8.py tests/test_mizar_text_5_10x2_5_loop_lora_r8_numpy.py tests/test_mizar_text_5_10x2_5_loop_lora_r8_shell.py tests/test_mizar_text_5_10x2_5_loop_lora_r8_tensors.py tests/validate_mizar_text_5_10x2_5_loop_lora_r8.py audit_reports/mizar_text_5_10x2_5_loop_lora_r8_validation.json
git diff --cached --stat
git commit -m "Add isolated Mizar text training with per-loop rank-8 LoRA"
git push
```
