# Mizar 最终文本权重 → 5＋10×2＋5，GBS1024 单 epoch 训练

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
使用原 `recursive_model_5_10_5.RecursiveLlamaForCausalLM` 显式加载，KV cache 仍为 30 个逻辑槽，
训练关闭 cache；不调用全局 AutoModel 注册。全量文本参数训练，无 MeSH/router/LoRA。
继承 embedding、final norm、lm_head、权重绑定、tokenizer、RoPE 和 attention 配置。

转换在 CPU 离线运行，默认 8 线程；严格加载最终 LM、检查绑定 head 的原始值一致、
逐项核对选中 tensor、hook 验证 30 次执行，再保存并重载验证 tensor、绑定、logits 和 tokenizer。
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
不读取 `training_state.pt`。从训练结果开始新 epoch 时要求本路线完整 step9244；
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
专用 `RSMOL_MIZAR_TEXT_5_10X2_5_*` 环境变量、输出、日志和 job 名与旧路线隔离。
转换输出环境变量为 `..._CONVERT_OUTPUT_DIR`，训练输出为 `..._OUTPUT_DIR`。

## 远程执行与提交

先完成本地 Git 提交、推送，再在远程登录节点同步并执行一次转换：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/scripts/convert_mizar_text_5_10x2_5.sh
```

默认转换输出为 `models/Mizar159M-text-5-10x2-5`。如需其他路径：

```bash
bash code/RSmol/scripts/convert_mizar_text_5_10x2_5.sh \
  --source-root /hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-text-5-10x2-5
```

转换成功后，提交 epoch 1。下例使用固定输出名，必须是新的空外部目录；重复实验请换名字。
`--output-dir` 即实际 checkpoint 父目录，脚本不会再自动追加 `formal/`：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN1=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_gbs1024/epoch1
bash code/RSmol/run_stage4_mizar_text_5_10x2_5_training_5090.sh \
  --model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-text-5-10x2-5 \
  --output-dir "$RUN1" --max-lr 8e-4 --min-lr 8e-5
```

由用户确认 epoch 1 已完成、生成完整 step9244 后，手动执行 epoch 2；不同 shell 会话需重设 RUN1：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN1=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_gbs1024/epoch1
RUN2=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_gbs1024/epoch2
bash code/RSmol/run_stage4_mizar_text_5_10x2_5_training_5090.sh \
  --model-dir "$RUN1/checkpoint-step-009244" \
  --output-dir "$RUN2" --max-lr 2e-4 --min-lr 2e-5
```

默认数据目录为 `data/SmolLM2-135M-10Bsubset`，可加 `--data-dir PATH`。
省略 output-dir 会使用隔离的 `formal_<timestamp>_<pid>/formal`。
日志位于远程 `code/RSmol/log/mizar_text_5_10x2_5_gbs1024/`，每 10 步显示实际 loss/lr/速度。
加 `--dry-run` 只输出配方报告，不读模型/数据、不初始化 CUDA、不提交 vc。

同一个 epoch 中断后的完整恢复使用 `--resume-from`，LR 必须与该 checkpoint 一致，
输出使用新空目录，累计训练到本 epoch 的 step9244：

```bash
bash code/RSmol/run_stage4_mizar_text_5_10x2_5_training_5090.sh \
  --resume-from "$RUN1/checkpoint-step-009000" \
  --output-dir "${RUN1}_resume" --max-lr 8e-4 --min-lr 8e-5
```

完整恢复继承母本的限制：保存八 rank 数据游标与 bounded-shuffle RNG，但不保存 buffer 内容；
模型 Python/Torch/CUDA RNG 只保存 rank0 快照。恢复不保证逐样本连续、八 rank 独立 RNG 恢复或 bitwise exact。
跨 epoch 请使用上面的 `--model-dir` 命令。

## 本地验证与提交代码

```powershell
cd C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
python -B tests/validate_mizar_text_5_10x2_5.py
```

报告：`audit_reports/mizar_text_5_10x2_5_validation.json`。
本地 28 项合同/母本 SHA/原完整更新循环与读取 AST/全 9245 点 LR/初始化/模拟 Bash 提交检查通过；
5 项真实 CPU 张量检查因缺少 torch/transformers/safetensors 跳过。
这不代表远程生产权重转换、八卡训练或恢复已经验证。
需要时可在远程 rsmol 环境运行以下合成 CPU 检查，它不是训练前置步骤：

```bash
python -B tests/validate_mizar_text_5_10x2_5.py --require-tensors
```

仅提交本次交付文件（下面的 git add 命令在 PowerShell 和 Bash 都可执行）：

```text
git add -- RM.md code/RSmol/mizar_text_5_10x2_5 code/RSmol/scripts/convert_mizar_text_5_10x2_5.py code/RSmol/scripts/convert_mizar_text_5_10x2_5.sh code/RSmol/scripts/train_stage4_mizar_text_5_10x2_5_ddp.py code/RSmol/scripts/train_stage4_mizar_text_5_10x2_5_ddp.sh code/RSmol/run_stage4_mizar_text_5_10x2_5_training_5090.sh tests/test_mizar_text_5_10x2_5.py tests/test_mizar_text_5_10x2_5_shell.py tests/test_mizar_text_5_10x2_5_tensors.py tests/validate_mizar_text_5_10x2_5.py audit_reports/mizar_text_5_10x2_5_validation.json
git diff --cached --stat
git commit -m "Add Mizar final text conversion and single-epoch recurrent training"
git push
```

未自动 commit/push，未访问远程生产资产，也未提交真实训练作业。
