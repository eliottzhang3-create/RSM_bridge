# 非 MeSH 双循环 LoRA：两个隔离的正式训练版本

均从原 `layer8_4_inject_audio_reinject` 母本完整复制，保留 input embeddings 串行残差。
只有中间十个物理层 5..14 加两套 LoRA；前五层、后五层无 LoRA。
两轮共享原 W 并全量训练，各层各投影的两套 A/B 独立。
q/k/v/o 与 gate/up/down 全覆盖，rank=8、alpha=16、scale=2、dropout=0、LoRA max_lr=6e-4。

| 版本 | 逻辑入口5 | 逻辑入口15 | 启动器 |
|---|---|---|---|
| 8→4 | input → CED8 → 第1套LoRA | input → CED4 → 第2套LoRA | `run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8_5090.sh` |
| 4→8 | input → CED4 → 第1套LoRA | input → CED8 → 第2套LoRA | `run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8_5090.sh` |

中间 gate 始终读取 input 残差后的 hidden；接口始终是 (A4,A8)，注册次序始终4、8。
音频残差只直接影响前126位置，LoRA作用于该轮完整序列。
严格继承原文本checkpoint后再注册新增模块。A采用独立CPU RNG的Kaiming uniform，B=0；
两版在同seed下对应初值一致，不改变原音频分支初值或后续训练RNG。
每套 LoRA 814,080 参数，两套合计新增1,628,160；训练时按真实投影形状校验。

完整实现和恢复说明：

- [8→4 版本](../mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8/README.md)
- [4→8 版本](../mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8/README.md)

## 正式提交

本地代码通过 GitHub 提交/推送后，远程登录节点同步。先运行一次：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
```

8→4：

```bash
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8_5090.sh --stage-workers 4
```

4→8：

```bash
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8_5090.sh --stage-workers 4
```

两条命令分别提交一个正式训练作业。沿用 mellow_c8204d8、vc submit、pdgpu-5090 /
32CPU / 256G / 8GPU / 1node，不自动插入 smoke/tensor 前置任务。
本地只有核心代码；权重、数据、checkpoint 均在远程Linux，CUDA/torchrun通过计算节点运行。

## 母本配置与隔离

仍为原 stage4_5_10_5 checkpoint-step-009244 文本初始化、hybrid 1,477,492行、fixed20、
683-token布局、原answer-only loss、FP32/GradScaler、8×8×4=GBS256、3epochs、
5771updates/epoch、17313total、866warmup、原AdamW/LambdaLR/GradNorm及4线程后台staging。
LM max_lr=3e-4；两套LoRA与原音频分支max_lr=6e-4。

两个输出根分别为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8_5090/formal_<timestamp>/
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8_5090/formal_<timestamp>/
```

各自最终产物为 formal/checkpoints/model--step-17313.ckpt 与 model--epo-3.ckpt。
route、runtime合同、job、日志、index-cache、tmpfs、输出根及full-resume身份全部分开。
完整checkpoint包含基础LM、全部LoRA和音频分支、optimizer/scheduler/scaler/GradNorm、8rank RNG及游标。
只能在对应版本原run root恢复，跨顺序或无LoRA母本会被拒绝。
现有评测器需要另行支持双LoRA和对应注入顺序。

## 本地验证

```bash
python -B tests/validate_mizar_s1_ced_loop_lora.py
```

149项实际检查通过，51项真实CPU tensor检查因本地缺依赖跳过（双LoRA两版本共14项、母本回归37项）。
包括非零LoRA逐轮执行的实际源码数值对照、scale2、有限差分、post-input gate、
12类参数分组与全部LR位置、首步/打开后的梯度审计失败用例、恢复身份、路径隔离、Bash假提交和原路线回归。
Python/Bash语法通过；34个母本原文件SHA保护，完整训练函数及骨干AST对照通过。
报告：`audit_reports/mizar_s1_ced_loop_lora_validation.json`。

服务器有原依赖时可手动加 `--require-tensors` 禁止skip；它是独立CPU合成检查，不是formal前置任务。
CPU张量检查涵盖真实反向、严格文本权重继承、两套独立存储、RNG与checkpoint后下一次更新对齐。
checkpoint测试的collective/scaler为CPU fixture；本地模拟与skip不代表远程GPU PASS。
本次未执行git commit/push、远程资产加载、真实GPU训练或八卡恢复。
