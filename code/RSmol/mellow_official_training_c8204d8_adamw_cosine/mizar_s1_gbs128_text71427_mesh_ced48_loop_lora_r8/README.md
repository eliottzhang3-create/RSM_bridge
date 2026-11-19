# Mizar 7–14×2–7 MeSH + CED4→8 + 两轮独立 LoRA r8

2026-11-19 新增隔离路线。完整复制 GBS128 的 text71427 音频母本，移植既有 MeSH CED4/8、Loop-LoRA r8 架构。保留母本及共享文本实现，使用独立 route、输出、job/log、index-cache、tmpfs 和恢复身份。本地 Windows 仅改代码，经 GitHub 同步；下面所有资产及训练均在远程 Linux。

## 正式提交

代码 commit/push 到 GitHub 后，在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_loop_lora_r8_3090.sh
```

入口在登录节点执行 CPU 配置、完成标记和资产检查、准备索引，然后 `vc submit` 到 **pdgpu-3090 / CPU64 / 256G / 8 GPU / 单节点**。镜像沿用 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`，计算节点激活 `mellow_c8204d8` 并启动8 rank torchrun。直接执行 formal，运行审计包含在正式任务内部。

默认文本权重来源：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/formal_reinit1000_lr6e4_20261118_112042_3963870/checkpoint-009244
```

仅继承模型/config/tokenizer及全部已训练 routers；**音频从 optimizer step0 新训练**，不读取文本 `training_state.pt`。严格验证来源完成标记、结构、loading_info及全部 FP32 safetensors；只允许 tied embedding/head 省略一个别名。先核对原模型，再添加 gate/LoRA，检查原权重保持一致。保留全部28物理层、42次执行和 embedding/head绑定，不再次抽层或初始化 routers。

## 模型执行与参数

以下 LM 索引为0-based，CED block编号为1-based：

```text
CED最终层 → 官方 freq_merge_126 mapper → audio126 + SEP1 + prompt256 + answer300
→ 前段物理0..6 → MeSH write/read0 → CED4 gate残差
→ 中段物理7..20（LoRA₁）→ MeSH write/read1 → CED8 gate残差
→ 同一中段物理7..20（LoRA₂）→ MeSH write/read2
→ 后段物理21..27 → norm/head
```

- 5槽 memory 每次 forward 新建，形状 `[B,5,683,576]`；slot0为原输入embedding，其余为零。3组 write/read 共6模块、12参数张量、17,310参数，完整继承并训练。
- Prelude routers查询前段输出；两轮 routers查询各自注入后的循环输入，写入对应循环输出。router softmax保持FP32。
- CED-small严格离线加载 `/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small`，全部冻结、eval。一次编码获取最终层及第4/8个完整block输出（final norm之前）。
- 主 mapper 保留官方 `freq_merge_126`，`126×4×384 → 126×576`，dropout0.05、resample=False、外部projection为Identity，从头训练。
- 独立 A4/A8 MLP：`LN1536 → Linear1536→576 → GELU → Dropout0 → Linear576→576 → LN576`，位于冻结编码器的no_grad之外。
- 逻辑7/21处在read后、循环前注入：`H_audio += tanh(Linear576→1(H_audio)) * A4或A8`。gate权重/偏置均零初始化；仅直接修改前126个音频位置。A4/A8显式随每次forward传递，没有跨样本音频缓存。
- 仅物理7..20具有LoRA：两轮、每层、每projection的A/B各自独立；targets为q/k/v/o_proj及gate/up/down_proj，rank8、alpha16、scale2、dropout0。A为Kaiming，B为零。LoRA作用于这些层的完整序列，基础W继续共享且可训练。前后7层、routers、embedding/head不安装LoRA。
- LoRA共392个矩阵，生产形状预计2,279,424参数；两个音频MLP/gate共2,444,930参数；较无注入母本新增4,724,354参数，运行时核验实际形状/数量。
- 新模块初始化隔离CPU RNG，保持主mapper及后续随机序列。关闭KV cache和decoder gradient checkpointing，保持参考路线的逐层作用域选择，不合并adapter。

| 训练项 | 最大LR |
|---|---:|
| 全量LM及MeSH routers | 3e-4 |
| 原官方mapper | 6e-4 |
| CED4/8 MLP与gate | 6e-4 |
| 两轮LoRA A/B | 6e-4 |
| CED-small | 冻结 |

## 数据与训练合同

GBS128 = 8 GPU × micro4 × GA4；3 epochs，每epoch11,542步，共34,626步，warmup1,731步，每epoch丢弃116行。使用1,477,492行hybrid及母本原有source counts、模板、CoT和随机顺序，seed1234，各rank1234+rank。

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
```

音频mono/resample32k，长音频中心裁剪20秒，短音频右补零；接口的两段10秒在CED前重拼为完整20秒，保持整段mel/BN/1012帧分段、频率时间对齐及逐样本dB。历史processed filepath2不作为第二条原录音。显式双原录音继续按母本分别处理。

prompt/answer均追加EOS后按256/300截断和padding，截断后不补回EOS。PAD使用已有`!`，不扩词表，SEP/EOS=0。683位置、prefix383，attention_mask/labels=None，外部answer-only shifted CE：logits382..681预测300个answer目标，包含保留下来的EOS，排除PAD。损失保持rank内microbatch有效token均值、GA4平均、DDP平均。

原AdamW betas(.9,.999)、eps1e-8、WD.01（bias及ndim≤1不衰减），原LambdaLR warmup/cosine floor.1、GradNormTracker(1,10,2.5,.995)、GradScaler保持；参数/forward均FP32，关闭autocast和TF32。

CPU索引后立即从共享原音频训练，默认4个worker后台复制；全部校验完成后，在optimizer边界8rank MIN集体切换。tmpfs为 `/dev/shm/mizar_s1_gbs256_text71427_mesh_ced48_loop_lora_r8_<token>`，其中gbs256为保留的清理守卫前缀，实际GBS128。退出先停止/reap producer，再按owner/token清理。

## 输出、完整恢复与审计

默认run root：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced48_loop_lora_r8_gbs128_text71427_mesh_3090/formal_<timestamp>
```

实际训练目录为 `<run-root>/formal/`。checkpoint保存于其 `checkpoints/`，步数5771、11542、17313、23084、28855、34626，名称如 `model--step-5771.ckpt`，末尾另存 `model--epo-3.ckpt`。

可用 `--output-dir` 指定新的空run root。中断后使用**同一run root**和本路线完整音频checkpoint：

```bash
RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced48_loop_lora_r8_gbs128_text71427_mesh_3090/formal_REPLACE_WITH_ACTUAL_RUN
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_loop_lora_r8_3090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt"
```

支持母本全部路径覆盖参数及 `--stage-workers`。`--text-model-dir`是文本初始化来源，`--resume-checkpoint`接受本音频路线完整状态。恢复严格绑定route、架构、代码/共享依赖、资产、数据、超参数，恢复模型（含两套LoRA/MLP/gates）、optimizer/scheduler/GradScaler/GradNorm、8rank RNG和下一数据游标；暂存目录不属于语义身份。

正式任务核验42次调用、683位置memory、首次两轮边界及12个router梯度。首步B/gate应有任务梯度，零B下A、零gate下MLP的零任务梯度属预期；第一次更新后核验A/MLP梯度打开。边界引用审计后立即释放。记录injection trace、两轮LoRA trace及gate诊断。

关注 `config_audit.json`、`first_recursive_backward_rank<N>.json`、`first_gradient_rank<N>.json`、`opened_gate_gradient_rank<N>.json`、`metrics.jsonl` 和最终 `training_report.json`。

## 离线验证

```bash
python -B tests/validate_mizar_s1_text71427_ced48_loop_lora.py
# 完整ML依赖环境可要求所有真实CPU张量测试执行：
python -B tests/validate_mizar_s1_text71427_ced48_loop_lora.py --require-tensors
```

报告为 `audit_reports/mizar_s1_text71427_ced48_loop_lora_validation.json`；包内 `reference_audit.json` 记录62份保护源码及完整骨干/训练流程AST继承检查。测试使用合成数据和模拟vc，不提交真实训练。`--require-tensors`不作为formal前置条件。

本地缺少Torch等依赖时，真实张量/音频测试明确跳过。远程权重加载、3090实际峰值显存及正式训练结果仍须以计算节点的实际日志和报告为准。
