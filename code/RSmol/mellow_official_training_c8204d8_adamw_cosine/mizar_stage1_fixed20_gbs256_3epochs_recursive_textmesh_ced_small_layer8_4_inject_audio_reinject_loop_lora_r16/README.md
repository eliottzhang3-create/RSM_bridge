# 文本 MeSH + 音频串行 input→CED8 / input→CED4 + 双循环 LoRA rank16

独立母本：`mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8`。
音频机制参考非 MeSH `layer8_4_inject_audio_reinject_loop_lora_r8`，rank16 实现核对已有非 MeSH rank16 路线。
本地 Windows 只维护核心代码；权重、数据与正式训练在远程 Linux，代码通过 GitHub 同步。
原母本、非 MeSH 路线、共享模型和现有评测器保持原样。

## 精确执行流程

音频位置为 `[0:126]`；文本侧为 `[126:]`，包括 SEP、prompt、answer/PAD 或生成位置。
仍是同一个语言模型，完整序列的 causal attention、position IDs 与 RoPE 保持原样。
物理层 0..4 为 prelude，5..14 共享循环两次，15..19 为 suffix，共20个独立物理层、30次执行。

| 边界 | 文本侧 | 音频侧 |
|---|---|---|
| Prelude 输出后 | 第0组 routers 写入/读出文本记忆 | prelude audio output → input 残差 → CED8 残差 |
| 第一次循环输出后 | 第1组 routers 写入/读出文本记忆 | loop1 audio output → input 残差 → CED4 残差 |
| 第二次循环输出后 | 第2组 routers 写入/读出文本记忆 | loop2 audio output 直接进入 suffix |

各边界处理后重新拼接音频和文本，进入下一段完整序列计算。
五槽 memory 实际形状为 `[B,5,L-126,576]`；训练时为 `[8,5,557,576]`。
slot0 初始化为原始文本侧 input embeddings，其余四槽为零，每次完整 forward 重建。
音频不分配记忆，不调用 routers，不参与任何直接 memory write/read。
文本仍可以通过 self-attention 接收音频信息。

Prelude 的 router query/write value 均为 prelude 文本输出。
后两次 router query 为对应循环的输入文本 hidden，write value 为该循环输出文本 hidden。
不把循环输出改作 router query；保留母本三次 write/read 的参数、公式和顺序。

## 音频注入与初始化

冻结、eval 的 CED-small 同一次编码提取第4和第8个完整 block 输出，位置在最终 LayerNorm 之前。
独立 MLP 将两份 `[B,126,4,384]` 特征映射为 A4/A8 `[B,126,576]`：
`LN1536 → Linear1536/576 → GELU → Dropout0 → Linear576/576 → LN576`。
MLP 在 CED 的 no_grad 之外，保持可训练；原 final-layer mapper / freq_merge_126 路径和 dropout0.05 保持。

令 E 为原 mapper 的126个 audio input embeddings：

```text
R = audio_linear(E)                         # Linear576/576，每次完整 LM forward 一次
u = h_audio + tanh(audio_gate(h_audio)) * R
next_audio = u + tanh(audio_gates[k](u)) * A_k
# logical5: k=8；logical15: k=4；logical25: 直接传递
```

两处共享 input projection 和 input gate，CED4/8 MLP/gate 分别独立，共三个 gate 网络。
CED gate 明确读取 post-input hidden；两条残差均不 detach，不增加额外归一化、尺度或第三次注入。
input projection 使用原 Linear 默认随机 weight、零 bias；三个 gate 的 weight/bias 全零。
严格完整加载原 MeSH 文本权重和六个 routers 后，依次注册 CED gates、input 分支和 LoRA。
新增初始化采用独立 CPU fork_rng，保留继承参数及调用者 RNG。

输入 projection/gate 新增332,929参数；两份 CED MLP/gate 新增2,444,930参数，合计2,777,859。
音频不走记忆是结构变化，即使 gate 和 LoRA B 全零，也不要求与原全序列 MeSH 输出相同。
零 LoRA 对照采用相同 hybrid 结构、直接计算基础投影的参考。

## LoRA 与正式训练配方

| 项目 | 配置 |
|---|---|
| LoRA 范围 | 物理层5..14；q/k/v/o、gate/up/down；作用于完整序列 |
| LoRA 两轮 | 每层、投影、轮次独立 A/B；第1轮索引0，第2轮索引1 |
| rank / alpha / scale / dropout | 16 / 32 / 2 / 0 |
| LoRA 初始化 | A Kaiming uniform，B zero；无 merge、无冻结切换 |
| LoRA 参数 | 每套1,628,160；两套3,256,320；基础W仅一份并继续全量训练 |
| 相对普通 MeSH CED 新增 | 串行音频2,777,859 + LoRA3,256,320 = 6,034,179 |
| 文本初始化 | 原 MeSH formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244 |
| 数据 | hybrid 1,477,492行；原 MusicCaps / SpeechCraft 范围保持 |
| 序列 | audio126 + SEP1 + prompt256 + answer300 = 683，prefix383 |
| 数值与监督 | FP32，无 autocast，TF32关闭；原 answer-only / rank-local token mean |
| Batch | 8GPU × microbatch8 × accumulation4 = GBS256 |
| 更新 | 3 epochs；5771/epoch；17313 total；866 warmup；每epoch丢弃116行 |
| max LR | 基础 LM/routers 3e-4；mapper、全部音频新增模块和两套 LoRA 6e-4 |
| 优化器与调度 | 原 AdamW / LambdaLR / GradNormTracker / GradScaler |
| staging | 立即从共享路径训练；4线程后台复制；optimizer 边界八卡一致切换 |
| 资源 | mellow_c8204d8；pdgpu-5090 / 32CPU / 256G / 8GPU / 1node |

共12类优化器参数角色；全部新增参数纳入 GradNorm、参数变化/hash、完整保存和恢复。
首步审计允许零 gate 对应上游 MLP/projection、零 B 对应 A 的零任务梯度，要求 gate/B 有梯度。
第一次更新后检查所有音频分支和两套 A/B 解锁；六个 routers 的梯度存在且有限，非零仅作诊断。
每次 forward 记录实际 memory / router query 形状；首步审计拒绝音频进入 memory 或 routers。
沿用 use_cache=False，拒绝 KV cache、错误音频特征形状和 decoder gradient checkpointing。

## 远程正式提交

本地代码提交并推送到 GitHub 后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_textmesh_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16_5090.sh --stage-workers 4
```

命令通过 vc wrapper 直接提交一个 formal，不添加自动 smoke/tensor 前置作业。
登录节点仅检查资产和准备 CPU metadata 索引，GPU训练在计算节点执行。
默认仍从原 MeSH 文本 checkpoint 开始音频训练，不加载已有 rank8 音频 checkpoint。

新 route：`ced_small_mizar_s1_fixed20_gbs256_recursive_textmesh_layer8_4_inject_audio_reinject_loop_lora_r16_v1`。
训练目录、启动器、route、output、job、日志、index-cache、tmpfs、full-resume 身份全部隔离。
默认输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_textmesh_layer8_4_inject_audio_reinject_loop_lora_r16_5090/formal_<timestamp>/
```

主要产物：`formal/runtime.json`、`formal/config_audit.json`、`formal/metrics.jsonl`、
`formal/first_gradient_rank<N>.json`、`formal/opened_gate_gradient_rank<N>.json`、
`formal/staging_report.json`、`formal/training_report.json`，以及
`formal/checkpoints/model--step-17313.ckpt`、`formal/checkpoints/model--epo-3.ckpt`。
保存步数仍为2886、5771、8656、11542、14428、17313，并保留每epoch保存。
自定义输出附加 `--output-dir /完整/新空目录`。

## 完整恢复

仅接受该新路线同源码、资产、数据、配方的完整 checkpoint。
严格恢复全部模型参数、optimizer/scheduler、GradScaler/GradNorm、每rank RNG和真实数据游标。
原 MeSH rank8、非 MeSH rank8/rank16、其他记忆范围或注入顺序均不能 full resume。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_textmesh_layer8_4_inject_audio_reinject_loop_lora_r16_5090/formal_替换为实际时间戳"
bash run_mizar_s1_gbs256_recursive_textmesh_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16_5090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

## 本地验证与证据边界

仓库根目录运行：

```bash
python -B tests/validate_mizar_s1_textmesh_ced_loop_lora_r16.py
# 在具备原依赖的环境中手动要求真实CPU tensor检查全部运行：
python -B tests/validate_mizar_s1_textmesh_ced_loop_lora_r16.py --require-tensors
```

覆盖新路线与 MeSH母本、非MeSH rank16 回归，真实 Bash 假提交与拒绝用例，
完整 train AST、18个复制源码受限差异、189个原文件SHA保护、原数据/staging/恢复合同，
以及实际源码的 NumPy memory/query/write、串行 gate、两轮 LoRA 和梯度审计失败用例。
提供8项新路线真实CPU tensor检查：严格文本继承、LoRA独立存储/RNG/轮次/反向、
同hybrid结构零LoRA对照、三次文本记忆、音频post-input gate/最终直通、跨forward隔离、完整保存后下一步对齐。
CPU checkpoint collective/scaler 使用明确fixture，不代表真实八卡恢复。

本地85项实际检查通过；22项真实CPU tensor检查因缺依赖跳过，其中新路线8项。
Python/Bash语法、假提交/拒绝和189个原文件SHA保护通过。
验证报告：`audit_reports/mizar_s1_textmesh_ced_loop_lora_r16_validation.json`；源码对照：本目录 `reference_audit.json`。
本地缺少 Torch/Transformers 等依赖时，tensor 检查明确 skip；`--require-tensors` 会拒绝跳过。
本次不下载资产、不使用CUDA、不提交真实训练，不将本地检查登记为远程PASS。
现有 MMAU/MMAR evaluator 尚未适配该 hybrid/rank16 合同，需单独适配。
