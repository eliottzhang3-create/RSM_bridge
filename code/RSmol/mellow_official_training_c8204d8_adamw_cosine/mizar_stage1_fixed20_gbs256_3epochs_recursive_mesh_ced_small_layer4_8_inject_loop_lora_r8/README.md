# MeSH CED 4→8 注入 + 中间十层双循环 LoRA / Mizar S1 GBS256

本路线完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject`。
母本启动器为 `run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_inject_5090.sh`。
仅在中间十个物理层增加两套独立 LoRA，两轮继续共享基础权重并全量训练。
本地仅维护核心代码；权重、数据及 checkpoint 位于远程 Linux，通过 GitHub 同步代码。

## 精确执行流程

```text
原始 audio embeddings / SEP / prompt / answer
  -> 原 MeSH prefix：物理层 0..4，无 LoRA
  -> write_routers[0] / read_routers[0] / memory read
  -> 前 126 个 audio hidden 加 CED 第 4 层残差
  -> 中段物理层 5..14，使用 LoRA 第 1 套（索引 0），逻辑位置 5..14
  -> write_routers[1] / read_routers[1] / memory read
  -> 前 126 个 audio hidden 加 CED 第 8 层残差
  -> 中段物理层 5..14，使用 LoRA 第 2 套（索引 1），逻辑位置 15..24
  -> write_routers[2] / read_routers[2] / memory read
  -> 原 MeSH suffix：物理层 15..19，无 LoRA
  -> final norm / LM head
```

仍为 20 个物理层、30 次逻辑执行，轨迹为 `0..14,5..14,15..19`。
保留 5 个瞬时 memory slots 和 6 个独立 write/read routers；router 不加 LoRA，继续全量训练。
每个 forward 重建 memory，第 0 槽为原始 input embeddings，其他四槽为零。
第一次 write/read 的 query 为 prefix output；后两次 query 为各自中段循环的输入 hidden，
写入值则是该轮中段 output。CED4/8 残差位于对应 memory read 之后，不增加第三次注入。

保留母本 input embeddings 音频 prefix 及两处 CED 注入；本路线没有非 MeSH 实验的额外 input 残差分支。
音频残差只直接作用于前 126 个位置；中段 LoRA 则作用于该轮完整序列。

## LoRA 的范围、初始化与训练

| 项目 | 固定配置 |
|---|---|
| 层范围 | 仅物理层 5..14（从 0 计数，即第 6–15 层） |
| Attention 目标 | q_proj、k_proj、v_proj、o_proj |
| MLP 目标 | gate_proj、up_proj、down_proj |
| 参数独立性 | 每层、每投影、每轮各有独立 A/B；两轮基础 W/bias 共享 |
| rank / alpha / scale | 8 / 16 / 2 |
| dropout | 0 |
| max_lr / decay | 6e-4 / 0.01；使用母本同一 LambdaLR 调度 |
| 基础 LM | 全量训练，含 prefix、共享中段、suffix、embedding、LM head、六个 routers |
| 初始化 | A 为 Kaiming uniform，B 为零；独立 CPU fork_rng |
| 安装顺序 | 严格加载原 MeSH 全部文本权重和六个 routers，再注册 gate 与 LoRA |

每个投影在第 i 轮计算 `linear(x,W,bias) + 2 * linear(linear(x,A_i),B_i)`。
基础 weight/bias 的原始参数对象和 state_dict 名称保留，新增 280 个 LoRA 矩阵。
每套 LoRA 814,080 参数，两套共新增 **1,628,160**；训练时根据真实投影形状核对。
相对普通 MeSH CED 母本，原 CED4/8 MLP/gate 的 2,444,930 参数加两套 LoRA，
合计额外 4,073,090；相对此次直接母本仅增加 LoRA 的 1,628,160。

每个中段层调用显式选择该轮 LoRA，返回时清理选择状态；反向使用该次 forward 选定的 A/B。
基础权重和两套 LoRA 始终可训练，不执行 adapter merge 或冻结切换。
不启用 decoder gradient checkpointing，保持母本原设置；误启用时显式报错。

## 保留的 CED 音频分支

冻结并保持 eval 的 CED-small 在同一次编码中提取第 4、8 个 block 的完整输出，
位置为 CED 最终 LayerNorm 之前。各自整理为 `[B,126,4,384]`，
由独立的 `LN1536 -> Linear(1536,576) -> GELU -> Dropout(0) -> Linear(576,576) -> LN576`
映射成 A4/A8，均为 `[B,126,576]`。
原 CED final-layer mapper / freq_merge_126 主路径与 dropout0.05 保留。

两个独立 `Linear(576,1)` gate 的 weight/bias 均为零初始化：

```text
h_audio = h_read[:, :126]
h_new_audio = h_audio + tanh(gate_layer(h_audio)) * A_layer
h_new = concat(h_new_audio, h_read[:, 126:])
```

MLP 在冻结 CED 的 no_grad 范围之外，保持梯度；SEP 和文本位置不直接注入。
MLP、gate 与 LoRA 新增初始化均隔离 CPU RNG，不改变母本 mapper 初值和后续 RNG。
每个 forward 显式传入 `(A4,A8)`，不缓存样本音频状态。
只接受完整 prefix 的 inputs_embeds、use_cache=False；拒绝 KV cache 或错误音频 shape/dtype/device。

## 保留的正式训练配置

| 项目 | 配置 |
|---|---|
| 文本初始化 | stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244 |
| 数据 | hybrid 1,477,492 行，保留 MusicCaps 和全部 SpeechCraft splits |
| 音频 | 原在线 fixed20、逐样本 dB、冻结 BN、两段 1012 帧 |
| 序列 | audio126 + SEP1 + prompt256 + answer300 = 683，prefix383 |
| Loss / attention | 原 answer mask、非 PAD 的 rank-local token mean；attention_mask=None |
| Batch | 8 GPU × microbatch8 × accumulation4 = GBS256 |
| 更新 | 3 epochs；5,771/epoch；17,313 total；866 warmup；每 epoch 丢弃 116 行 |
| max_lr | 基础 LM/routers 3e-4；原 mapper、CED4/8 MLP/gate、两套 LoRA 6e-4 |
| 优化器 | AdamW，betas=(0.9,0.999)，eps=1e-8，decay0.01，bias/一维不衰减 |
| 调度与梯度 | 原 LambdaLR warmup/cosine、GradNormTracker、GradScaler |
| 数值 | FP32 forward，无 autocast，TF32 关闭 |
| 数据顺序 | 原 seed1234 / rank seed1234+rank、shuffle、sampler |
| staging | 共享读取立即开始 formal；4 线程后台复制；optimizer 边界八卡一致切换 |
| 资源 | mellow_c8204d8；pdgpu-5090 / 32CPU / 256G / 8GPU / 1node |

增加四类 LoRA A/B 优化器角色，共十类角色；原参数 LR、decay 和更新流程保持。
首次反向时 B 为零，因此两套 A 的任务梯度为零、B 有非零梯度；B 更新后检查 A/B 梯度。
原 gate 首次有梯度、对应 MLP 初始任务梯度为零，gate 打开后检查 MLP。
保留全部可训练参数梯度存在/有限检查、六个 routers 梯度检查、30 层执行与音频注入轨迹。
router 非零梯度仅作诊断。新增 LoRA 轮次轨迹、参数变化、完整新参数 hash 和配置审计。
全部 LoRA 纳入原 GradNorm、checkpoint、完整恢复与重载审计。

## 远程正式提交

本地代码提交并推送到 GitHub 后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8_5090.sh --stage-workers 4
```

该命令提交一个正式训练作业，不自动插入 smoke/tensor 前置任务。
登录节点只核对资产并准备/复用 CPU metadata 索引，GPU 训练通过 vc wrapper 在计算节点执行。
默认文本与 CED 资产目录沿用直接母本。

独立 route 为 `ced_small_mizar_s1_fixed20_gbs256_recursive_mesh_layer4_8_inject_loop_lora_r8_v1`。
目录、启动器、route、output、job、日志、index cache、tmpfs 和 full-resume 身份均隔离。
原 MeSH 母本、共享模型和上一轮两个非 MeSH LoRA 版本保持不变。

默认输出根：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_loop_lora_r8_5090/formal_<timestamp>/
```

主要产物位于该根目录下：

```text
formal/runtime.json
formal/config_audit.json
formal/first_gradient_rank<N>.json
formal/opened_gate_gradient_rank<N>.json
formal/metrics.jsonl
formal/staging_report.json
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/training_report.json
```

保存步骤仍为 2886、5771、8656、11542、14428、17313，并保留每 epoch 保存。
如需固定输出位置，提交时附加 `--output-dir /完整/新空目录`。

## 完整恢复

只接受本路线同源码、资产、数据与配方的完整 checkpoint；包括基础 LM、两套 LoRA、
全部音频分支、optimizer/scheduler、GradScaler/GradNorm、每 rank RNG 和真实数据游标。
无 LoRA 的 MeSH 母本、非 MeSH LoRA 版本或其他顺序的 checkpoint 均不能 full resume。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
# 替换为本新路线的实际 run 根目录和已保存的 checkpoint
RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_loop_lora_r8_5090/formal_替换为实际时间戳"
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8_5090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

## 本地验证与证据边界

仓库根目录执行：

```bash
python -B tests/validate_mizar_s1_mesh_ced_loop_lora.py
# 在具备原依赖的环境手动运行，禁止真实 CPU tensor 检查被跳过：
python -B tests/validate_mizar_s1_mesh_ced_loop_lora.py --require-tensors
```

当前本地 100 项实际检查通过，23 项真实 CPU tensor 检查因缺少依赖跳过：
新双 LoRA 版本 7 项，MeSH CED4/8 母本 9 项，非 MeSH CED4/8 回归 7 项。
Python/Bash 语法、正式启动器假提交、非零 LoRA 数值/轮次对照、scale2/有限差分、
十类优化器角色及 LR、音频/LoRA 梯度审计失败用例、MeSH memory/query/write 行为、
staging 与母本回归通过。完整骨干和 train 函数剥离明确新增节点后与母本 AST 相等，
另有 102 个原文件 SHA 保护及 18 个复制文件仅命名变化的校验。

验证报告：`audit_reports/mizar_s1_mesh_ced_loop_lora_validation.json`。
源码差异审计：本目录 `reference_audit.json`。
真实 CPU tensor 测试覆盖独立存储、零 B 母本等价、逐投影轮次/反向、严格文本权重继承，
以及保存后下一步与不中断更新对齐。checkpoint collective/scaler 为明确 CPU fixture，
不能据此声称真实八卡恢复已通过。
`runtime_checks.py` 提供手动 `--require-tensors` 检查，不进入正式训练调用链。

本次未执行 git commit/push、远程资产加载、真实 GPU 训练或分布式恢复，不登记远程 PASS。
现有评测器需要另行适配本路线的双 LoRA；每样本编码一次，完整逐 token 重算时显式传入同一对 A4/A8。
