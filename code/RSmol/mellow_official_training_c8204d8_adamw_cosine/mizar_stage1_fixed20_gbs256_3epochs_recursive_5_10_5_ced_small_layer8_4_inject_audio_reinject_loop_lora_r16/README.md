# 非 MeSH input → CED8→4 + 两轮独立 LoRA：rank16 / alpha32

从 `mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8`
完整复制隔离，正式入口为 `run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16_5090.sh`。
相对直接母本，仅改变 rank8→16、alpha16→32、对应参数计数与隔离身份；scale 仍为2。
原 rank8 的两个顺序、MeSH 路线和共享模型保持原样。
本地 Windows 维护核心代码，经 GitHub 同步；权重、数据及正式训练位于远程 Linux。

## 模型与 LoRA

仍为非 MeSH 20 个物理层、30 次逻辑执行，轨迹 `0..14,5..14,15..19`。
前五层物理0..4、后五层物理15..19不加 LoRA，继续全量训练。
仅中间十个物理层5..14的七个投影增加双 LoRA：

- Attention：q_proj、k_proj、v_proj、o_proj。
- MLP：gate_proj、up_proj、down_proj。

每个物理层、每个投影有两套独立 A/B。第一轮逻辑5..14用索引0，第二轮逻辑15..24用索引1。
两轮基础 W/bias 共享且全量训练；LoRA 作用于该轮完整序列：

```python
y = linear(x, W, bias) + (32 / 16) * linear(linear(x, A[loop]), B[loop])
```

| 项目 | rank8 母本 | 本版本 |
|---|---:|---:|
| rank | 8 | 16 |
| alpha | 16 | 32 |
| scale | 2 | 2 |
| LoRA dropout | 0 | 0 |
| LoRA max_lr | 6e-4 | 6e-4 |
| 矩阵 decay | 0.01 | 0.01 |
| LoRA A/B 矩阵数 | 280 | 280 |
| 十层单套 LoRA 参数数 | 814,080 | 1,628,160 |
| 两套 LoRA 参数数 | 1,628,160 | 3,256,320 |
| 保留的串行音频分支参数数 | 2,777,859 | 2,777,859 |
| 相对普通 CED 循环母本新增合计 | 4,406,019 | 6,034,179 |

真实 SmolLM2-135M 为 hidden576、MLP1536、Q9/KV3 heads：q/o 的形状576×576，
k/v 为192×576，gate/up/down为1536×576或其转置。trainer 根据实际投影形状核对参数数。
本版本相对直接 rank8 母本额外增加1,628,160参数。

A 仍为 Kaiming uniform（a=sqrt(5)），B=0；严格加载全部原文本权重后再注册音频模块及 LoRA。
保留继承的基础 weight/bias 对象与 checkpoint key。新增模块采用独立 CPU fork_rng，
与 rank8 母本对应的基础权重、音频分支初值和后续 CPU RNG保持一致；LoRA A/B按新形状独立初始化。
所有基础 LM 和两套 LoRA 始终可训练，不在两轮之间切换 requires_grad。
每层调用显式选择该轮 A/B，返回或异常时均复位；反向保留该次前向选择的张量。
保持 use_cache=False、完整序列重算、decoder gradient checkpointing关闭。

## 保留的音频输入和注入顺序

CED-small冻结且所有子模块eval。每批编码一次，同时取final和完整block4/block8的最终norm前输出。
原 freq_merge_126 mapper 输出 E `[B,126,576]`，E作为 LM 输入音频prefix。
A4/A8分别经过独立 `LN1536 -> Linear(1536,576) -> GELU -> Dropout0 -> Linear(576,576) -> LN576`。
每个完整 LM forward 仍从原E计算一次共享 `A_input=audio_linear(E)`，保持梯度。

| 逻辑入口（零基） | 串行音频残差 | 中段 LoRA |
|---|---|---|
| 5 | 共享 input 残差 → CED8 残差 | 第1套，rank16/alpha32 |
| 15 | 同一 input 分支残差 → CED4 残差 | 第2套，rank16/alpha32 |

CED gate读取input残差之后的hidden。音频残差只直接作用前126位置；音频接口仍为 `(A4,A8)`，
两套MLP/gate注册顺序仍为4、8。保留三个零初始化gate、共享input投影和原始音频prefix。
零B、零gate时保持母本参考计算；第一步检查两轮B有任务梯度、A任务梯度为零，
B首次更新后检查两轮A/B任务梯度。音频gate关闭时上游任务梯度为零，打开后再检查。

## 保留的正式训练配置

文本初始化仍为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244
```

仅继承原文本权重/tokenizer，新音频训练的优化器、调度器和数据游标从头建立。

| 项目 | 固定配置 |
|---|---|
| 数据 | 原hybrid 1,477,492行，manifest/resolved_paths及全部sources保持 |
| 音频 | fixed20原始波形在线处理，冻结CED，无hidden cache |
| 序列 | 126 audio + SEP1 + prompt256 + answer300 = 683，prefix383 |
| 标签与loss | PAD !/17，EOS/SEP0；attention_mask=None；原answer-only rank-local token mean |
| 数值 | FP32，无autocast/TF32，保留GradScaler |
| Batch | 8 GPU × microbatch8 × accumulation4 = GBS256 |
| 更新 | 3epochs，5771/epoch，17313total，866warmup，每epoch丢116行 |
| max_lr | 基础LM 3e-4；mapper、音频分支、两套LoRA均6e-4 |
| 优化器 | 原AdamW betas(.9,.999)、eps1e-8、矩阵decay.01、bias/一维不衰减 |
| 调度/梯度 | 同一LambdaLR、GradNormTracker、GradScaler、12类参数角色和审计 |
| 随机与采样 | seed1234、rank seed1234+rank及原shuffle/sampler |
| staging | 共享读取立即开训；默认4线程后台复制；optimizer边界八卡共同切换 |
| 保存 | step2886/5771/8656/11542/14428/17313及原epoch保存 |
| 环境与资源 | mellow_c8204d8；pdgpu-5090 / 32CPU / 256G / 8GPU / 1node |

训练器、优化器、骨干和文本加载器整个文件与 rank8 母本相同，完整源码审计只允许指定rank/alpha/count与命名变化。
两套 LoRA 仍纳入首步/更新后梯度审计、参数变化/全量新增参数hash、GradNorm、checkpoint和恢复审计。

## 远程正式提交

将本地修改经 GitHub 提交/推送后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16_5090.sh --stage-workers 4
```

直接通过vc submit提交一个formal作业，无自动smoke/tensor前置任务。
登录节点只进行资产检查、CPU metadata索引及提交，CUDA/torchrun在计算节点运行。
保留 `--output-dir`、`--resume-checkpoint`、`--resolved-paths`、`--text-model-dir`、
`--ced-model-dir`、`--index-cache-root`、`--stage-workers` 参数。

新route为 `ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r16_v1`。
默认输出根：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r16_5090/formal_<timestamp>/
```

目录、启动器、route、output、job、日志、index-cache、tmpfs和完整恢复身份均隔离。
主要产物为 `formal/runtime.json`、`config_audit.json`、`first_gradient_rank<N>.json`、
`opened_gate_gradient_rank<N>.json`、`metrics.jsonl`、`staging_report.json`、`training_report.json`。
最终权重为该run下的：

```text
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
```

## 完整恢复

仅接受本rank16/alpha32路线同源码、资产、数据与配方的完整formal checkpoint，
恢复全部基础LM、LoRA和音频分支、AdamW/LambdaLR、GradScaler/GradNorm、8rank RNG及真实游标。
rank8的A/B形状和恢复合同不同，不能直接full resume；其他顺序及MeSH checkpoint也不兼容。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
# 替换为本rank16路线的实际run根目录和已保存的checkpoint
RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r16_5090/formal_替换为实际时间戳"
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16_5090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

## 验证与边界

仓库根目录执行：

```bash
python -B tests/validate_mizar_s1_ced_loop_lora_r16.py
# 具备原依赖时手动运行，禁止真实CPU tensor检查被跳过：
python -B tests/validate_mizar_s1_ced_loop_lora_r16.py --require-tensors
```

本地108项实际检查通过，33项真实CPU tensor检查因缺Torch/Transformers等依赖跳过：
新版7项、原两条rank8路线共14项、原串行音频母本12项。
通过Python/Bash语法、rank16/alpha32及GQA参数计数、非零LoRA数值/轮次/scale2、有限差分、
post-input gate、全部12类参数LR/decay及调度、梯度审计失败用例、跨rank恢复合同拒绝、
正式Bash假提交、staging与原rank8两顺序/串行音频母本回归。

179个原有文件SHA保护；26个复制源码文件按明确允许的变化逐文件校验。
完整train/optim/骨干/loader AST与rank8母本相等，音频与训练合同除rank/alpha/count和身份外相等。
报告为 `audit_reports/mizar_s1_ced_loop_lora_r16_validation.json`，
本目录 `reference_audit.json` 保存源码审计；原文件基线为 `audit_reports/mizar_s1_ced_loop_lora_r16_parent_baseline.json`。

可选真实CPU tensor测试覆盖rank16矩阵形状/独立存储、基础权重/RNG继承、零B输出和基础梯度、
非零逐投影轮次/反向、首步/更新后梯度、严格文本加载、rank8状态加载拒绝，
以及完整保存后下一步与不中断更新一致性。checkpoint的collective/scaler为CPU fixture，
不能据此声称真实八卡恢复通过。runtime_checks.py手动调用严格验证器，不进入formal调用链。

本次未执行git commit/push、远程资产加载、真实GPU训练或分布式恢复，不登记远程PASS。
当前MMAU/MMAR评测器绑定rank8合同，本rank16/alpha32实验的评测需要单独适配。
