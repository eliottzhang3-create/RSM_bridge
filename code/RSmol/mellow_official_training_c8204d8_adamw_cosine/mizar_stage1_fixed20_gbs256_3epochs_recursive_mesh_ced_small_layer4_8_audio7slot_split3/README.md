# CED4/8 / MeSH audio7slot split3 / Mizar S1 GBS256

完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_audio7slot`。
架构和训练配方保持该版本，只替换六个音频 router 的初始化，更新对应诊断和独立 route 身份。
原 K=7、48listen、CED MeSH 路线及共享模型文件均未修改。
本地只有代码；通过 GitHub 同步，真实权重、数据、音频和 GPU 训练位于远程 Linux。

## 初始化：原 slot0 概率三等分

以下槽位和行号均为零基。原五路 router 对同一 query 的 logits 是 `[z0,z1,z2,z3,z4]`，
新七路初始 logits 为：

```text
[z0-log(3), z1, z2, z3, z4, z0-log(3), z0-log(3)]
```

对应 softmax 概率为 `[p0/3,p1,p2,p3,p4,p0/3,p0/3]`。
三个指数项之和等于原第一项，softmax 分母和其他四槽概率保持不变；两新增槽合计得到原 slot0 概率的2/3。
此关系仅针对相同 query；整个七槽模型的后续 query/输出不必与原五槽模型一致。

先 strict 加载原五槽文本 checkpoint，再逐一初始化对应阶段、对应类型的三个 write 和三个 read 音频 router。
权重映射如下：

```python
W_new = W_old[[0, 1, 2, 3, 4, 0, 0]].clone()
b_new = b_old[[0, 1, 2, 3, 4, 0, 0]].clone()
b_new[[0, 5, 6]] -= math.log(3.0)
```

实现使用逐行 copy 到新模块的独立参数，**音频第一行 bias 也减去 log(3)**。
六个原文本 router 完整保留原参数；所有音频 router 与原 router 无参数绑定，三行也不绑定。
新模块注册隔离 CPU RNG，不改变 mapper、MLP 或原 LM 的初始参数。
不再设置 K 或新增行零权重；相同初值只用于初始化，七行后续正常独立训练。

## 架构保持不变

| 内容 | 设置 |
|---|---|
| Transformer | hidden576，20物理层、30次执行，`0..14,5..14,15..19` |
| 音频记忆 | 前126个位置、7槽，prelude前初始化为 `[E,0,0,0,0,A4,A8]` |
| 文本记忆 | SEP/prompt/answer/PAD位置、5槽，初始化为 `[embedding,0,0,0,0]` |
| Routers | 原文本3 write/3 read；新增音频3 write/3 read，共12个 |
| CED | 冻结/eval，一次编码提取第4/8个block完整输出，位于最终LN之前 |
| 中间特征顺序 | `[B,504,384] → [B,2,4,63,384] → [B,126,4,384]` |
| A4/A8 MLP | 两套独立 `LN1536 → Linear1536/576 → GELU → Dropout0 → Linear576/576 → LN576` |
| 注入 | A4/A8只在初始化各存入slot5/6，后续不再注入；无旧listen router/gate |
| 三次交互 | prelude后、循环0后、循环1后；所有槽均参与普通write/read |
| Query/value | prelude均用输出；两次循环用输入作query、输出作write value |
| Memory运算 | additive write后weighted read，无额外residual、归一化或补偿缩放 |
| Forward | 每次完整重建memory，无KV cache；推理每次必须显式传入A4/A8 |

`E` 是 audio input embeddings，A4/A8仍来自两个MLP，不把新增槽内容改成E。
文本保留五槽机制，完整 Transformer 的因果注意力仍会感知音频状态。
相对于原 CED MeSH，MLP 2,443,776参数、音频 router 24,234参数，新增合计2,468,010；与K=7版本相同。

概率守恒不表示模型函数保持不变。若同一组三槽的write/read权重均三等分，则本次普通写入再读取的贡献是
`3 * (p_write0/3) * (p_read0/3) * V = p_write0*p_read0*V/3`。
三个seed的读取贡献为 `p_read0*(E+A4+A8)/3`。这正是指定初始化和原MeSH运算的结果，不额外乘3补偿。

## 原训练配方

- 原LM及六个文本router maxLR3e-4；原mapper、两个MLP和全部六个音频router maxLR6e-4。
- 矩阵decay0.01，bias/ndim≤1不衰减；AdamW betas(.9,.999)、eps1e-8。
- 原LambdaLR 5%warmup+cosine/floor0.1、GradNormTracker、GradScaler；FP32 forward、无autocast/TF32。
- hybrid 1,477,492行、原shuffle/rank RNG/连续分片、原音频fixed20预处理和官方最终层mapper/dropout0.05。
- 原audio126+SEP1+prompt256+answer300，总683 tokens；answer-only CE/EOS/PAD规则及attention_mask=None不变。
- 8 GPU × microbatch8 × accumulation4 = GBS256，3epochs，每epoch5771updates，总17313、warmup866、每epoch丢弃116行。
- DataLoader workers0；后台原始音频字节复制默认4workers，八卡在optimizer边界一致切换本地文件。
- 原保存节点、模型/optimizer/scheduler/scaler/GradNorm、八rank RNG、数据游标和完整恢复流程。

## 正式训练提交

本地改动提交并推送 GitHub 后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_split3_5090.sh --stage-workers 4
```

直接提交formal，无自动smoke/tensor检查前置任务。资源沿用
`pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`，原镜像和`mellow_c8204d8`环境。
登录节点只做原有资产检查和CPU metadata索引；训练通过`vc submit`在GPU作业中执行。

默认输出root：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_5090/formal_<timestamp>/
```

也可显式指定，注意root内还会生成`formal/`：

```bash
OUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_5090/formal_$(date +%Y%m%d_%H%M%S%N)
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_split3_5090.sh \
  --stage-workers 4 --output-dir "$OUT"
```

默认初始化仍来自原文本checkpoint，而非已训练的K=7音频checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244
```

CED默认`/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small`。
launcher保留`--text-model-dir`、`--ced-model-dir`、`--resolved-paths`、`--index-cache-root`覆盖选项。
输出、日志、index cache、tmpfs和resume身份与K=7路线独立。

## 记录和恢复

`formal/config_audit.json`保存架构/初始化身份、optimizer参数分组、SHA及每来源真实音频probe。
probe为eval，运行后恢复训练状态和RNG；记录六个router概率、seed/memory/read RMS。
训练首两次及每25次update记录七槽均值、两个新槽最大值/合计、熵和梯度；概率单位为fraction。

新增诊断字段`slot0_group_mean/max`表示槽0/5/6概率之和，`split_group_spread_mean/max`表示这三个槽概率的最大差。
fresh时三槽概率相同，训练后允许自然分化。`first5_expanded_logsumexp_mean/min`表示扩展router前五项的logsumexp，
其中slot0已经减去log(3)，不把它误称为原五路logsumexp。
0.5%/1%超过比例沿用作观察，不设概率上限或自动修改初始化。

`first_gradient_rank<N>.json`检查训练参数梯度存在/有限、冻结CED无梯度和30层轨迹。
`audio_gradient_metrics.jsonl`记录首两次及每25次update累积unscale后的新分支/新行梯度。
没有gate解锁期限，不要求真实样本每个梯度都非零；合成CPU测试另行验证有效非零梯度通路。

最终完整产物位于输出root内部：

```text
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/training_report.json
```

恢复只接受相同本路线源码/资产/数据/训练合同的未完成完整checkpoint。
新身份为`ced_small_mizar_s1_fixed20_gbs256_recursive_mesh_layer4_8_audio7slot_split3_v1`，合同记录
`router_initialization=split_slot0_three_ways`、source slot0、target slots0/5/6及bias offset=-log(3)。
恢复先构建模型，再strict覆盖完整训练状态，不把已训练参数重新三等分。
原K=7、48listen及其他路线checkpoint不能full resume到本路线；completed checkpoint可审计但不能resume。

中断后替换为原输出root和实际已保存的中间step：

```bash
OUT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_5090/formal_<实际timestamp>"
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_split3_5090.sh \
  --output-dir "$OUT" --resume-checkpoint "$OUT/formal/checkpoints/model--step-2886.ckpt" --stage-workers 4
```

现有评测器尚未适配新route/runtime身份，不能直接用于此checkpoint。

## 验证范围

```bash
# 仓库根目录；不使用真实资产、CUDA或作业提交
python -B tests/validate_mizar_s1_mesh_ced_audio7slot_split3.py
# 在具有原训练依赖的环境中要求真实CPU tensor检查全部执行
python -B tests/validate_mizar_s1_mesh_ced_audio7slot_split3.py --require-tensors
```

路线内`python -B runtime_checks.py --require-tensors`转发到同一入口，均不作为formal自动前置任务。
包含实际初始化函数的NumPy执行、六路参数/线性logit公式和极端logits概率守恒、实际memory公式/无额外缩放、
完整K=7模型/训练代码对照、旧路线回归、Bash模拟提交和staging失败清理、恢复身份拒绝。
可选真实CPU tensor测试覆盖strict文本加载、RNG/参数独立、softmax、梯度/因果性、完整恢复后下一次更新。

本地51项实际检查通过、Python/Bash语法通过；13项真实CPU tensor检查因缺
Torch/Transformers/Torchaudio/Soundfile跳过。报告：`audit_reports/mizar_s1_mesh_ced_audio7slot_split3_validation.json`。
源码对照报告为本目录`reference_audit.json`；均不代表远程真实权重、8GPU正式训练或恢复已经通过。
