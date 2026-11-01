# CED4/8 / MeSH audio7slot split3 + post-read gated injection / Mizar S1 GBS256

完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_audio7slot_split3`。
在现有七槽 split3 模型上增加两个独立零初始化 tanh Linear gate；复用原两套 MLP 的 A4/A8。
gate 的输入明确为 memory read 后的音频 hidden，残差也加在该 hidden 上。
原 split3、K=7、48listen、layer4_8_inject 和共享模型均未修改。
本地只有代码；经 GitHub 同步，真实权重、数据、音频和 GPU 训练位于远程 Linux。

## 精确架构

```text
音频 memory = [E,0,0,0,0,A4,A8]；SEP/文本 memory = [embedding,0,0,0,0]
    -> Prelude 5层
    -> pre write/read -> R_pre
    -> 音频 R_pre + tanh(G4(R_pre_audio)) * A4
    -> 第一轮中段10层（逻辑5..14）
    -> loop0 write/read -> R_0
    -> 音频 R_0 + tanh(G8(R_0_audio)) * A8
    -> 第二轮共享中段10层（逻辑15..24）
    -> loop1 write/read
    -> Suffix 5层 -> final norm -> LM head
```

以上逻辑索引为零基；两次注入轨迹固定为 `[[5,4],[15,8]]`（逻辑索引、CED层号）。
20个独立物理层、30次执行，物理轨迹保持 `0..14,5..14,15..19`。
六个原文本router保持五路；六个独立音频router保持七路，共12个router。
没有额外的旧五路 listen router。

普通 write/read 的 query/value 语义保持：
pre 阶段均取 prelude 输出；循环后 query 取该循环的输入（已包含其前方的门控残差），
write value 取循环输出。gate 使用 read 后 hidden，不使用 router query 或循环输出。
第三次 read 后没有额外注入。

每个 gate 为独立 `Linear(576,1,bias=True)`，再经 tanh，每个音频token一个标量，
输出 `[B,126,1]`，广播到576通道：
```python
h_audio = h_read[:, :126]
g = torch.tanh(gate(h_audio))
h_next = torch.cat((h_audio + g * A, h_read[:, 126:]), dim=1)
```
仅前126个音频位置直接注入，SEP/文本在此操作中原样拼接，随后照常参与因果注意力。
没有附加投影、归一化、缩放或直接 memory 写入；后续普通 write 正常承接循环结果。

## A4/A8与 split3 初始化

冻结且始终eval的CED一次编码，提取第4/8个block完整输出（最终LN之前）。
各层按 `[B,504,384] -> [B,2,4,63,384] -> [B,126,4,384]` 整理。
两套独立MLP保持 `LN1536 -> Linear1536/576 -> GELU -> Dropout0 -> Linear576/576 -> LN576`。
两套MLP每次各计算一次，得到同一对A4/A8，同时用于prelude前记忆槽初始化及两次直接残差。
原最终层mapper及其dropout0.05不变，不增加CED前向或新的MLP。

严格加载原五槽文本checkpoint后，六个音频router仍对应初始化为：
```python
W_new = W_old[[0,1,2,3,4,0,0]].clone()
b_new = b_old[[0,1,2,3,4,0,0]].clone()
b_new[[0,5,6]] -= math.log(3.0)
```
相同query初始概率为 `[p0/3,p1,p2,p3,p4,p0/3,p0/3]`，三行后续独立训练。
原文本router不改；保留split3原本的memory公式，不增加乘3补偿。
两个gate的weight/bias均为0，tanh初始输出为0；注册使用独立CPU fork_rng，
保持继承参数和调用者RNG。相同权重/输入/随机状态下，初始forward与原split3数值对齐。

初始零gate仅阻断新增残差路径对A的梯度；两套MLP仍可经七槽初始memory获得梯度。
gate自身可从首次反向学习。不要求真实样本的每个梯度都非零，不设置解锁期限。

新增gate参数 `2*(576+1)=1,154`。
相对原五槽CED MeSH，新增总数为 `2,443,776+24,234+1,154=2,469,164`。
完整模型总数和可训练数由远程实际加载后的 `formal/config_audit.json` 记录。

## 原训练配方

- 原LM及六个文本router maxLR3e-4；原mapper、两套MLP、六个音频router及两个gate maxLR6e-4。
- 矩阵weight_decay0.01，bias/ndim≤1不衰减；AdamW betas(.9,.999)、eps1e-8。
- 原LambdaLR 5%warmup+cosine/floor0.1、GradNormTracker、GradScaler；FP32，无autocast/TF32。
- hybrid1,477,492行，原shuffle/rank RNG/连续分片及fixed20音频处理。
- audio126+SEP1+prompt256+answer300，总683 tokens；原answer-only CE/EOS/PAD规则，attention_mask=None。
- 8 GPU × microbatch8 × accumulation4 = GBS256，3epochs，每epoch5771updates，总17313、warmup866。
- DataLoader workers0；后台音频字节复制默认4workers，八卡在optimizer边界一致切换本地文件。
- 原保存节点、optimizer/scheduler/scaler/GradNorm、八rank RNG、数据游标及完整恢复。

## 正式训练提交

本地代码提交并推送GitHub后，在远程登录节点执行：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_split3_inject_5090.sh --stage-workers 4
```

直接提交formal，无自动smoke或tensor检查前置任务。
`pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`，沿用原镜像和 `mellow_c8204d8`。
登录节点仅执行原资产检查和CPU metadata索引；训练通过vc submit在计算节点执行。

默认输出root：
```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_inject_5090/formal_<timestamp>/
```

默认初始化仍来自原文本checkpoint，并非已训练split3音频checkpoint：
```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244
```
CED默认 `/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small`。
launcher保留 `--output-dir`、`--text-model-dir`、`--ced-model-dir`、
`--resolved-paths`、`--index-cache-root`、`--stage-workers` 覆盖选项。

## 诊断、产物和恢复

`formal/config_audit.json`记录架构/零gate/split3初始化合同、参数计数、分组、SHA和每来源真实音频probe。
probe为eval，之后恢复训练状态与RNG。
`formal/metrics.jsonl`中的 `audio_listen_rank0_last_microbatch` 沿用首两次及每25次update的诊断节奏，
记录rank0最后一个microbatch的六个router概率、memory/seed RMS，以及 `inject_layer4/8`：
`mean_gate`、`mean_abs_gate`、`max_abs_gate`、`saturation_fraction`（abs>0.95）、
`residual_rms`、`read_hidden_rms`、`residual_to_read_norm`。
槽位概率无硬性上限，训练后允许自然分化。

`first_gradient_rank<N>.json`检查所有可训练梯度存在/有限、冻结CED无梯度和层执行/注入轨迹。
`audio_gradient_metrics.jsonl`增加 `layer4_gate/layer8_gate` 的累积unscale后梯度，
与MLP/音频router同频记录。诊断只detach记录，不改变计算图和RNG。

输出root内最终产物：
```text
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/training_report.json
```

route身份为 `ced_small_mizar_s1_fixed20_gbs256_recursive_mesh_layer4_8_audio7slot_split3_inject_v1`。
只接受本路线同源码/资产/数据/训练合同的未完成完整checkpoint恢复。
恢复严格覆盖全部参数及训练状态，不重新清零gate或重新三等分router。
原split3、K7、48listen和inject路线的checkpoint均不能直接full resume至此版本。
未实现已训练split3音频checkpoint的权重迁移。completed checkpoint可审计但不能resume。

中断后替换为原输出root与实际已有step：
```bash
OUT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_inject_5090/formal_<实际timestamp>"
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_split3_inject_5090.sh \
  --output-dir "$OUT" --resume-checkpoint "$OUT/formal/checkpoints/model--step-2886.ckpt" --stage-workers 4
```

每个完整LM forward仍必须显式传入同一对A4/A8，无KV cache或跨样本音频状态缓存。
现有评测器尚未适配本路线，不能直接用于其checkpoint。

## 本地验证

仓库根目录：
```bash
python -B tests/validate_mizar_s1_mesh_ced_audio7slot_split3_inject.py
# 在具有原训练依赖的环境中，要求真实CPU tensor检查全部执行
python -B tests/validate_mizar_s1_mesh_ced_audio7slot_split3_inject.py --require-tensors
```
路线内 `python -B runtime_checks.py --require-tensors` 转发同一入口。
不下载资产、不用CUDA、不提交作业，不作为formal自动前置任务。

本地70项实际检查通过，Python/Bash语法通过；16项真实CPU tensor检查因缺
Torch/Transformers/Torchaudio/Soundfile明确跳过。
包含实际forward/gate代码的NumPy执行、零gate与原split3对照、
非零gate的post-read输入/边界/音频范围/普通memory写入核验、
完整训练循环AST对照、split3/K7/48listen/CED回归及模拟Bash提交/staging清理。
待依赖可用的真实CPU测试覆盖完整logits对齐、梯度、非零gate因果性、strict加载和恢复后下一次更新。

报告为 `audit_reports/mizar_s1_mesh_ced_audio7slot_split3_inject_validation.json`；
路线内 `reference_audit.json` 记录母本源码SHA及语义对照。
远程真实权重加载、8GPU正式训练与完整恢复尚未执行，不登记远程PASS。
