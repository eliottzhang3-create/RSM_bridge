# CED 4/8 初始记忆 / MeSH audio7slot / Mizar S1 GBS256

本目录完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_listen`。
本地修改代码，经 GitHub 同步到远程 Linux；真实权重、音频、数据和 GPU 运行均在远程。
原 CED MeSH、48listen 和共享 `recursive_model_5_10x2_5_mesh.py` 未修改。

## 架构

保留 hidden576、20 物理层、30 次逻辑执行 `0..14,5..14,15..19`、两次中段循环。
完整序列仍是 audio126 + SEP1 + prompt256 + answer300，共683 tokens。
Transformer 始终处理完整序列并保留原因果注意力；仅 memory/router 按位置区分：

| 范围 | 槽数 | 初始化（在 prelude 前） | router |
|---|---:|---|---|
| 前126个 audio tokens | 7 | `[input_embedding,0,0,0,0,A4,A8]` | 新增独立3 write + 3 read |
| SEP、prompt、answer（包括 PAD） | 5 | `[input_embedding,0,0,0,0]` | 原3 write + 3 read |

冻结 CED 一次前向提取第4/8个完整 block 输出，位于最终 LN 之前。
沿用48listen的分段/时间/频率顺序 `[B,504,384] → [B,2,4,63,384] → [B,126,4,384]`，
经两套独立可训练 MLP：`LN1536 → Linear1536/576 → GELU → Dropout0 → Linear576/576 → LN576`，
得到 A4/A8，在原 slot0 用 input embedding 初始化的同时分别放入零基 slot5/6。
不保留旧 tanh gates 或两套五路 listen routers，后续不再次注入 encoder 信息。
七个槽都参与后续普通 additive write 和 weighted read；A4/A8 槽不是只读槽。

三次交互仍为 prelude 后、循环0后、循环1后。Prelude query/value 都是 prelude 输出；
两次循环的 query 是各自循环输入、write value 是循环输出。
先向每槽加 `p_write * value`，再以 `p_read` 加权求和，无额外 residual/归一化/缩放。
读取后拼接 audio/text hidden，再执行原完整 Transformer stack。
每次完整 forward 重建 memory；无 KV cache，推理必须在每次调用显式传 `(A4,A8)`。
“文本保持不变”指五槽机制和原 router；注意力仍可感知改变后的音频状态。

## K=7 初始化与学习率

先 strict 加载完整五槽文本 checkpoint，确认六个原 router 和全部原 LM 参数完整加载后，
在隔离 CPU RNG 的上下文中注册六个 `Linear(576,7)`：

```python
audio.weight[:5] = corresponding_original.weight
audio.bias[:5] = corresponding_original.bias
audio.weight[5:] = 0
audio.bias[5:] = -7
```

每个音频 router 对应同一阶段、同一 write/read 类型的原 router；参数独立，无共享 storage。
原五槽 checkpoint 字段 `mesh_memory_slots=5` 保留，runtime 另外声明 audio7/text5。
新增两套 MLP 2,443,776 参数、六套音频 router 24,234 参数，合计新增 **2,468,010**。

| 参数 | max LR | decay |
|---|---:|---|
| 原 LM（含原六个文本 router） | 3e-4 | 矩阵0.01，bias/ndim≤1为0 |
| 原最终层 freq_merge_126 mapper | 6e-4 | 同上 |
| 两套 A4/A8 MLP、六个音频 router（全部行） | 6e-4 | 同上 |

K=7 是 fresh 初始化常数，新增行和 bias 正常参与训练，没有固定概率约束或额外退火。
用户十条文本标定中最坏位置在 K=7.2 时两槽合计约0.170621%，换算 K=7 约0.208318%。
这是原文本 query 的离线反事实结果，实际音频及改变后的后续循环需要观察真实诊断。
即便 A4=A8=0，有限 K 的七路 softmax 和普通 write 也会改变行为，不能视作原模型精确复原。

## 保持原训练配方

- hybrid 1,477,492行；全部 MusicCaps / SpeechCraft splits、原 shuffle/rank RNG/连续分片。
- 单录音全通道均值、32kHz fixed20 中心裁剪/右补零；完整20秒经16kHz mel、逐样本 dB、冻结 BN 后分段。
- CED 冻结且始终 eval；最终层官方 mapper/dropout0.05 不变。
- 8 GPU × microbatch8 × accumulation4 = GBS256；3 epochs；5771 updates/epoch；总17313；warmup866；每 epoch 丢弃116行。
- 原 answer-only CE、EOS/PAD 规则、attention_mask=None、labels=None、use_cache=False。
- AdamW betas(.9,.999)/eps1e-8，LambdaLR 5%warmup+cosine/floor0.1，原 GradNormTracker/GradScaler。
- FP32 forward，无 autocast/TF32，DataLoader workers0；后台音频字节复制，全部完成后八卡一致在 optimizer 边界切本地。
- 保存节点、全量 optimizer/scheduler/scaler/GradNorm、八 rank RNG、数据游标和完整恢复流程保留。

## 正式提交

先将本地改动提交/推送到 GitHub，然后在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_5090.sh --stage-workers 4
```

直接提交 formal，无自动 smoke/tensor 检查前置任务。沿用
`pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`、原镜像和 `mellow_c8204d8`。
登录节点只做原有资产检查和 CPU metadata 索引；GPU 训练由 `vc submit` 作业执行。

默认输出 root：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_5090/formal_<timestamp>/
```

输出、日志、index cache、tmpfs 前缀和 resume route 身份均独立。
`--output-dir` 指这个 root，内部再生成 `formal/`，可显式指定：

```bash
OUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_5090/formal_$(date +%Y%m%d_%H%M%S%N)
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_5090.sh \
  --stage-workers 4 --output-dir "$OUT"
```

默认文本初始化源不变：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244
```

CED 默认 `/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small`。
launcher 支持原 `--text-model-dir`、`--ced-model-dir`、`--resolved-paths`、`--index-cache-root` 覆盖。

## 记录与恢复

`formal/config_audit.json` 保存结构、完整 optimizer 参数分组、初始 SHA 和每来源真实音频 probe。
probe 运行 eval 后恢复训练状态和 RNG；保存实际七槽概率、A4/A8 RMS 及直接 seed read 的 RMS。
训练首两次 update 和每25次记录六个音频 router 的7槽均值、两个新槽各自最大值、合计均值/最大值、
合计超过0.5%/1%的比例、熵、旧五路 logsumexp 和 memory/read RMS。
概率单位是 **fraction**（0.005=0.5%），均为 rank0 最后一个 microbatch。
这些是诊断，不会因概率超过阈值而自动改 K、裁剪概率或阻止正常学习。

`first_gradient_rank<N>.json` 检查全部训练参数梯度存在/有限、冻结 CED 无梯度及30层轨迹。
`audio_gradient_metrics.jsonl` 记录首两次/每25次累积且 unscale 后的新增分支及新两行梯度。
不再使用零 gate 的解锁期限；真实样本某梯度为零只记录，不要求每个参数每个样本都非零。
真实合成 CPU 测试另行验证 MLP、所有新 router 新行的非零梯度路径。

最终完整产物：

```text
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/training_report.json
```

checkpoint 增加 `audio_gradient_state` 的检查次数/游标；route 身份为
`ced_small_mizar_s1_fixed20_gbs256_recursive_mesh_layer4_8_audio7slot_v1`。
只允许相同本路线源码、资产、数据、训练合同的未完成完整 checkpoint 恢复。
恢复先构建模型，再 strict 覆盖完整训练 state，不重置已训练的新增行/bias。
例如中断后（替换为实际原 root 和中间 step）：

```bash
OUT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_5090/formal_<实际timestamp>"
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_5090.sh \
  --output-dir "$OUT" --resume-checkpoint "$OUT/formal/checkpoints/model--step-2886.ckpt" --stage-workers 4
```

completed checkpoint 可严格审计但不能 resume。原五槽 CED/48listen checkpoint 不能 full resume 到本路线。
现有评测器也不能直接加载本架构，需后续增加独立适配。

## 验证范围

从仓库根目录可手动运行：

```bash
python -B tests/validate_mizar_s1_mesh_ced_audio7slot.py
# 在具备原训练依赖的环境中，要求真实 CPU tensor 检查不允许 skip：
python -B tests/validate_mizar_s1_mesh_ced_audio7slot.py --require-tensors
```

路线内 `python -B runtime_checks.py --require-tensors` 转发到同一入口。
不下载资产、不运行 CUDA、不提交作业，也不作为 formal 的自动前置任务。
包含实际 memory 代码的 NumPy 公式对照、源码 AST 训练循环对照、Bash 模拟提交/失败清理、
严格恢复合同、可选真实 CPU autograd/strict text loading/恢复后下一次更新对照。
本地验证报告为 `audit_reports/mizar_s1_mesh_ced_audio7slot_validation.json`。
本地缺 Torch/Transformers/Torchaudio/Soundfile，tensor 项会明确跳过；本地检查不等于远程真实权重、8 GPU 训练或恢复通过。
本次实际结果：34项通过，12项真实CPU tensor检查跳过，Python/Bash语法通过。
