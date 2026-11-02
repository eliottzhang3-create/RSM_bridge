# CED-Small / 非 MeSH 5+10×2+5 / input embeddings → CED 8→4 串行注入

本路线完整复制隔离自同级
`mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject`，
在两次中段循环入口增加原 audio_reinject 路线的共享 input 投影/gate。
每个入口先注入 input embeddings，再用更新后的 hidden 计算中间层 gate。
原两条训练路线及共享骨干保持原文件。

本地 Windows 维护代码、测试和文档，通过 GitHub 同步；权重、数据与正式训练在远程 Linux。
本路线从共同的原文本 checkpoint 开始新的音频 S1 训练。

## 前向与串行顺序

冻结且 eval 的 CED-Small 每批一次编码，同时取得：
最终层特征、完整 block4 和 block8 输出（位于 CED 最终 norm 前）。
原特征整理、两套独立 MLP 和原 final-layer mapper 均保持 8→4 母本。

- 最终层经原 freq_merge_126 mapper 得到 E [B,126,576]，仍用作原输入前缀。
- input 分支：A_input = audio_linear(E)，共享 Linear(576,576)，每次完整 LM forward 计算一次。
- 中间层分支：A4/A8 [B,126,576]，各自使用独立的 LN1536 → Linear1536/576 → GELU → Dropout0 → Linear576/576 → LN576。
- input 使用一个共享 Linear(576,1) gate，两次根据当次 hidden 重算。
- 中间层使用独立 gate8 和 gate4，均为 Linear(576,1)。

| 入口 | 零基逻辑索引 | 第一步 | 第二步 |
|---|---:|---|---|
| 前五层后、第一轮中段循环前 | 5 | 共享 input 残差 | gate8 读取更新后的 hidden，注入 A8 |
| 第一轮中段循环后、第二轮前 | 15 | 同一个 input 投影/gate | gate4 读取更新后的 hidden，注入 A4 |

实际计算：

```python
A_input = audio_linear(inputs_embeds[:, :126])  # 每次 LM forward 一次，不 detach
# 在 logical_index=5、15：
h = hidden_states[:, :126]
h1 = h + tanh(audio_gate(h)) * A_input
h2 = h1 + tanh(audio_gates[str(layer)](h1)) * A_layer  # layer=8、4
hidden_states = cat((h2, hidden_states[:, 126:]), dim=1)
```

gate 为每个音频 token 一个可正可负的标量。
只有前126个音频位置直接更新，后续 SEP/prompt/answer 仍经过原因果注意力。
第二轮中段循环结束后无第三次注入。
无额外归一化、缩放、dropout 或残差 detach；input 信息来自原始 mapper E。
接口仍显式传入 audio_information=(A4,A8)，每次 forward 重置诊断和轨迹，不缓存跨样本特征。
仅支持完整音频前缀的 inputs_embeds、全序列重算和 use_cache=False。

实际串行轨迹为：

```text
[[5,"input"], [5,"layer8"], [15,"input"], [15,"layer4"]]
```

## 初始化与参数

严格加载原文本骨干并检查全部 missing/unexpected/mismatched/error 后，再注册两个中间层 gate、
共享 input 投影和共享 input gate，之后构建 optimizer/DDP。
中间层 MLP/gate 注册顺序仍为4、8；
input 模块使用另一个独立 CPU fork_rng(devices=[])，沿用原 audio_reinject 初始化函数。
原 mapper/LM 和对应新增分支的初值、调用者 CPU RNG 保持可比；不接触其他 CUDA generators。

| 新增部分 | 参数数 |
|---|---:|
| 两套中间层 MLP | 2,443,776 |
| gate4、gate8 | 1,154 |
| 共享 input Linear(576,576) | 332,352 |
| 共享 input gate | 577 |
| 相对普通 CED 循环母本合计 | 2,777,859 |

三个 gate 的 weight/bias 均为零；input 投影 weight 使用原 PyTorch Linear 默认初始化，bias=0。
初始 forward 与普通 CED 循环母本对齐。
首次 backward 应有 mapper、LM 和三个 gate 的非零梯度，input 投影及两套 MLP 的任务梯度为零；
第一次更新后 gate 打开，再检查三条上游分支均获得梯度。
AdamW 的解耦衰减可能在任务梯度为零时更新投影/MLP 矩阵，不能用权重变化替代任务梯度检查。

## 保留的训练合同

文本初始化：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244
```

只继承文本权重/tokenizer；音频 S1 的 optimizer、scheduler、GradScaler、GradNormTracker、RNG、数据游标从头开始。

- 20物理层、30次执行，轨迹0..14,5..14,15..19，无 MeSH memory/router。
- 原 hybrid 1,477,492行及 resolved_paths，保留全部数据源。
- 原 fixed20 音频；整段 mel/BN 后切分，CED 冻结且所有子模块 eval。
- audio126 + SEP1 + prompt256 + answer300 = 683，prefix383；PAD !/17，SEP/EOS0。
- attention_mask=None，原 answer mask + 非PAD目标；rank-local token mean、GA平均、DDP平均。
- FP32，无 autocast/TF32，保留原 GradScaler。
- 8 GPU × microbatch8 × accumulation4 = GBS256；3epochs。
- 5,771 updates/epoch，17,313 total，866 warmup，每轮丢弃116行。
- LM maxLR3e-4；mapper、两套 MLP、三个 gate、input 投影均 maxLR6e-4。
- 八类参数显式分组，新增模块优先于泛 LM 名称匹配，矩阵 decay0.01，bias/ndim≤1不衰减。
- 原 AdamW betas(.9,.999)/eps1e-8、LambdaLR、GradNormTracker、seed1234及采样/遍历。
- 共享原始音频立即开始训练，默认4线程后台复制，八卡在 optimizer 边界共同切换。
- 保存步骤2886、5771、8656、11542、14428、17313，以及最终 epoch3 checkpoint。

## 远程正式提交

代码经 GitHub 同步后，在服务器登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_5090.sh --stage-workers 4
```

直接通过 vc submit 提交 formal，无自动 smoke 或 tensor 前置任务。
使用 mellow_c8204d8，以及 pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node。
登录节点只检查资产、准备/复用 CPU metadata 索引并提交，训练在计算节点运行。

默认 CED 路径 /hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small；
manifest 为 /hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json；
resolved_paths 为
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv。
入口保留 --output-dir、--resume-checkpoint、--resolved-paths、--text-model-dir、
--ced-model-dir、--index-cache-root、--stage-workers 参数。

## 产物与完整恢复

新 route：

```text
ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_v1
```

默认输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_5090/formal_<timestamp>/
```

该 run root 内保存：

```text
formal/runtime.json
formal/config_audit.json
formal/trainable_params.txt
formal/first_gradient_rank<N>.json
formal/opened_gate_gradient_rank<N>.json
formal/staging_report.json
formal/metrics.jsonl
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/training_report.json
```

输出、日志、job名、CPU index cache、/dev/shm tmpfs 及恢复身份独立。
runtime 同时记录 layer_injection、audio_reinjection、serial_audio_injection；
串行顺序和中间 gate 读取 post-input hidden 都纳入合同哈希。
前两次及每25次更新记录两类 gate 幅度、饱和比例、各自残差/hidden范数比和完整串行轨迹。
注入残差诊断以各分支自己的输入 hidden 为分母；不改变 loss 或增加正则。

只接受本组合路线、相同代码/数据/资产/配方的完整 formal checkpoint。
原8→4、audio_reinject、4→8或 MeSH checkpoint 均不能 full resume。
完整恢复包含全部模型分支、optimizer、scheduler、GradScaler、GradNormTracker、
八rank RNG和 epoch/update 游标，不清零已训练 gate 或重置投影。

将 RUN_ROOT 换成需要恢复的本组合路线 run root：

```bash
RUN_ROOT=/absolute/path/to/existing/combined_route_run
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_5090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

## 验证与状态

仓库根目录统一验证：

```bash
python -B tests/validate_mizar_s1_ced_layer84_audio_reinject.py
```

本地实际通过109项检查；37项真实CPU tensor检查因缺 torch/torchaudio/transformers/soundfile 跳过，
其中本新路线12项、原8→4 8项、原 audio_reinject 7项、原4→8 7项、普通 CED循环3项。
Python编译、Bash语法、非零gate串行数值/有限差分、两路关闭对照、三gate梯度审计失败用例、
Bash模拟提交和原路线回归通过。
报告为 audit_reports/mizar_s1_ced_layer84_audio_reinject_validation.json。

源码审计逐文件保护137个原文件（LF归一化，适用于Windows/Linux Git checkout）。
完整复制26个Python/Bash源文件，其中18个除隔离命名外相同；
去掉明确新增 input 注入后，完整递归模型 AST 与8→4母本一致；
去掉新增合同/诊断后，完整 train 函数包括数据、优化器更新、保存和恢复均与母本一致。
input 分支三个方法与原 audio_reinject AST相同。
reference_audit.json 由以下命令生成：

```bash
python code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject/audit_reference.py
```

具备原依赖的服务器环境可手动执行严格CPU张量检查：

```bash
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate mellow_c8204d8
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
python -B tests/validate_mizar_s1_ced_layer84_audio_reinject.py --require-tensors
```

该检查使用合成输入，不下载资产、不运行CUDA、不提交作业；
--require-tensors 在缺依赖或有skip时返回非零，属于独立可选检查。
真实张量检查覆盖串行gate输入/共享A、零gate输出和继承参数梯度、
单分支关闭等价、严格from_pretrained逐tensor继承、RNG/初值、参数量和checkpoint恢复后的下一次更新。
CPU checkpoint测试中的collective/GradScaler使用明确fixture，不代表八卡CUDA恢复验证。

尚未执行远程真实权重加载、八卡训练和完整恢复，未登记远程PASS。
现有评测器需为本组合架构单独适配。
