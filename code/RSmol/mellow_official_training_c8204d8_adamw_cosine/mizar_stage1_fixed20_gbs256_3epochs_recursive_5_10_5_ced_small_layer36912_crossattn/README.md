# CED 第 3/6/9/12 层选择与两轮共享 cross attention

本路线完整复制隔离自
`mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small`，
即 `run_mizar_s1_gbs256_recursive_5_10_5_ced_small_5090.sh` 对应的非 MeSH 母本。
保留原 CED 最终层、官方 mapper 和递归 LM 主路径，额外在两次中段循环前注入音频信息。

本地 Windows 维护代码、测试和文档；权重、数据、日志和正式训练在远程 Linux，
通过 GitHub 同步代码。本地交付不代表真实权重加载或 8 GPU 训练已经通过。

## 1. 架构

### 音频分支

冻结 CED-Small，始终 eval。在同一次 `forward_features` 中用只读 hooks 取得
第3/6/9/12个 block 的输出（零基2/5/8/11），统一取最终 LayerNorm **之前**的状态。
hooks 不替换输出，正常结束或异常退出都会移除。
原最终层仍照常经过最终 LayerNorm 和原 mapper，生成输入最前面的126个音频 tokens。

整段 fixed20 波形、重采样、mel、逐样本 top_db120、冻结 BN 和1012帧分段均沿用母本。
各层分别保留 segment/frequency/time 顺序：

```text
[2B,252,384]
→ [B,2 segments,4 frequency bins,63 time steps,384]
→ [B,2 segments,63 time steps,4 frequency bins,384]
→ [B,126,1536]
→ 独立 Linear(1536,576)
→ ai: [B,126,576]，i=1,2,3,4 对应 CED 层3,6,9,12
```

仅合并频率与通道，不平均时间/频率，不增加分支归一化、激活、dropout 或 FFN。
四个投影参数独立，位于 CED 的 no_grad 范围之外。
每次 LM forward 各投影只执行一次，结果供两次交互复用。

### 两次文本交互

原骨干保持20个独立物理层、30次逻辑执行、精确轨迹
`0..14,5..14,15..19`，没有 MeSH memory 或 router。
前五层输出 h0，在零基逻辑索引5前执行第一轮交互；
第一次中段循环输出 h1，在逻辑索引15前执行第二轮交互，随后进入第二次中段循环。

对当前状态 H，令 T=H[:,127:]：

```text
s = masked_mean(H[:,127:383], tokenizer_prompt_attention_mask)
alpha = softmax(Gate1(s))                          # [B,4]
A = sum_i alpha_i * ai                             # [B,126,576]
Z = CrossAttention(Q_input=T+b_round, K_input=A, V_input=A)
g = tanh(Gate2(T))                                 # [B,text_length,1]
T_new = T + g * Z
```

| 模块 | 共享与初始化 |
|---|---|
| 四个 Linear(1536,576) | 深度之间独立；Xavier uniform weight、零 bias；两轮复用结果 |
| Gate1: Linear(576,4)+softmax | 两轮共享；weight/bias全零，初始四层权重均为0.25 |
| Gate2: Linear(576,1)+tanh | 两轮共享；weight/bias全零，每文本 token 一个标量门控 |
| MultiheadAttention | 两轮共享；hidden576、9 heads、head_dim64、dropout0、带bias |
| b1、b2 | 两个独立可训练的576维零初值向量，仅加入对应轮次的Q输入 |

Cross attention 权重按 PyTorch MultiheadAttention 初始化，bias全零。
轮次偏置不进入 Gate1、Gate2、K、V 或残差基底。新增参数共 **4,874,693**。

残差覆盖 prompt、answer 和母本原有的可见 PAD；前126个音频位置及SEP不直接注入。
Gate1 只汇总 tokenizer prompt attention_mask 标记的有效位置，排除 answer 和直接池化的PAD，
不通过 token ID 猜测 PAD。此 mask 仅用于池化，LM仍保持 `attention_mask=None`。
Gate1 允许 prompt 位置共享完整已知问题摘要，不通过答案摘要泄露未来 answer。

推理保留完整256-token prompt及相同mask，每个生成步骤完整重算，`use_cache=False`；
模型拒绝KV cache。`Mellow.generate_prefix_inference` 返回
`(prefix, audio_output, context)`，context的 `audio_grids`、`prompt_mask`
必须传给每次LM forward。现有普通CED evaluator尚未适配此架构，本次交付范围为正式训练。

## 2. 初始化、梯度与学习率

原 CED/mapper 构造顺序不变；严格加载并审计原非 MeSH 文本权重后才注册新增模块。
新增模块在 CPU RNG fork 内初始化并恢复调用方RNG，不改变母本原参数的初始化随机状态。

Gate2初始残差严格为零。首次反向允许投影、Gate1、cross attention和轮次偏置梯度为零，
但梯度必须已连接且有限；mapper、原LM、Gate2必须有非零梯度。
Gate2更新打开后，逐一检查四个投影和两个轮次偏置的非零梯度。
每个训练段最多等待32次更新；未全部连通会失败，并保存各rank证据。

| 参数 | max LR |
|---|---:|
| 原LM全部参数 | 3e-4 |
| 原CED mapper | 6e-4 |
| 四个投影、Gate1、Gate2、cross attention、b1/b2 | 6e-4 |

保留 AdamW betas=(0.9,0.999)、eps=1e-8、WD0.01，bias及ndim≤1参数不衰减，
包括两个一维轮次偏置。保留母本 LambdaLR 和 GradNormTracker。
调度首步系数为0.1，首步实际LR为LM 3e-5、mapper/新增参数6e-5，
随后按原5% warmup/cosine调度。

## 3. 保留的训练合同

- hybrid manifest共1,477,492行，包含MusicCaps与三个SpeechCraft splits。
- 原在线fixed20解码；直接读共享音频开始formal，后台默认4线程复制，
  全部完成后在optimizer update边界八卡MIN all-reduce确认并切换到本地副本。
- `126 audio + 1 SEP + 256 prompt + 300 answer = 683`，prefix383。
- 保留母本Mizar文本构造、EOS追加/截断、固定padding、可见PAD和answer-only CE。
  loss为每rank每microbatch有效answer token mean，再按GA和DDP平均，
  不改成全局有效token加权平均。
- FP32前反向，无autocast，关闭TF32，保留GradScaler与GradNormTracker。
- 8 GPU × microbatch8 × accumulation4 = GBS256，3 epochs。
- 每epoch 5,771 updates，共17,313 updates，866 warmup，每epoch丢弃116行。
- seed1234、各rank RNG、数据顺序和sampler游标沿用母本。

入口、输出、日志、CPU index cache、tmpfs和resume身份独立。
母本及共享 `recursive_model_5_10_5.py` 未修改。
`audit_parent.py` 对原mapper/预处理、数据工具、完整epoch/update循环和共享骨干做源码/AST对照。

默认远程资产：

```text
text_model=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244
ced_model=/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small
manifest=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
resolved_paths=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
```

## 4. 正式提交

将本地新增路线、入口、测试和文档加入Git并经GitHub同步后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer36912_crossattn_5090.sh --stage-workers 4
```

直接提交formal，不自动执行smoke、runtime_checks或tensor验证，不要求历史smoke report。
GPU由wrapper的 `vc submit` 提交，不在登录节点直接运行CUDA。
资源为 `pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`；
沿用母本容器及 `mellow_c8204d8` 环境，不要求升级依赖。
提交前的CPU索引准备/复用不申请GPU。

支持 `--output-dir`、`--resume-checkpoint`、`--resolved-paths`、
`--text-model-dir`、`--ced-model-dir`、`--index-cache-root`、`--stage-workers`。
fresh输出目录必须为空，默认时间戳run root为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
mizar_s1_ced_small_gbs256_recursive_5_10_5_layer36912_crossattn_5090/formal_<timestamp>/
```

## 5. 产物与恢复

上述run root内：

```text
formal/runtime.json
formal/config_audit.json
formal/trainable_params.txt
formal/first_gradient_rank<N>.json
formal/opened_gate_gradient_rank<N>.json
formal/staging_report.json
formal/metrics.jsonl
formal/training_report.json
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
```

中途保存步数为2886、5771、8656、11542、14428、17313。
完整checkpoint包含CED、原mapper、LM及全部新增模块、optimizer、scheduler、
GradScaler、GradNormTracker、8 rank RNG及epoch/update游标。
metrics按指定记录步保存rank0最后一个microbatch的两轮层权重、gate均值/绝对值、
cross attention RMS及残差RMS；这些是训练诊断，不是验证集指标。

仅接受本路线未完成的完整formal checkpoint，route为
`ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer36912_crossattn_v1`。
恢复核对代码/资产/数据/配方、完整状态及游标；
拒绝母本、4→8、8→4、MeSH或其他架构checkpoint。
仍需原始文本和CED资产；job/node/tmpfs路径不作为持久恢复身份。

在远程 `code/RSmol` 目录设置实际run root和已保存step后执行：

```bash
RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer36912_crossattn_5090/formal_YYYYMMDD_HHMMSS
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer36912_crossattn_5090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

用实际时间戳与未完成checkpoint替换示例；复用原run root时追加metrics，
并重新建立本作业临时staging。

## 6. 验证状态

2026-10-29本地重跑：**56项实际检查通过，14项真实CPU tensor检查跳过**。
缺少torch、torchaudio、transformers、soundfile；NumPy行为检查不是autograd证明。

通过范围包括实际方法的NumPy行为/因果性、strict loader拒绝条件、参数inventory/LR分组、
源码/AST母本对照、Python编译、Bash语法与fresh/resume模拟提交、
本路线共享音频立即启动与producer退出清理，以及母本回归。
新路线11项及母本3项真实CPU tensor检查未执行；
门控开门时点断言已按母本非零首步LR修正，修正本身不算执行通过。

从仓库根目录汇总：

```bash
python tests/validate_mizar_s1_ced_multilayer_crossattn.py --output audit_reports/mizar_s1_ced_multilayer_crossattn_validation.json
```

可在远程项目依赖环境手动补齐真实CPU计算：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
conda activate mellow_c8204d8
python tests/validate_mizar_s1_ced_multilayer_crossattn.py --require-tensors
```

此命令使用合成小模型，不下载资产、不调用CUDA、不提交训练；
依赖缺失、跳过或失败时 `--require-tensors` 返回失败。
checkpoint tensor测试模拟八卡collective和CUDA RNG接口，
即使通过也只证明CPU continuation，不作为真实8 GPU恢复验收。

真实权重加载、显存、吞吐、8 GPU训练/恢复仍待远程验证，未登记远程PASS。
正式作业保留真实来源探针、交互轨迹、首步和开门后梯度检查、
冻结CED不变检查与最终checkpoint重读审计。
原有普通CED评测入口尚未适配本新架构。

官方block→final norm次序参考
[CED源码](https://huggingface.co/mispeech/ced-small/blob/main/modeling_ced.py)；
实际实验以远程本地资产与本路线源码合同为准。
