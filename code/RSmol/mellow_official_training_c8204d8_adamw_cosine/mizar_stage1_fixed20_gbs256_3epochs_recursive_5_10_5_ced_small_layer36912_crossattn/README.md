# CED 第 3/6/9/12 层选择与两轮共享 cross attention

2026-10-30：当前为 **v2 分支归一化与固定 0.1 注入尺度版本**。
用户反馈 v1 真实训练可以启动，但观察到注意力残差放大、梯度爆炸。
本次仅修改交互分支架构，保留原训练配方；新版本远程稳定性尚待验证。
v1 音频 checkpoint 不能直接作为 v2 的 full resume；从原文本初始化重新训练，使用新的空输出目录。

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
→ 独立 LayerNorm(1536)
→ 独立 Linear(1536,576)
→ 独立 LayerNorm(576)
→ ai: [B,126,576]，i=1,2,3,4 对应 CED 层3,6,9,12
```

仅合并频率与通道，不平均时间/频率，不增加激活、dropout 或 FFN。
每个深度独立拥有投影前后的两个 LayerNorm 和投影，均位于 CED 的 no_grad 范围之外。
每次 LM forward 各音频分支只执行一次，结果供两次交互复用。

### 两次文本交互

原骨干保持20个独立物理层、30次逻辑执行、精确轨迹
`0..14,5..14,15..19`，没有 MeSH memory 或 router。
前五层输出 h0，在零基逻辑索引5前执行第一轮交互；
第一次中段循环输出 h1，在逻辑索引15前执行第二轮交互，随后进入第二次中段循环。

对当前状态 H，令 T=H[:,127:]：

```text
U = LN_text(T)                                    # only the interaction branch
s = masked_mean(U[:,:256], tokenizer_prompt_attention_mask)
alpha = softmax(Gate1(s))                          # [B,4]
A = sum_i alpha_i * ai                             # [B,126,576]
Z = CrossAttention(Q_input=U+b_round, K_input=A, V_input=A)
Z_hat = LN_out(Z)                                 # elementwise_affine=False
g = tanh(Gate2(U))                                # [B,text_length,1]
residual = 0.1 * g * Z_hat                         # fixed, non-trainable scale
T_new = T + residual                              # original T is the residual base
```

| 模块 | 共享与初始化 |
|---|---|
| 四个 Linear(1536,576) | 深度之间独立；Xavier uniform weight、零 bias；两轮复用结果 |
| 四个输入 LN1536 + 四个输出 LN576 | 各深度独立；eps1e-5；可训练 weight=1、bias=0；两轮复用结果 |
| LN_text(576) | 两轮共享；eps1e-5；可训练 weight=1、bias=0；仅用于 Gate1/Gate2/Q 分支 |
| LN_out(576) | 两轮共享；eps1e-5；elementwise_affine=False，无可训练参数 |
| Gate1: Linear(576,4)+softmax | 两轮共享；weight/bias全零，初始四层权重均为0.25 |
| Gate2: Linear(576,1)+tanh | 两轮共享；weight/bias全零，每文本 token 一个标量门控 |
| MultiheadAttention | 两轮共享；hidden576、9 heads、head_dim64、dropout0、带bias |
| b1、b2 | 两个独立可训练的576维零初值向量，仅加入对应轮次的Q输入 |

Cross attention 权重按 PyTorch MultiheadAttention 初始化，bias全零；输出投影 weight 不做零初始化。
轮次偏置在文本 LN 之后加入，不进入 Gate1、Gate2、K、V 或残差基底。
固定 γ=0.1，不加入 optimizer。v2 交互模块共 **4,892,741** 个参数，较 v1 多 **18,048** 个 LN 参数。

LN 仅沿每个 token 的最后一维计算，不混合 token/batch；不将 U 写回主路径。
在数值有限时，非仿射输出 LN 使每 token 的输出 RMS 不超过约1，结合逐 token 标量
`|g|≤1`，单次实际注入残差每 token RMS 不超过约0.1（允许浮点误差）。
这是绝对幅度约束，不表示主路径的10%，也不是对全部训练梯度的稳定性证明。

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
LN 注册在原交互模块初始化之后，采用常量初始化，不消耗额外随机数。

Gate2初始残差严格为零。首次反向允许投影、Gate1、cross attention和轮次偏置梯度为零，
但梯度必须已连接且有限；mapper、原LM、Gate2必须有非零梯度。
Gate2更新打开后，逐一检查四个投影、八个音频LN、文本LN和两个轮次偏置的非零梯度。
每个训练段最多等待32次更新；未全部连通会失败，并保存各rank证据。

| 参数 | max LR |
|---|---:|
| 原LM全部参数 | 3e-4 |
| 原CED mapper | 6e-4 |
| 四个投影、九个可训练LN、Gate1、Gate2、cross attention、b1/b2 | 6e-4 |

保留 AdamW betas=(0.9,0.999)、eps=1e-8、WD0.01，bias及ndim≤1参数不衰减，
包括两个一维轮次偏置和全部一维LN参数。保留母本 LambdaLR 和 GradNormTracker。
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

不传 `--resume-checkpoint`，默认使用新的时间戳输出目录，从原文本 checkpoint 开始新的 v2 训练。
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
metrics按指定记录步保存rank0最后一个microbatch的两轮层权重、gate均值/绝对值，以及：

- `cross_attention_rms`：输出LN之前的原始CA输出RMS。
- `normalized_cross_attention_rms`：非仿射输出LN之后的RMS。
- `residual_rms`：包含γ=0.1和gate的实际注入RMS。
- `residual_token_rms_max`：所有文本token实际注入RMS的最大值，应不超过约0.1。
- `residual_scale`：固定0.1。

这些是rank0最后一个microbatch的训练诊断，不是所有rank的稳定性或验证集指标。

仅接受本路线未完成的完整formal checkpoint，route为
`ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer36912_crossattn_v2`。
恢复核对代码/资产/数据/配方、完整状态及游标；
拒绝v1未归一化交互、母本、4→8、8→4、MeSH或其他架构checkpoint。
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

2026-10-30 v2本地验证：**58项实际检查通过，15项真实CPU tensor检查跳过**。
缺少torch、torchaudio、transformers、soundfile；NumPy行为检查不是autograd证明。

通过范围包括实际方法的NumPy行为/因果性、放大CA输出与饱和门控下的逐token残差上限、
原文本残差基底保持、strict loader拒绝条件、LN/γ结构审计、参数inventory/LR分组、
源码/AST母本对照、Python编译、Bash语法与fresh/resume模拟提交、
本路线共享音频立即启动与producer退出清理，以及母本回归。
新路线12项及母本3项真实CPU tensor检查未执行，包括零门控logits/原参数梯度等价、
开门后新增LN梯度、真实tensor逐token残差上限和完整checkpoint continuation。
历史v1在2026-10-29的记录为56项实际检查通过、14项tensor检查跳过。

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

用户已确认v1训练可运行但发生梯度爆炸；v2真实权重加载、显存、吞吐、
8 GPU训练稳定性及恢复仍待远程验证，不登记为v2远程PASS。
正式作业保留真实来源探针、交互轨迹、首步和开门后梯度检查、
冻结CED不变检查与最终checkpoint重读审计。
原有普通CED评测入口尚未适配本新架构。

官方block→final norm次序参考
[CED源码](https://huggingface.co/mispeech/ced-small/blob/main/modeling_ced.py)；
实际实验以远程本地资产与本路线源码合同为准。
