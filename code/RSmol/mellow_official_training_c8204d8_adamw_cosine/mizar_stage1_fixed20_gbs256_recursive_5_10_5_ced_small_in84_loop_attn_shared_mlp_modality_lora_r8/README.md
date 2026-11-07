# 非 MeSH CED8→4：Attention共享、Transformer MLP模态独立、两轮独立LoRA

完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8`。
只改变中间十层Transformer投影的LoRA模态共享规则及对应训练审计；完整骨干、音频注入和训练更新流程按母版保留。
本地Windows仅维护代码/测试/文档，GitHub中转，远程Linux保存权重/数据并执行真实训练。
内部目录使用 `in84` 缩写以兼容Windows路径长度；正式launcher保留完整架构名称。

## 参数划分与前向

20个物理层：prelude0..4，中段5..14共享基础权重循环两次，suffix15..19。
完整执行轨迹为 `0..14,5..14,15..19`，共30次；无MeSH、memory或router。
前五层、后五层、embedding、各norm及LM head沿母版全量训练且不加LoRA。

| 中间层投影 | 同一轮Audio/Text | 两轮之间 |
|---|---|---|
| q_proj | 共享该投影一对A/B | 独立 |
| k_proj | 共享该投影一对A/B | 独立 |
| v_proj | 共享该投影一对A/B | 独立 |
| o_proj | 共享该投影一对A/B | 独立 |
| gate_proj | Audio/Text分别独立A/B | 独立 |
| up_proj | Audio/Text分别独立A/B | 独立 |
| down_proj | Audio/Text分别独立A/B | 独立 |

共享仅指同层、同投影、同轮的Attention音频/文本位置；不同层、不同投影、不同轮均有自己的LoRA。
所有LoRA rank8、alpha16、scale2、dropout0。基础W/bias仍由两种模态及两轮共享，始终全量训练。
Transformer MLP以外的CED mapper、中间特征MLP、input projection及gates均保持原样，不额外包LoRA。

Attention对完整序列计算：

```text
y = Linear(x, W, bias) + 2 * Linear(Linear(x, A_shared[loop]), B_shared[loop])
```

三个Transformer MLP投影各自计算：

```text
base = Linear(x, W, bias)
audio_delta = Linear(Linear(x[:, :126], A_audio[loop]), B_audio[loop])
text_delta  = Linear(Linear(x[:, 126:], A_text[loop]), B_text[loop])
y = base + 2 * concat(audio_delta, text_delta, dim=1)
```

每个基础投影对完整序列只计算一次。位置0..125为audio，126起为text，包括SEP、prompt、answer/PAD及生成位置。
MLP的gate/up/down均保持原token轴顺序，以同一位置边界选择分支；不使用学习得到的模态router。
完整序列causal self-attention、position IDs、RoPE及原SwiGLU乘积保持，模态独立残差不隔断注意力联系。

state_dict名称示例：

```text
caption_decoder.lm.model.layers.5.self_attn.q_proj.lora_A.shared.0
caption_decoder.lm.model.layers.5.mlp.gate_proj.lora_A.audio.0
caption_decoder.lm.model.layers.5.mlp.gate_proj.lora_A.text.0
# lora_B同理；最后索引0/1分别为两轮。
```

严格加载原文本LM全部权重后，注册音频gates、input分支及LoRA；保留原W/bias对象与基础checkpoint键。
A使用Kaiming uniform(a=sqrt5)，B全零，LoRA初始化在独立CPU fork_rng中，不扰动原主干/mapper/音频分支及调用者RNG。
无adapter merge、无weight.data修改、无循环间requires_grad切换。每次层调用使用显式轮次，finally复位；反向保留各自前向选中的参数。
沿母版use_cache=False、全序列重算，拒绝KV cache和decoder gradient checkpointing。

## 参数量

按母版hidden576、MLP1536、Q9/KV3 heads（K/V输出192）的投影形状：

| 范围（十层、两轮合计） | LoRA参数 |
|---|---:|
| Attention shared | 614,400 |
| Transformer MLP audio | 1,013,760 |
| Transformer MLP text | 1,013,760 |
| 合计 | 2,641,920 |

每层每轮4+3×2=10对A/B，共200对、400份独立参数张量；原版LoRA1,628,160，新版多1,013,760。
基础权重不重复计数。原音频串行分支2,777,859保持，两者相对普通CED循环母本共新增5,419,779。
正式trainer校验实际投影形状、400份独立存储及三类实际参数数；不以文件大小估算。
每个token每轮在每个目标投影只使用一个rank8残差。完整多模态forward会使用所有注册LoRA参数。

## 保留的音频及正式配方

```text
前五层 → audio input残差 → CED8残差 → 第一次中段循环
       → audio input残差 → CED4残差 → 第二次中段循环 → 最后五层
```

CED-small冻结/eval，单次编码提取final及完整block4/block8的最终norm前输出。
final经原freq_merge_126 mapper得到E，原两份独立MLP得到A4/A8。
每个完整LM forward只计算一次R=audio_linear(E)，每次先由当前audio hidden计算共享input gate并加R，
再由post-input hidden计算对应CED gate并加A8/A4；音频残差只直接更新前126位置，loop2后不再注入。
原mapper dropout0.05、CED4/8 MLP dropout0、三个gate零初始化及input projection默认weight/零bias保持。

文本初始化保持：
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244`。
仅继承模型/tokenizer，新建音频训练状态和全部LoRA；不加载原音频LoRA checkpoint作完整恢复。

| 项目 | 沿母版配置 |
|---|---|
| 数据/输入 | hybrid1,477,492行；fixed20原始音频在线处理 |
| 布局 | audio126+SEP1+prompt256+answer300=683，prefix383 |
| 监督 | 原answer-only CE，rank-local microbatch token mean→GA均值→DDP均值 |
| PAD/EOS/attention | PAD !/17，EOS/SEP0，attention_mask=None |
| 精度 | FP32，无autocast，TF32关闭，原GradScaler |
| Batch | 8GPU×microbatch8×GA4=GBS256 |
| 更新/调度 | 3epochs，5771/epoch，17313total，866warmup，每epoch丢116行 |
| maxLR | 基础LM3e-4；所有LoRA、mapper及音频新增模块6e-4 |
| 优化器 | 原AdamW betas(.9,.999)/eps1e-8；矩阵decay.01，bias/ndim≤1无衰减 |
| 其他 | 原LambdaLR、GradNormTracker、seed1234/rank seed1234+rank及数据顺序 |
| staging | 共享路径立即开训，默认4线程后台复制；全部完成后optimizer边界八卡一致切换 |
| 资源 | mellow_c8204d8，pdgpu-5090/32CPU/256G/8GPU/1node，原容器 |

优化器共20类角色：8类原角色及两轮×Attention shared/MLP audio/MLP text×A/B的12类LoRA角色。
首步分别检查六组B的非零任务梯度及对应A的零任务梯度，第一次更新后检查六组A/B全部解锁；
保留零audio gate对应上游零任务梯度的审计规则，不把weight decay引起的变化当作任务梯度证据。
每个完整forward重置70个投影的轻量Python调用计数/形状；首步及周期日志检查两轮各一次和scope复位。
每轮40次Attention共享残差、30次MLP audio、30次MLP text，合计140次基础投影、200次低秩分支调用。

## 正式提交

本地代码提交并推送GitHub后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_modality_lora_r8_5090.sh --stage-workers 4
```

直接通过vc wrapper提交一个formal，无自动smoke/tensor前置任务；登录节点只做CPU资产检查、metadata索引和提交。
保留--output-dir、--resume-checkpoint、--resolved-paths、--text-model-dir、--ced-model-dir、--index-cache-root、--stage-workers。
默认输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_modality_lora_r8_5090/formal_<timestamp>/
```

训练目录、launcher、route、output、job、日志、index-cache、tmpfs及完整恢复身份独立。
route为`ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_modality_lora_r8_v1`。
主要产物：formal/runtime.json、config_audit.json、metrics.jsonl、first_gradient_rank<N>.json、opened_gate_gradient_rank<N>.json、
staging_report.json、training_report.json及checkpoints/。metrics新增lora_execution。
保存step2886/5771/8656/11542/14428/17313及最终model--epo-3.ckpt；最终step为formal/checkpoints/model--step-17313.ckpt。

## 完整恢复

只接受新路线同代码、资产、数据、依赖和配方的完整formal checkpoint，使用原run root。
严格恢复全模型、所有400份LoRA、optimizer/scheduler/scaler/GradNorm、八rank RNG和数据游标，不重置已训练LoRA。
原两套LoRA、四套全模态LoRA、textmesh/MeSH或其他注入顺序的checkpoint均不直接full resume。
每次恢复重新staging并从共享路径开训，保留原文本初始化路径。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_modality_lora_r8_5090/formal_替换为实际时间戳"
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_modality_lora_r8_5090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

## 验证与证据边界

仓库根目录手动运行，均不会提交真实作业：

```bash
python -B tests/validate_mizar_s1_attn_shared_mlp_modality_lora_r8.py
# 具备原依赖时手动要求真实CPU tensor用例全部运行：
python -B tests/validate_mizar_s1_attn_shared_mlp_modality_lora_r8.py --require-tensors
```

本地113项实际检查通过，35项真实CPU tensor检查缺依赖跳过，其中新版9项。
覆盖实际源码NumPy非零投影/梯度有限差分、共享Attention两段梯度相加、MLP位置125/126隔离与错轮拒绝、
真实30次循环和post-input CED gate、20类优化器/全LR位置、梯度审计失败、调用元数据拒绝、
Bash模拟正式提交/参数转发/拒绝、共享启动/后台复制/失败清理/八卡切换接口及原8→4/4→8回归。
源码对照校验146原文件SHA保护、19复制文件除命名外一致、完整骨干及train AST受限差异、原数据/更新/保存恢复合同。

另提供9项真实CPU tensor用例：独立存储/基础对象/RNG、同结构零LoRA logits和基础梯度、
非零全部投影/两轮反向、20类首步和更新后梯度/优化器、严格文本权重继承、
保存后下一次更新及RNG/tracker对齐、重复forward/形状/异常scope恢复、共享Attention autograd相加、MLP直接边界与梯度隔离。
checkpoint collective/scaler为明确CPU fixture，不代表真实八卡恢复。
验证报告：`audit_reports/mizar_s1_attn_shared_mlp_modality_lora_r8_validation.json`；本目录reference_audit.json记录当前源码对照。
缺Torch/Transformers/torchaudio/soundfile时明确skip，--require-tensors会拒绝缺依赖或任何skip。
本轮未执行git commit/push、远程生产权重读取、GPU训练、八卡恢复或真实作业提交，不登记远程PASS。
现有MMAU/MMAR evaluator尚需另行适配本路线的新LoRA共享合同和state_dict。
