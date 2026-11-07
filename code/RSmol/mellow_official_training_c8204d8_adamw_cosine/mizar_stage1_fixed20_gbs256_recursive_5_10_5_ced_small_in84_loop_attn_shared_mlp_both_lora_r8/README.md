# 非MeSH CED8→4：Attention共享＋MLP共享与模态独立并行LoRA

完整复制隔离自`mizar_stage1_fixed20_gbs256_recursive_5_10_5_ced_small_in84_loop_attn_shared_mlp_modality_lora_r8`。
本地Windows维护代码，经GitHub中转；权重、数据及GPU正式训练位于远程Linux。
内部目录用`in84`及`mlp_both`缩写兼容Windows路径长度；both表示shared＋modality，不改变正式launcher名称。
仅改变中间Transformer MLP的LoRA组成和缩放，以及相应参数分类、合同和审计。

## LoRA定义

20物理层、30逻辑执行，完整轨迹`0..14,5..14,15..19`；非MeSH，无memory/router。
只有中间十个物理层5..14的q/k/v/o和gate/up/down使用LoRA。
不同物理层、不同投影、不同轮次各自拥有独立A/B。
前五层、后五层、embedding、norm和LM head沿母版全量训练，不新增LoRA。
基础W/bias始终全量训练，在Audio/Text及两轮之间共享，保留原参数对象与checkpoint键。

| 投影类别 | 分支 | rank | alpha | alpha/rank |
|---|---|---:|---:|---:|
| Attention q/k/v/o | Audio/Text shared | 8 | 16 | 2 |
| Transformer MLP gate/up/down | Audio/Text shared | 8 | 8 | 1 |
| Transformer MLP gate/up/down | Audio private | 8 | 8 | 1 |
| Transformer MLP gate/up/down | Text private | 8 | 8 | 1 |

令Delta(x)=B(A(x))。每个Attention投影仍为：

```text
y = Linear(x,W,bias) + 2 * Delta_attention_shared[loop](x)
```

每个Transformer MLP投影为：

```text
base = Linear(x,W,bias)                         # 全序列仅一次
shared = Delta_mlp_shared[loop](x)               # 全序列
private_audio = Delta_mlp_audio[loop](x[:,:126])
private_text = Delta_mlp_text[loop](x[:,126:])
y = base + 1 * (shared + concat(private_audio,private_text,dim=1))
```

两个残差读取该投影同一输入，并行计算后相加，不串行传递、不重复基础Linear、不另除以2。
gate/up投影读取原hidden，down投影读取原SwiGLU乘积；原激活与乘积公式保持。
每个MLP token使用两条scale1分支，显式系数之和为2；这不保证两条独立残差之和的实际幅度、梯度或更新量等于原scale2单分支。
音频位置0..125，文本126起（包括SEP、prompt、answer/PAD及生成位置），沿用固定切片，无学习模态router。
完整causal self-attention、RoPE、position IDs保持，模态私有LoRA不隔断注意力。
CED mapper、中间音频特征MLP、input projection和gates均沿母版，不额外加LoRA。

A仍用Kaiming uniform(a=sqrt5)，B全零，dropout0；严格加载原文本权重后才注册音频模块及LoRA。
独立CPU fork_rng保留调用者RNG及基础模型/音频模块初始化；新增LoRA布局重新初始化，不继承母版已训练adapter。
每次调用显式选轮，finally复位，反向使用该次前向选中的张量；无merge、detach、基础weight.data写入或requires_grad切换。
保持use_cache=False、完整序列重算和decoder gradient checkpointing关闭。

state_dict示例，末位0/1分别表示两轮：

```text
caption_decoder.lm.model.layers.5.self_attn.q_proj.lora_A.shared.0
caption_decoder.lm.model.layers.5.mlp.gate_proj.lora_A.shared.0
caption_decoder.lm.model.layers.5.mlp.gate_proj.lora_A.audio.0
caption_decoder.lm.model.layers.5.mlp.gate_proj.lora_A.text.0
# lora_B同理。
```

## 参数及训练审计

真实hidden576、MLP1536、Q9/KV3 heads，K/V输出192。十层、两轮合计：

| LoRA类别 | 参数 |
|---|---:|
| Attention shared | 614,400 |
| MLP shared | 1,013,760 |
| MLP audio | 1,013,760 |
| MLP text | 1,013,760 |
| 合计 | 3,655,680 |

比直接母版2,641,920多1,013,760。每层每轮4＋3×3=13对A/B，总260对、520份独立参数张量。
原串行音频分支2,777,859保持；相对普通CED循环母本共新增6,433,539参数，基础权重不重复计数。
完整多模态forward使用全部LoRA；每个MLP token在各投影仅激活shared与自身private两个rank8残差。

合同使用alpha_by_family/scale_by_family，避免一个全局alpha误用于Attention和MLP。
优化器以投影目标和branch共同分类：attention_shared与mlp_shared分别记录，全部LoRA maxLR6e-4、矩阵decay.01。
保留8类原角色；两轮×4类LoRA×A/B产生16类LoRA角色，共24类。
首步分别检查8组B的任务梯度和相应A零任务梯度，首次更新后检查8组A/B及音频上游任务梯度。
保留关闭audio gate时对应上游梯度为零的规则，不把weight decay当任务梯度证据。

每完整forward重置调用/形状及分支计数；检查140次基础投影、260次LoRA分支调用。
每轮40次Attention shared＋30次MLP shared＋30次MLP audio＋30次MLP text，正确边界和scope复位。
正式trainer按真实投影形状检查四类参数量、520份独立存储及dtype/device，不以文件大小估算。

## 保留的音频及训练配方

```text
前五层 → audio input残差 → CED8残差 → 第一轮中段循环
       → audio input残差 → CED4残差 → 第二轮中段循环 → 最后五层
```

CED-small冻结/eval，每批一次编码提取final及完整block4/block8最终norm前输出。
final经原freq_merge_126 mapper得到E，两份原独立MLP得到A4/A8。
每完整LM forward仅算一次audio_linear(E)，保持梯度；每入口先加共享input gate残差，再用post-input hidden计算对应CED gate。
音频残差只直接更新前126位置，第二轮后不再注入；mapper dropout.05、音频MLP dropout0及三个零gate初始化保持。

文本初始化仍为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244
```

仅继承原文本模型/tokenizer，新建音频训练状态与所有LoRA；直接母版音频checkpoint不作为full resume。

| 项目 | 沿母版配置 |
|---|---|
| 数据 | hybrid1,477,492行，原manifest/resolved_paths及全部sources |
| 输入 | fixed20在线原始音频，126audio＋SEP1＋prompt256＋answer300=683，prefix383 |
| 监督 | 原answer-only CE，rank-local microbatch有效token均值→GA均值→DDP均值 |
| Token | PAD !/17，EOS/SEP0，attention_mask=None |
| 数值 | FP32，无autocast，TF32关闭，保留原GradScaler |
| Batch | 8GPU×microbatch8×GA4=GBS256 |
| 更新 | 3epochs，5771/epoch，17313total，866warmup，每epoch丢116行 |
| maxLR | 基础LM3e-4；全部LoRA/mapper/音频新增模块6e-4 |
| 优化器 | AdamW betas(.9,.999)/eps1e-8；矩阵decay.01，bias/一维无衰减 |
| 调度/随机 | 原LambdaLR/GradNormTracker、seed1234/rank seed1234+rank及数据顺序 |
| staging | 共享路径立即开训，默认4线程后台复制；完成后optimizer边界八卡共同切换 |
| 保存 | step2886/5771/8656/11542/14428/17313及epoch3 |
| 资源 | mellow_c8204d8，原容器，pdgpu-5090/32CPU/256G/8GPU/1node |

骨干及文本loader整个AST与母版一致，train更新函数整个AST相等；仅扩展角色和梯度家族审计。
原AdamW/LambdaLR/GradNorm、loss、数据、staging、save/reload/RNG恢复逻辑保持。

## 正式提交

本地代码通过GitHub提交推送后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_shared_modality_lora_r8_5090.sh --stage-workers 4
```

入口直接通过vc wrapper提交一个formal，无自动smoke/tensor前置任务；登录节点只做CPU资产检查、metadata索引和提交。
保留--output-dir、--resume-checkpoint、--resolved-paths、--text-model-dir、--ced-model-dir、--index-cache-root、--stage-workers。
route为`ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_shared_modality_lora_r8_v1`。
训练目录、launcher、route、output、job、日志、index-cache、tmpfs和完整恢复身份独立。
默认输出根：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_shared_modality_lora_r8_5090/formal_<timestamp>/
```

主要产物为formal/runtime.json、config_audit.json、metrics.jsonl、first_gradient_rank<N>.json、
opened_gate_gradient_rank<N>.json、staging_report.json、training_report.json及checkpoints/。
最终权重为本run根目录下：

```text
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
```

## 完整恢复

仅接受本路线同源码、资产、数据、依赖与配方的完整formal checkpoint，使用原run root。
完整恢复基础LM、音频分支、520份LoRA、optimizer/scheduler/scaler/GradNorm、八rank RNG和数据游标，不重置adapter。
直接母版MLP只有private且alpha16，新版增加shared并将MLP各分支alpha改8；即使旧private矩阵形状相同也不能直接full resume。
其他r4/r8/r16、注入顺序、MeSH/textmesh checkpoint均按独立合同拒绝。
每次恢复重新staging，从共享路径开训，保留原文本初始化绝对路径。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_shared_modality_lora_r8_5090/formal_替换为实际时间戳"
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_shared_modality_lora_r8_5090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

## 本地验证与证据边界

仓库根目录手动运行，均不提交真实训练：

```bash
python -B tests/validate_mizar_s1_attn_shared_mlp_both_lora_r8.py
# 具备依赖时手动要求真实CPU tensor用例全部执行：
python -B tests/validate_mizar_s1_attn_shared_mlp_both_lora_r8.py --require-tensors
```

本地152项实际检查通过；45项真实CPU tensor检查缺Torch/Transformers等依赖跳过，其中新版10项。
包含实际源码NumPy非零投影、MLP共享与private并行/scale1/基础Linear仅一次、125/126边界、
共享MLP双模态梯度相加、模态私有与轮次隔离、有限差分、完整30次循环及post-input gate，
24类优化器/全部LR位置、8组梯度审计失败、分支调用/scale/边界/异常scope拒绝、
真实Bash模拟提交与参数转发、共享启动/后台复制/失败清理/八卡切换接口，以及直接母版/原r8两顺序/串行音频母本回归。
Python/Bash语法、257个原文件SHA保护、20个复制源码除隔离命名外一致、整个骨干/loader与train更新函数AST对照及原训练合同通过。
Windows沙箱临时目录/Bash子进程限制已在此前验证中确认，本次完整流程经批准在沙箱外运行。

另提供10项真实CPU tensor用例：初始化/基础对象/RNG/独立存储、零LoRA logits与基础梯度、
非零全部投影及轮次反向、24类首步和更新后梯度/优化器、完整checkpoint下一步及RNG/tracker一致、
严格文本权重继承、重复forward与异常scope、Attention共享autograd、MLP模态边界与共享/私有autograd，
以及MLP scale1并行和错误alpha/scale/别名存储拒绝。checkpoint collective/scaler为明确CPU fixture，不代表真实八卡恢复。
报告为`audit_reports/mizar_s1_attn_shared_mlp_both_lora_r8_validation.json`；本目录reference_audit.json记录当前源码对照。
缺依赖时明确skip，--require-tensors会拒绝任何skip，不进入formal调用链。
本轮未执行git commit/push、远程生产权重读取、GPU训练或真实八卡恢复，不登记远程PASS。
现有MMAU/MMAR evaluator需要为本路线新LoRA合同和state_dict单独适配。
