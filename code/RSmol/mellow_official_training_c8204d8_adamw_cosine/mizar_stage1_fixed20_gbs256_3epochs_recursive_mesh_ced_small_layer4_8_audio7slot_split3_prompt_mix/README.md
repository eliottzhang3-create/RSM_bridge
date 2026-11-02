# Prompt-conditioned E/A4/A8 mixture / MeSH audio7slot split3 / Mizar S1

从原 clean split3 训练路线完整复制隔离，后缀为 audio7slot_split3_prompt_mix。
保留原训练配方、七槽 split3 初始化及后续 MeSH write/read，新增有效prompt条件的输入融合。
原训练路线、launcher 和共享模型未修改。本地仅改代码，经GitHub同步；
权重、数据及GPU训练位于远程Linux。

## 确认的计算

冻结CED每批一次编码，沿用原最终层mapper得到E，以及两套独立中间MLP得到A4/A8，
三者均为[B,126,576]。仅取原256-token截断/padding后的有效prompt input embeddings，
不使用answer、SEP、生成文本或contextual hidden。

~~~python
P = lm.get_input_embeddings()(prompt_input_ids)          # [B,256,576]
q = (P * prompt_attention_mask[..., None]).sum(1) / prompt_attention_mask.sum(1, keepdim=True)
alpha = softmax(prompt_mix_mlp(q), dim=-1)                # [B,3], order E,A4,A8
S_E = alpha[:, 0, None, None] * E
S_4 = alpha[:, 1, None, None] * A4
S_8 = alpha[:, 2, None, None] * A8
E_mix = S_E + S_4 + S_8
~~~

prompt_mix_mlp为新增独立Linear(576,576) → GELU → Linear(576,3)，
无额外LN/dropout/温度/正则损失。Linear权重默认随机初始化，bias全0。
strict加载原文本权重后，在CPU torch.random.fork_rng(devices=[])内构造，
保持继承模块参数及原全局RNG不变，在optimizer/DDP前注册。

池化使用tokenizer attention_mask，检查形状、binary值及每条至少一个有效token。
有效的!（ID17）和EOS/special token参与池化，不以ID17过滤有效文字。
PAD只在此均值中排除；原LM仍使用attention_mask=None，保持原attention规则。
q、alpha及三分支不detach，梯度连通原prompt embedding与新MLP。

- prelude前126个音频输入：E_mix。
- 音频memory 7槽：[S_E,0,0,0,0,S_4,S_8]。
- slot0仅存加权E，三份加权种子的和进入prelude。
- 每个sample只有3个标量权重，沿音频token及channel广播。
- 同一组三份加权tensor只计算一次，供mixture与memory共同使用，不再次乘alpha。
- 文本/SEP/answer/PAD仍是原5槽[embedding,0,0,0,0]。

## 保留的 split3 与 MeSH

先strict加载原五槽MeSH文本checkpoint，包括原六个文本router。
六个独立音频router仍为Linear(576,7)，逐一复制对应文本router：

~~~python
W_new = W_old[[0,1,2,3,4,0,0]]
b_new = b_old[[0,1,2,3,4,0,0]]
b_new[[0,5,6]] -= log(3)
~~~

同一query的初始概率为7项：
[p0/3,p1,p2,p3,p4,p0/3,p0/3]。
原文本router完整保留；新增router及各行参数独立，初始约束后续不强制维持。

prelude结束、两次中段循环结束均执行原all-slot additive write / weighted read。
prelude用output作query/value；循环用input作query、output作write value。
后续无直接残差注入、重置种子或额外缩放。
初始种子的直接read贡献为p_read0/3 * E_mix；普通三分槽write/read贡献仍为
p_write0*p_read0*V/3，不乘3补偿。

20物理层、30次执行0..14,5..14,15..19，音频仍126个序列位置。
每次完整forward显式传audio_memory_seeds=(S_E,S_4,S_8)并重建memory，use_cache=False。
Mellow.generate_prefix_inference()返回(prefix, encoder_output, weighted_seeds)；
未来evaluator必须把同一次前缀构造返回的weighted_seeds传入每次完整LM forward。
无跨sample的模块种子缓存。现有MMAU/MMAR evaluator需要独立适配本新route。

## 原训练配置

| 项目 | 设置 |
|---|---|
| 初始化 | 原MeSH文本checkpoint-009244；fresh音频optimizer/scheduler/scaler/tracker/RNG/cursor |
| 数据 | hybrid 1,477,492行，原template/tokenize/shuffle/sampler/rank seed |
| 音频 | mono32k、中心裁/右补20秒，整体mel/BN后切分，原最终层mapper/dropout0.05 |
| 布局 | audio126 + SEP1 + prompt256 + answer300 = 683，prefix383，PAD17/EOS0 |
| Loss | 原answer-only CE，rank-local有效token均值，再GA/DDP平均 |
| Batch | 8 GPU × microbatch8 × GA4 = GBS256 |
| 时长 | 3epochs，每epoch5771updates，总17313，warmup866，每epoch丢弃116行 |
| LM/文本router | maxLR3e-4 |
| mapper/中间MLP/音频router/新增prompt MLP | maxLR6e-4 |
| AdamW | betas(.9,.999)，eps1e-8，矩阵wd0.01，bias/ndim≤1不衰减 |
| 调度/梯度 | 原LambdaLR、GradNormTracker及GradScaler |
| 精度 | FP32，无autocast/TF32 |
| 数据读取 | DataLoader workers0，立即读共享原始音频，后台4workers复制，八rank在optimizer边界一致切换 |
| 保存step | 2886、5771、8656、11542、14428、17313，及epoch3最终文件 |

新增prompt MLP 334,083参数；原两套中间MLP2,443,776、六个音频router24,234。
相对普通五槽CED MeSH共新增2,802,093，相对clean split3仅新增334,083。

## 正式提交

本地提交并推送GitHub后，在远程登录节点执行：

~~~bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_split3_prompt_mix_5090.sh --stage-workers 4
~~~

直接通过vc submit提交formal，不自动提交smoke或tensor前置任务。
沿用mellow_c8204d8、原镜像、pdgpu-5090 / 32CPU / 256G / 8GPU / 1node。
登录节点只执行原有资产检查和CPU metadata索引。

默认资产与数据：

~~~text
text: /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244
CED: /hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small
manifest: /hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
resolved: /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
~~~

保留--output-dir、--resume-checkpoint、--text-model-dir、--ced-model-dir、
--resolved-paths、--index-cache-root、--stage-workers参数。

默认输出root（内部再生成formal目录）：

~~~text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_prompt_mix_5090/formal_<timestamp>/
~~~

route/output/log/job/index-cache/tmpfs/resume身份独立。
最终产物：

~~~text
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/training_report.json
~~~

## 完整恢复与诊断

只接受本新路线同一源码/数据/资产/训练配置的完整未完成checkpoint。
route为ced_small_mizar_s1_fixed20_gbs256_recursive_mesh_layer4_8_audio7slot_split3_prompt_mix_v1。
runtime合同记录prompt来源/池化mask、MLP结构/初始化、branch顺序、prelude与memory内容，
同时绑定源码及资产hash。旧split3、split3_inject、MeSH4→8/8→4及非MeSH路线checkpoint
均不能作为本路线full resume。

新MLP参数纳入model/optimizer/GradNorm状态，恢复包括scheduler/scaler、八rank RNG和数据游标。
先构造完整架构再strict覆盖，加载后不重新初始化已训练router/MLP。
完成checkpoint只能审计，不能继续resume。

从实际已保存的中间step恢复（替换实际timestamp/step）：

~~~bash
OUT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_prompt_mix_5090/formal_<实际timestamp>"
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_split3_prompt_mix_5090.sh \
  --output-dir "$OUT" --resume-checkpoint "$OUT/formal/checkpoints/model--step-2886.ckpt" --stage-workers 4
~~~

首两次及每25次update记录原router/seed/memory/read诊断，以及alpha各分支mean/min/max/熵、
原始及加权分支RMS、E_mix RMS。这些是rank0最后一个microbatch的观察，不影响loss。
梯度审计包含新prompt MLP：所有训练参数梯度存在且有限、冻结CED无梯度。
真实样本不强制每个梯度非零，合成测试另行验证非零通路。

## 本地验证

仓库根目录：

~~~bash
python -B tests/validate_mizar_s1_mesh_ced_audio7slot_split3_prompt_mix.py
# 原训练依赖完整时，要求所有真实CPU tensor检查执行：
python -B tests/validate_mizar_s1_mesh_ced_audio7slot_split3_prompt_mix.py --require-tensors
~~~

路线内python -B runtime_checks.py --require-tensors转发到同一检查。
不下载资产、使用CUDA或提交真实作业，也不作为formal的自动前置条件。

本地90项实际测试通过，Python/Bash语法与模拟提交检查通过。
真实CPU tensor检查30项因缺Torch/Torchaudio/Transformers/Soundfile跳过，其中本路线17项。
覆盖实际代码的mask池化/权重广播/种子内容、训练推理一致/答案无泄漏、三次memory运算、
finite difference、split3初始化及原路线回归。
可选tensor测试覆盖strict加载/RNG继承、autograd、梯度审计、因果性及完整恢复后下一次update。

protected_sources.json固定1978个原代码/脚本/文档的normalized-LF SHA256，检查原文件保持。
audit_reference.py核对整个MeSH模型、整个trainer、优化器/调度/恢复及训练合同，
只允许明确的种子接口、prompt MLP验证/诊断和身份变化。

报告：audit_reports/mizar_s1_mesh_ced_audio7slot_split3_prompt_mix_validation.json，
本目录reference_audit.json。
尚未执行远程真实权重tensor加载、八卡训练或完整恢复，未登记远程PASS。
