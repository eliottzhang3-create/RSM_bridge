# PCLM / Tiny BERT：CED 4/8/12 层隔离实验评估与实施方案

日期：2026-11-02。状态：完成论文、作者代码和本地训练路线调研；尚未实施模型修改，未下载权重，未启动远程训练。本文件不是测试 PASS 或远程训练报告。

## 1. 结论与已经确认的范围

工程上可行，建议作为“现有 prompt_mix 条件分支替换为可训练 Tiny BERT”的独立实验。当前路线已经实现按 prompt 加权混合多层音频表示；此次主要检验预训练上下文文本编码是否比 LM 输入 embedding 的 masked mean 更适合生成层权重。不能预先承诺评测提升。

用户已明确：**第12层保留官方 mapper，只替换 prompt 条件分支为 Tiny BERT。** 因此第4/8层仍用各自原有独立 MLP，第12层 E 仍来自 CED 最终归一化输出与官方 CEDTemporalMapper，不新增 A12 MLP，也不改成未经 final norm 的 block12 hook。

训练维持原有 3 epochs 联合 SFT。Tiny BERT 全部编码器参数与新权重 MLP 参与训练，max LR 6e-4；其他学习率分组沿用父路线。保留七槽 split3、原音频融合及递归 MeSH 执行方式。

本地 Windows 只做代码、文档和可执行的本地检查。真实权重、数据、训练环境及 GPU 验证位于远程 Linux；代码通过 GitHub 中转。正式训练仍由对应的 vc submit 包装脚本提交，登录节点不执行 CUDA 训练。

## 2. 调研依据及适用边界

- [论文 §5 与 Table 2](https://arxiv.org/html/2605.27772v1#S5)：PCLM 用 prompt 编码生成 softmax 层权重，先将各层投影到 LM 空间，再加权求和。相较论文的 SFT w/o PCLM 对照，VoxParadox 上 AF3 为 34.80→60.00，Qwen2-Audio 为 45.70→68.65。这些是论文特定模型、数据和训练阶段下的结果，不能视作当前项目的预期增益。
- [作者 Qwen2AudioPCLM 实现](https://github.com/ihp-lab/VoxParadox/blob/main/qwen2audio/modeling_qwen2audio.py)：使用 BERT 的最后一层 CLS；公开实现冻结 BERT，权重头为 Tanh MLP，并对 prompt 去除选项等内容。用户要求训练 Tiny BERT，故应明确记录为实验差异。
- [指定模型](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2)及[配置](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/raw/main/config.json)：2层、hidden128、2头、FFN512、vocab30522、最大位置512、dropout0.1，英文 uncased WordPiece。
- [Transformers BERT 文档](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertModel)：BertModel 可关闭 pooling layer；本方案直接使用 last_hidden_state 的 CLS。

论文的音频骨干和分阶段训练与当前 CED 联合训练不同。其对语音编码器层间信息变化的结论，不能直接当作 CED 4/8/12 层已经存在同样问题的证据。此次沿用用户的联合三轮 SFT，不引入论文的额外 alignment 阶段或 DPO。

## 3. 父路线及实际计算

父提交入口：

`C:/Xlance/GZ_bridge/Recursive_SALM/RSM_bridge/code/RSmol/run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_split3_prompt_mix_5090.sh`

父路线根目录：

`C:/Xlance/GZ_bridge/Recursive_SALM/RSM_bridge/code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_audio7slot_split3_prompt_mix`

已检查根 README、路线 README、launcher、run/config/contract、dataset/collate、CED/decoder/Mellow、optimizer 与训练审计/恢复路径。以下使用“路线内文件”标示这些目录内的文件。

### 音频侧继续沿用

- CED-Small 冻结且 eval，每个 microbatch 只运行一次音频编码器。
- 完整 fixed20 波形先完成原重采样、mel、BN，再按现有两个 segment 处理；不改切片或时间/频率重排。
- 第4/8层 hook 输出分别经原频率合并 MLP：LN1536→Linear1536→576→GELU→Linear576→576→LN576。
- E 来自最终层 post-final-norm 特征和原官方 mapper。三路投影后均为 `[B,126,576]`。
- 官方 mapper 包含其原有 global/position/type/refine 等结构，不能将它误写成 A4/A8 的简单 MLP。

### 文本条件分支的替换

```text
原分支：有效 prompt 的 LM embedding → masked mean[576]
        → Linear576→576 → GELU → Linear576→3 → softmax

新分支：同一个有效 prompt → BERT WordPiece tokenizer
        → 预训练 BertModel(add_pooling_layer=False)
        → last_hidden_state[:,0,:][128]
        → Linear128→576 → GELU → Linear576→3 → softmax
```

保留现有权重头的隐藏宽度576及 GELU，只调整输入维度128，避免同时更改隐藏宽度、激活等变量。输出列顺序必须固定为 **(E,A4,A8)**，不能按口头的“4/8/12”顺序重新排列。

设 `alpha=(alpha_E,alpha_4,alpha_8)`：

```text
S_E = alpha_E * E
S4  = alpha_4 * A4
S8  = alpha_8 * A8
E_mix = S_E + S4 + S8
audio memory seeds = [S_E, 0, 0, 0, 0, S4, S8]
```

alpha 是每样本3个标量，对126个音频位置及576个通道共享；不是 token gate 或 cross-attention。它只依赖问题文本，相同 prompt 在 eval 时不会因音频不同而改变权重。

继承三次普通全槽 write/read、七路音频 router 的 split3 初始化、SEP/文本五槽、完整前缀调用及每次显式传入相同音频种子。不添加额外残差、×3补偿或额外音频编码。递归物理层轨迹维持 `0..14,5..14,15..19`，20个物理层执行30次，use_cache=False。

## 4. Prompt 合同：先解决语义一致性，再接模型

建议第一版采用“**LM 实际可见 prompt**”作为 BERT 文本来源，避免 BERT 额外看到超过 LM 原256-token截断范围的内容：

1. dataset 对每行仍只调用一次原 text_pair，完成原随机模板选择与大小写处理。
2. 保持 LM 原 prompt256、answer300 的 tokenization 和 padding 不变。
3. 在 CPU 用 prompt input_ids 与 attention_mask 提取有效 prompt，按已知包装边界移除外加 BOS/EOS；长文本被截断时不得假设末尾一定存在 EOS。不得读取 answer 字段。
4. 解码时保留有效正文 token，仅按 mask 排除 padding。尤其不能 `ids != 17`，也不能无差别 `skip_special_tokens=True`：LM 将 `!` 注册为 pad，正文有效的 `!` 也可能使用同一 ID。
5. 用 Tiny BERT 的 vocab 独立重新分词，uncased、右侧 padding/截断、`max_length=256`（含 CLS/SEP）、独立 attention_mask。建议固定 padding 到256以便合同和 batch 形状稳定。BERT pad0 与 LM pad17严格分开。
6. 保留父路线问题正文和 MCQ 选项，不引入作者代码的去选项/去编号规则；这些规则不一定适用于 Mizar 的题目格式。
7. 训练和推理调用同一个 helper。改变 answer 不得改变 BERT tokens、alpha 或音频种子。

正文特殊符号、空 prompt、截断、真实 `!` 和包装 EOS 的行为必须有明确测试，不能靠全局删除某个 token ID 解决。空正文可经 BERT 转成 CLS/SEP，不允许产生全零 attention_mask。

BERT 输入应作为新 batch 字段，例如 `prompt_bert`，包含 input_ids/attention_mask/token_type_ids。dataset、collate_fn、move_batch、Mellow、decoder 与 inference 接口需要贯通；不能只在 decoder 中实例化模型而遗漏输入链路。推理生成过程中只对最初用户 prompt 编码一次，每个新 token 不重新运行 BERT，生成的回答也不能成为 gate 输入。

## 5. 训练合同

| 项目 | 新路线要求 |
|---|---|
| 初始化 | 沿用原文本 MeSH checkpoint 初始化；从头开始本次音频训练，不加载父音频路线 full resume |
| CED | 冻结、eval |
| LM 与原文本 routers | 全量训练，max LR 3e-4 |
| 官方 mapper、A4/A8 MLP、六个音频 routers | 全部训练，max LR 6e-4 |
| Tiny BERT 与新权重 MLP | 全部训练，max LR 6e-4 |
| 训练长度 | 3 epochs；1,477,492条时每轮5771 updates，共17313；每轮丢弃116行 |
| GBS | 8 GPU × microbatch8 × accumulation4 = 256 |
| 精度 | 原 FP32；不启用 autocast 或 TF32 |
| 调度 | 原5% warmup=866 updates，之后原cosine，最低倍率0.1 |
| AdamW | betas=(0.9,0.999)，eps=1e-8；矩阵 decay0.01，bias及一维参数不decay |
| 损失/梯度 | 原 answer-only CE、rank-local有效token均值、GA4、DDP平均、GradNormTracker/GradScaler |
| 序列 | audio126+SEP1+prompt256+answer300=683，prefix383；PAD17、EOS/SEP0 |
| 数据/I/O | 原shuffle/模板/rank seed、workers0、共享存储先读与后台4-worker拷贝、update边界全rank切换 |
| 保存 | steps2886/5771/8656/11542/14428/17313及epoch3最终文件 |
| 远程资源 | 原 mellow_c8204d8 环境，pdgpu-5090，1节点8卡、32CPU、256G |

按原 scheduler，BERT 的起始 LR 约6e-5，峰值6e-4，末端约6e-5。“6e-4 max LR”不能误改成恒定6e-4。BERT 不额外 warmup、不先冻结、不新增辅助损失，不静默降低用户指定 LR。

梯度链应为：answer CE → LM 的 mixed prefix / memory → alpha → 权重 MLP → Tiny BERT。新增 BERT 分支不得包在 no_grad 内，不得 detach CLS/alpha。冻结 CED 的 no_grad 范围不能覆盖可训练 mapper 或 prompt 分支。

保留 BERT 配置中的 dropout0.1，随 model.train/eval 正常切换；它不再是作者代码中固定 eval 的冻结特征提取器。训练期间不可跨 optimizer step 缓存 CLS 或 alpha。随机模板存在时，tokens 也不能只按原始行号缓存而忽略实际模板。

新增 dropout 会改变整体随机轨迹，不能承诺与父训练逐步数值一致；但应保留父数据顺序、模板逻辑、初始化参数及配方。BERT/新头应在继承模块构造完成后、optimizer/DDP 前，在独立 CPU RNG 上下文中构造及加载，避免消耗父模块初始化 RNG。

## 6. 参数、显存与效果风险

按官方 config 的标准 BertModel 结构、不含 pooler/MLM/NSP heads 计算：

| 部分 | 参数数 |
|---|---:|
| Tiny BERT embeddings + 2 encoder blocks | 4,369,408 |
| 新权重头128→576→3 | 76,035 |
| 新 prompt 分支合计 | 4,445,443 |
| 被替换的原权重头576→576→3 | 334,083 |
| 相对父路线净增加 | **4,111,360** |

以上是结构计算，实施后必须以实际 named_parameters manifest 复核。仅净增 FP32 参数、梯度及 Adam 两份 moment 约62.7 MiB/卡；未包含激活、DDP buckets、GradNormTracker 副本、临时张量或分配器开销，不能作为显存总量或余量保证。真实吞吐、峰值显存须在远程8卡正式作业中记录。

主要不确定性：

- **收益空间**：父路线已具备 prompt-conditioned 三层混合，替换收益可能小于从“仅最终层”接入多层的收益。
- **容量与域**：Tiny BERT 仅2层；能否学到当前题目需要的任务差异，以及 CED 三层是否有互补信息，都要由评测验证。英文 uncased 与非英语输入的适配也受限。
- **优化**：对预训练 BERT 使用6e-4联合训练可能导致快速漂移、softmax饱和或层选择塌缩。先遵守用户 LR，用有限梯度、梯度范数/更新幅度、alpha分布/熵监测，不默认添加熵正则或温度参数。
- **门控表达能力**：只有prompt条件，没有音频条件；不能根据某条音频的噪声或特征质量动态选择层。
- **层权重解释**：三路 projector 结构和尺度不同，且后接 MeSH routers。alpha 大小本身不等于真实层贡献，应同时看分支RMS、加权后RMS与router使用情况。
- **实验归因**：第一轮同时引入预训练文本特征、上下文编码及额外参数，适合判断整套替换是否有用；它不是严格参数量匹配的因果实验。后续若需要拆分原因，再考虑冻结Tiny BERT或常量gate对照，不默认增加训练任务。

## 7. 隔离与文件级实施清单

建议完整复制父路线代码及提交入口，使用 `_prompt_mix_pclm_tinybert` 后缀。新入口建议：

`run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_split3_prompt_mix_pclm_tinybert_5090.sh`

这是计划名称，当前尚未创建。新 route contract、output、job/log、index_cache、tmpfs前缀、diagnostics/report与resume身份均独立。音频数据和冻结权重资产可只读复用，不能覆盖父输出、改父模型实现或把新代码回写成父默认行为。

| 路线内文件/模块 | 具体工作与验收重点 |
|---|---|
| launcher、run.sh、write_config.py、contract.py | 贯通 --bert-model-dir；固定新route/参数合同；资源与原调度不变；禁止父route输出或resume混用 |
| 新 BERT 资产校验模块 | 验证 config/vocab/权重、完整revision、大小与SHA256；只读离线加载；不从训练进程自动联网下载 |
| dataset / collate | 单次生成随机prompt，复用LM有效prompt提取helper，增加独立BERT tokens与mask；不改原answer/采样逻辑 |
| train.py move_batch | 显式将BERT token字典搬到目标device；检查batch大小、长度、dtype和mask |
| models/mellow.py、models/decoder.py | 加入BERT与新头，传入BERT tokens；维持原音频mix、种子、prefix、forward及inference合同 |
| optim.py | 新增prompt_text_encoder角色，BERT与新头进6e-4组；完整覆盖、无重复、无遗漏，并继承原decay规则 |
| train.py 审计 | 替换旧头576输入/334083参数硬断言；审核BERT全部可训练及梯度；保留原投影/router/执行轨迹审计 |
| checkpoint / runtime manifest | 保存BERT与新头state、optimizer、scheduler、tracker、scaler、all-rank RNG与数据游标；新路线严格恢复 |
| tests 与路线 README | 新路线独立检查和报告；记录架构/资产/参数/学习率与真实验证状态，父路线回归不被破坏 |

### BERT 加载与 DDP 的必须处理项

当前 DDP 使用 find_unused_parameters=False。直接从 last_hidden_state 取CLS时，默认 pooler 不参与损失；若仍注册为可训练参数，会触发unused参数问题。使用 **BertModel(add_pooling_layer=False)**，不注册 MLM/NSP 等不使用的任务头，不通过启用 find_unused_parameters 掩盖漏接分支。

预训练资产加载必须检查 missing/unexpected/mismatched keys：允许舍弃哪些 pooler/预训练任务头应按实际权重键显式列举，所有 embeddings/encoder 权重必须正确加载，不能默许其随机初始化。不得仅构造 BertConfig 后宣称已加载预训练模型，也不得以无审计的 strict=False 绕过问题。

新权重头保持随机非零权重和零bias的原初始化风格；不将最后一层全零以强行制造完全均匀alpha，否则首步BERT梯度会被阻断，违背首步梯度审计目标。

### 远程资产与恢复身份

建议远程模型目录：`/hpc_stor03/sjtu_home/jinwei.zhang/models/bert_uncased_L-2_H-128_A-2`。这是拟定路径，尚未验证其存在。

权重、vocab及config在远程准备，代码与小型清单通过GitHub同步，不把权重或数据纳入Git。实施时固定已验证的完整HF revision及SHA；本次尚未取得本地/远程资产哈希，不将main或网页短SHA当作可复现完整身份。

runtime/resume identity至少包含：新route版本、父代码/配置来源、BERT revision与文件指纹、tokenizer类别/vocab/大小写/截断/padding、CLS策略、无pooler、head128→576→3/GELU、权重顺序、全量训练/LR、数据身份和原训练合同。完整resume只接受本实验checkpoint并strict加载，不能从父audio checkpoint恢复optimizer或沿用旧prompt头。

## 8. 分层验证与训练观察

本次仅制定检查计划，没有运行下列测试：

1. **静态与隔离**：Python编译、bash语法、mock提交参数；新路径/contract独立；父目录及入口的前后hash保持一致。核实各参数恰好进入一个正确LR/decay组。
2. **Prompt行为**：随机模板只选一次；正文 `!` 保留；padding变化不影响eval gate；空/长prompt行为明确；改变answer不影响BERT输入或alpha；训练/推理helper一致；LM与BERT词表不混用。
3. **张量与梯度**：小型随机BERT/config可验证CPU计算图，但不能证明预训练加载成功；另用真实远程资产验证加载清单。alpha有限且和为1；三路mixed/seeds数值符合公式；实际answer loss能给BERT、新头、投影和音频routers有限梯度；CED保持无梯度；无unused trainable参数。
4. **MeSH与推理**：七槽/五槽语义、split3初始化及30次执行不变。生成只计算一次BERT/音频特征，每个完整LM调用传同一组weighted seeds；不能用父评测入口直接漏掉新分支。
5. **恢复**：覆盖BERT dropout、优化器状态和各rank RNG，比较连续训练与保存后恢复的下一步游标/LR/loss/alpha/参数更新；检查错误BERT资产或父route checkpoint被拒绝。
6. **远程正式作业**：沿用该父路线“直接提交formal、不自动追加smoke或tensor前置任务”的规则，保留启动审计及训练中的原梯度/数值检查；记录真实GPU型号、峰值显存和update耗时。不能将本地静态成功或张量测试跳过标成远程PASS。

诊断沿用父输出格式，新增BERT加载/参数角色/梯度状态，保留alpha三列均值/范围/熵、分支及mixed RMS。rank0最近microbatch统计必须标注其scope，不能冒充全训练集分布。观察不同题型和训练阶段的gate变化；整体loss下降或某一列alpha变大都不足以证明方法有效。

## 9. 效果评估与交付顺序

主比较为父prompt_mix与新Tiny BERT路线，使用同数据、同文本初始化、同3epochs配方和同评测协议，优先对齐epoch1/2/3（steps5771/11542/17313）。训练loss仅作为优化健康指标，判断收益以held-out任务指标为准。

若使用项目既有MMAU/MMAR，需另行适配本路线模型构造、BERT资产与tokenizer、strict checkpoint加载和逐token种子传递；其数据/scorer可复用，但当前父评测器不能直接宣称兼容。报告总分和分项、有效样本/失败数、峰值显存及吞吐。若要检验特定副语言机制，再单列相应任务，不能把泛音频QA总分当作该机制的完整证明。

落地顺序：

1. 复制并确认父路线保护、独立身份和远程BERT资产接口。
2. 实现共享prompt helper、数据字段和BERT/权重头，打通训练及inference计算图。
3. 更新optimizer、训练审计、资产manifest及严格resume，完成上述有意义的本地检查。
4. 通过GitHub同步代码，在远程准备并校验离线BERT资产，经新launcher提交3epochs正式作业。
5. 使用兼容的新路线评测入口，对齐父路线协议与训练进度比较效果。

完成代码、实际加载清单、梯度/恢复检查和真实训练/评测前，不将此方案标为已实现或实验成功。
