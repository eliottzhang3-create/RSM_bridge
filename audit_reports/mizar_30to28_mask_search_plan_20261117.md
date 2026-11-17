# Mizar 30→28 层穷举搜索最终计划

日期：2026-11-17。本文仅记录待实现方案，不代表代码已实现、远程资产已验证或 GPU 作业已提交。

## 1. 已确认的目标

- 放弃 Pruning as a Cooperative Game 的代理网络/Shapley 方法。
- 使用官方最终 Mizar 的完整架构与已训练参数，包括 CED-Small、已训练 mapper/projector、SmolLM2 LM。
- 保护原始第 0、29 层，从原始第 1…28 层中选择两层跳过，共 C(28,2)=378 个组合。每个候选最终执行 28 层。
- MMAU test-mini、MMSU、MMAR、ReasonAQA 各固定抽取 25 题，合并为同一个 100 题 manifest。
- 主指标：完整候选正文的 token log probability 之和，逐题取最大值，再按 100 题 accuracy 排名。
- BF16 混合精度推理；100 题的 encoder＋projector 输出预计算一次，所有 mask 和 GPU 复用。
- 只提交一个单节点 8×3090 作业。8 卡每轮共同评估同一个 mask，完成全体汇总后再切换。
- 第一个剪层 mask 的全部 100 题额外自由生成，打印输出与标准答案；accuracy 始终只来自 likelihood。
- MeSH 结构及其训练不属于本轮范围。

## 2. 以官方原生 Mizar 为推理参考

已核对本地官方源码：

- `C:/Xlance/assets/Mizar/Mizar_159M-main/Mizar_159M-main/pipeline/prepare_inference.py`
- `C:/Xlance/assets/Mizar/Mizar_159M-main/Mizar_159M-main/pipeline/generate_vllm.py`
- `C:/Xlance/assets/Mizar/Mizar_159M-main/Mizar_159M-main/pipeline/model_io.py`
- `C:/Xlance/assets/Mizar/Mizar_159M-main/Mizar_159M-main/docs/evaluation.md`

上游对应来源：

- https://github.com/kaiyangli-ml/Mizar_159M/blob/main/pipeline/prepare_inference.py
- https://github.com/kaiyangli-ml/Mizar_159M/blob/main/pipeline/generate_vllm.py

本轮继承官方原生输入/解码语义，不沿用项目 matched-CED evaluator 的中心裁剪和问题端 EOT。

| 项目 | 本轮规则 |
| --- | --- |
| checkpoint | `/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt` |
| 加载 | 校验原始 release SHA/资产身份，FP32 严格加载全部 state_dict，再进入 BF16 推理 |
| 音频 | 按实际解码采样率重采样至 32kHz，转 mono，取前 640000 samples，不足右补零 |
| 多文件 | 这轮抽样限定单原始录音；ReasonAQA 的两个不同音频文件样本不进入候选池 |
| 题面 | `question a) choice b) choice ...`，原选项顺序，统一 lowercase |
| 问题 EOT | 不额外追加 `<\|endoftext\|>` |
| prompt 长度 | 右侧 padding/truncation 至 256 tokens，记录实际 token IDs 和截断信息 |
| prefix | 126 audio tokens＋1 个 SEP token＋256 prompt tokens＝383 tokens，hidden size 576 |
| token ID | PAD=`!`/17，SEP=EOS=0；保留固定 prompt 的可见 padding 语义 |
| 自由生成 | unrestricted greedy，temperature=0、top_p=1 的 argmax 语义，无字母约束、无候选限制 |
| 生成停止 | 遇 EOS=0 或生成 300 个新 token |
| 上下文上限 | 1024；候选正文不得因超长而静默截断 |
| 随机设置 | seed=1234、全模块 eval、关闭梯度和 TF32、固定并记录 attention backend |

官方原生是 FP32＋vLLM；本轮按用户要求改为 BF16，并为动态跳层和 teacher-forced likelihood 使用独立 PyTorch/Transformers 入口。两种变化都记录在协议中，不声称逐 bit 复现 vLLM。生成和 likelihood 共用同一个模型、mask、prefix 和精度策略。

为降低动态跳层/cache 索引风险，本轮 `use_cache=False`、`attention_mask=None`，生成每一步完整重算。temperature/top_p 不应用到 likelihood 的 logits 上。

## 3. BF16 的具体边界

1. 从原 FP32 checkpoint 严格加载，不改写源权重文件，也不绕过 release 校验。
2. CUDA 神经网络前向采用 BF16 autocast，覆盖 encoder、mapper/projector 和 LM 的适用算子；模型保持 eval。
3. 解码、波形重采样、STFT/log-mel 等音频前处理保留 FP32；norm 等数值敏感算子遵循 PyTorch 混合精度规则。
4. 缓存的最终 mapped audio tokens 和送入 LM 的 prefix 明确转换为 BF16；候选 token embedding 使用相同的 LM 输入边界。
5. 对正文预测位置的 logits 先转换为 FP32，再计算 `log_softmax` 和 token 分数求和。BF16 模型前向不意味着用 BF16 累加长文本分数。
6. 作业内检查全部 8 张卡的 BF16 支持，记录 Torch/CUDA/GPU/backend/autocast/cache dtype；不静默回退 FP16。

## 4. 100 题 manifest

### 抽样规则

- 默认 seed=1234，各数据集独立、无放回抽取 25 题；使用固定来源顺序和确定的随机算法。
- MMAU 来自项目已固定版本的 test-mini，MMAR 来自已固定的完整 1000 题来源，MMSU 来自现有三份 parquet。
- ReasonAQA 默认限定原始 test.json 的 `AudioCaps-MCQ.json`、`Clotho-MCQ.json` 两类，排除开放题、yes/no 和两个不同录音的样本。本计划不混用旧 sample5000 的 manifest。
- 候选须可解析出有限选项、具有唯一 GT 映射；候选正文只从原题选项解析，不能由 GT 反向构造。
- MMSU null/空选项不作为候选，保留有效选项的原始相对顺序及原 slot 映射；转换为本轮统一题面时使用连续的 a)、b) 等标签。
- 大小写转换后正文重复、GT 歧义、空题面、无法定位音频等情况在冻结前报告。确定的元数据过滤及排除数量进入审计；音频缺失或解码失败不静默补抽。
- 候选正文的完整长度在冻结前检查，超过 1024 总上下文上限则报告，不能截断后冒充完整正文评分。
- prompt 的 256-token 截断按官方原生规则执行，明确保存完整题面、实际输入和截断标志。

### manifest 至少保存

- schema、seed、抽样算法、来源版本/文件哈希、各数据集候选池与排除统计。
- 全局 0…99 索引、数据集、原 ID、原文件/分片/行号、音频路径或 parquet payload 定位。
- 原问题、原选项/slot、实际 prompt、规范化候选正文、GT 索引与原始 GT。
- prompt token IDs、每个正文 token IDs、选项长度、截断/歧义检查结果。
- 实际音频/波形指纹、预处理协议、checkpoint/tokenizer/source identity。
- manifest 自身的稳定内容哈希。

caption、GT、任务标签等均不进入模型输入。冻结后所有 mask 必须使用完全相同的 100 个 ID、token IDs 和音频条件，不能重新采样或跳题。

## 5. 预计算及复用

- 在同一个 GPU 作业内，指定 rank0 对已冻结的 100 题运行 encoder＋已训练 projector/mapper，各题仅编码一次。
- 缓存形状为每题 `[126,576]` 的最终音频 token，而不是某个 LM 层的 hidden states，也不是旧模型/旧精度产生的 CED cache。
- 可进一步缓存完整 `[383,576]` 的 BF16 LM 输入 prefix。embedding、separator 和 prompt 均不因删层 mask 变化，故可共享。
- 100 题 audio-token BF16 纯 tensor 约 13.8 MiB，完整 prefix 约 42.1 MiB，不计文件头；无需复杂的大规模缓存系统。
- 原子发布缓存及 ready manifest，其他 rank 仅在身份、哈希和维度验证后读取；一次准备后所有组合复用。
- 缓存身份绑定 checkpoint、tokenizer、manifest、前处理、源码及精度设置。中断恢复时只能复用同身份缓存。
- 首次在作业内验证相同 BF16 输入边界下，在线路径与缓存路径一致；不额外提交 GPU 检查作业。

## 6. Mask 和 8 卡执行流程

### Mask

- `drop_mask` 长度 30，1=跳过、0=执行，`drop_mask[0]=drop_mask[29]=0`，恰好两个 1。
- 使用原始层号穷举 `(i,j)`，`1 <= i < j <= 28`；按字典序固定 mask_id=0…377。
- 首个剪层组合是原始 0-based 层号 `[1,2]`。日志同时标注 1-based 层号 `[2,3]`，防止混淆。
- 整个 block 跳过时输出等于输入；不把 hidden states 置零、不临时清零参数、不反复拷贝 checkpoint。
- 自定义只读执行轨迹核对当前 mask 下应执行的 28 个原始层。旧 official evaluator 的强制 30 层 hooks 不用于此入口，原代码保留。

### 并行

- 单节点 8 个进程，每个进程绑定一张继承可见的 GPU，各加载一份同源 Mizar。无 DDP 梯度包装，无 tensor parallel。
- 固定分片 `global_index % 8 == rank`，rank0…3 每轮 13 题，rank4…7 每轮 12 题；不补齐到 104 题，不重复样本。
- 每轮所有 rank 收到并核对相同 mask_id、drop pair、manifest/cache identity。
- 每卡按自己固定的题目逐题计算候选 likelihood；首版每卡每次 1 题，正文评分不使用跨 mask 的历史状态。
- 汇总完成后必须核验精确 100 个唯一题目，四个数据集各 25；分数、正确数与实际执行 mask 一致。
- 总 accuracy=`sum(correct)/100`，子集 accuracy=`subset_correct/25`。不能平均 8 个 rank 的 accuracy，因为分片大小是 13/12。
- rank0 写逐题结果和本轮汇总，打印并 flush；全体同步后才进入下一 mask。任何 rank 失败均中止本轮，不提交部分分母分数。

## 7. Likelihood 口径

对条件 x（同一音频 prefix、问题和全部选项）及候选正文 c_j：

`score_j = sum_t log P(c_j[t] | x, c_j[:t])`

- 候选正文统一 lowercase，不带选项编号，不额外添加 BOS、EOS 或人为前缀空格；实际 token IDs 在 manifest 固定。
- 只累加正文 token。prompt、separator、padding、额外 EOS 不进入求和。
- 首个正文 token 由 prefix 最后位置（382）的 logits 预测，后续 token 按 causal shift 对齐；不能把答案 token 自己所在位置的 logits 错用为其概率。
- 所有正文 token 使用 teacher forcing；不是比较 freely generated output 与候选的相似度。
- `argmax(score_j)` 得到选项，再与 GT 索引比较。选项分数完全相同则按原选项顺序确定预测，并记录 tie；非有限值硬失败。
- 保存 raw sum、token 数及辅助 mean score，主预测及组合排名只使用 raw sum 产生的 accuracy。
- 不做长度归一化、PMI、答案字母评分、scorer 文本包含匹配或 generation accuracy 混合。

## 8. 第一批 100 题自由生成日志

这里将“第一批”明确为搜索中首个剪层组合 `[1,2]` 的全部 100 题，而非只生成每卡第一题，也不以不剪层基线替代。

- 同一道题、同一个 mask、同一份 cached prefix 分别做自由生成和候选 likelihood。
- 自由生成只拿 prefix，不输入 GT、不输入某个候选的 teacher-forcing 续文、不拿 likelihood 获胜选项冒充 output。
- 8 卡分别生成其 12/13 题，总计精确 100 条；生成上限与停止条件见官方参考设置。
- worker 保存完整记录，主日志由协调端完整逐条输出，标注全局索引、数据集、原 ID、rank、mask，防止多个进程的多行文本混杂。
- 每条打印问题/选项、自由生成 output、GT 正文及索引、likelihood 预测、各选项 sum logprob、是否正确、生成 token 数、停止原因。
- 同时保存 raw token IDs、保留特殊 token 的 raw decode，以及按官方方式跳过特殊 token 的可读 output。日志用 JSON escaped string/完整记录，不能只显示截断摘要。
- 自由生成仅供检查，不产生主 accuracy，不影响 mask 选择；后续 377 个组合只算 likelihood。
- 首批输出和 accuracy 日志完成后自动继续，不把整作业挂起等待人工确认。

示意日志（下列为格式占位，不是实测结果）：

```text
[FIRST_MASK_SAMPLE] index=<0..99> dataset=<name> id=<id> drop_0based=[1,2]
question=<question and options>
generated_output=<freely generated text>
gold=<index and full text>
likelihood_prediction=<index and full text>
option_logprobs=<all candidate sums> likelihood_correct=<true/false>
generated_tokens=<n> stop_reason=<eos/length>

[MASK_DONE] completed=<k>/378 drop_0based=<i,j> correct=<n>/100 acc=<percent>
mmau=<n>/25 mmsu=<n>/25 mmar=<n>/25 reasonaqa=<n>/25 elapsed_s=<seconds>
```

## 9. 基线、排名、恢复及产物

- 建议在同一个作业内另算一次全零 mask 的 30 层 likelihood 基线，安排在 378 个剪层组合之后，不替代上述首批生成。
- 因此候选排名始终是 378 行，基线单列；若包含基线，合计 379 次 100 题 likelihood 评估，只有首个剪层 mask 做 100 条自由生成。
- 按总正确数降序排名，同分保持并列。层号只用于稳定展示顺序，不以辅助 margin/长度平均分擅自打破主排名。
- 每轮落盘逐题选项分数和摘要；续跑核对完整身份，仅补未完成记录。部分轮次不能标为完成。
- 第一轮生成也有独立覆盖记录；恢复时不遗漏尚未生成的样本，不伪造已有输出。
- 产物建议包含 `manifest.json`、`cache_manifest.json`、`run_config.json`、`first_mask_generations.jsonl`、逐 mask 的 `likelihood.jsonl`、`mask_scores.jsonl`、`ranking.csv/json`、`baseline.json`、`search_report.json`、per-rank 日志与原始层号映射。
- 搜索只产出排名、mask 和可审计结果。本轮不自动生成 MeSH、不训练、不覆盖源 checkpoint。

## 10. 实现边界和验证

- 建议新增隔离包 `code/RSmol/mizar_layer_pair_search/` 及单作业 3090 提交入口；不修改旧 MMAU/MMAR/MMSU/ReasonAQA evaluator、训练源码及 release manifest。
- 复用现有数据读取、路径审计和 release 验证；新实现原生 Mizar 前处理适配、BF16 cache、masked LM 前向、likelihood 和同步调度。
- 使用仓库现有 `vc submit` wrapper、镜像和 `mellow_c8204d8` 环境，资源参考既有 8 卡模板：`pdgpu-3090 / 8 GPU / 1 node / 64 CPU / 256G RAM`。一个提交调用负责整个流水线，不为 mask 单独提交任务。
- CPU/静态检查覆盖抽样身份、候选/GT 隔离、378 个组合、13/12 精确覆盖、causal likelihood 对齐、正文求和范围、汇总与中断恢复等关键错误。
- 真实 BF16 支持、缓存一致性、mask 执行轨迹、代表性 mask 与实际删层的 logits/likelihood 等价检查放在同一作业内；BF16 数值容差明确记录，不将近似一致写成 bitwise 一致。
- 最终报告只有在 378 组各覆盖 100 题、首批 100 条生成齐全且所有身份/数值检查通过时才标记完成。
- 本地 Windows 只维护代码、测试与计划，经 GitHub 中转；生产 manifest/cache、权重、数据及输出留在远程 Linux，不在本地启动生产推理。

## 11. 解释结果时的限制

- 原生 Mizar 的正式 benchmark 是自由生成评分；本轮是 BF16 选项 likelihood 的删层选择实验，分数名称应明确区分。
- 100 题的总 accuracy 步长为 1%，每个 25 题子集为 4%；378 组的领先分数可能存在抽样偶然性，应保留并列并在未参与搜索的数据上复核。
- 已确认使用正文 logprob 之和，会有短答案偏好；保留长度及 mean 诊断，但不改变用户确认的主规则。
- 首个 mask 的自由生成质量可能受到该删层组合影响；生成日志用于暴露实际行为，不预设它必然与完整 30 层相同。
