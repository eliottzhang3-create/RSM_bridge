# 完整 Mizar 的逐层音频访问截断实验

对原始完整 30 层 Mizar，先测完整音频 baseline，再测进入第 3～20 层前屏蔽 audio＋SEP 的 18 个条件，共 19 组、1900 个题目决策。所有条件均执行完整 30 层，不采用此前删第 3、18 层的模型。

这是独立包，旧 `mizar_layer_pair_search`、训练、模型、数据解析、manifest 生成器及已有提交入口均不修改。Windows 本地维护代码，经 GitHub 同步到远程；权重、数据、manifest、缓存和正式输出留在远程 Linux。

## 提交命令

同步本次新增文件后，在远程执行以下两种入口之一。每条命令只提交一个完整作业。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

# pdgpu-4090 / 1 GPU / 8 CPU / 32G RAM / 单节点
bash run_mizar_audio_cutoff_4090.sh

# 或 pdgpu-3090 / 8 GPU / 64 CPU / 256G RAM / 单节点
bash run_mizar_audio_cutoff_8gpu_3090.sh
```

可追加 `--check-args` 只查看参数和资源，不读取资产或提交。已有 manifest 使用默认路径时无需其他参数；若之前用了不同路径：

```bash
bash run_mizar_audio_cutoff_4090.sh \
  --manifest /absolute/path/to/existing/manifest.json
```

默认 manifest：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/mizar_prune_calibration100_seed1234_v1/manifest.json
```

它必须是之前已经成功冻结的同一份 100 题。新入口只读取和校验，不重新抽样或分词；保存的 token IDs、音频指纹、原问题、选项和 GT 全部复用。若不存在，先使用原 manifest 生成命令准备；新作业不会静默生成替代样本。

沿用 `mellow_c8204d8` 环境、原 vc wrapper 和镜像。`RSMOL_CONDA_BASE` 可覆盖 conda 根，默认 `/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3`。所有 GPU 推理均在 vc 分配的作业内执行。

## 实验定义

输入保持 `[audio126][SEP1][prompt256][candidate_body]`。第 L 层表示从 1 开始计数的原始 decoder block；进入这一层前开始生效，持续至第 30 层。

| 条件 | 正常融合音频的层 | 使用屏蔽的层 |
| --- | --- | --- |
| baseline | 1～30 | 无 |
| cutoff_03 | 1～2 | 3～30 |
| cutoff_04 | 1～3 | 4～30 |
| … | … | … |
| cutoff_20 | 1～19 | 20～30 |

从截止层开始，所有 query 位置 ≥127（prompt、其原可见 padding、候选正文）不能访问 key 位置 0～126（126 个 audio＋SEP）。attention logits 对这些连接加负无穷。保留文本间的完整 causal mask；未来候选 token 不可见。audio/SEP 自己的 query 行保持合法的 causal attention，避免全行屏蔽产生 NaN；它们此后无法再影响文本输出。

SEP 在前面层也可能吸收音频，因此与 audio 一并屏蔽。文本 hidden states 保留前 L−1 层已经吸收的信息。这轮回答的是“直接访问音频需要持续到多深”，不代表文本从未利用音频。

保留序列长度、原生 position IDs 和 RoPE 编号，不删除或重新编号 token。原生 LM forward 不重写，只在选定 decoder 层入口替换 attention mask；baseline 的层入口完全不修改 mask。该实现用于分析准确率，不报告删除音频 token 的速度收益。

## 推理、评分和缓存

- 严格加载原 Mizar final checkpoint、全部 30 个不同 block、encoder 和训练后的 projector。官方资产 SHA 和模型结构审计沿用原实现。
- 源参数 FP32；前向 BF16 autocast；音频前处理、log_softmax 和正文 token 分数求和 FP32；TF32 关闭。
- prefix 仍为 383×576，原输入题面、SEP0、PAD17、可见 padding、音频 first20 和 SoundFile 解码均沿用冻结 manifest 协议。
- 完整候选正文 loglikelihood **求和**后 argmax，只对正文打分，不含标签/BOS/EOS/额外前导空格。保留 token 数和 mean 诊断，但主评分使用 sum。
- 无 KV cache，逐次完整重算。模型不会看到候选的未来 token。
- 此轮不做自由生成，19 组都只计算 likelihood。
- rank0 预计算 100 个 BF16 prefix，各题 encoder＋projector 一次，全部条件复用。独立缓存绑定本轮模型、manifest、代码、精度身份，避免混用旧搜索输出。
- 单卡每条件顺序覆盖全部 100 题；八卡按全局索引模 8 分成 13/12 题，共同完成同一个条件，汇总打印、持久化和 barrier 后才切换。

## 输出与查看日志

默认独立输出目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
  mizar_audio_sep_cutoff03_20_bf16_v1_1gpu4090/
  mizar_audio_sep_cutoff03_20_bf16_v1_8gpu3090/
```

提交日志使用与旧搜索不同的前缀：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
ls -lt log/mizar-audio-cutoff.*.log
tail -n 100 -F "$(ls -t log/mizar-audio-cutoff.*.log | head -n 1)"
```

`BASELINE_DONE` 表示本轮完整模型基线已完成；之后每组打印 `CUTOFF_DONE`，包含截止层、仍能访问音频的层数、正确数、总 accuracy、相对 baseline 的差值、四个子集统计、新答对/新答错的数量。100 题 accuracy 步长 1%，子集步长 4%；差值字段以比例表示，例如 0.10 表示提高 10 个百分点。

```text
run_config.json                         固定来源、协议和 1/8 卡拓扑身份
cache/                                  独立 BF16 prefix 缓存和校验清单
ranks/rank00 …                          每卡原始 journal、加载/运行/完成审计
runtime_checks.json                     所有卡的 baseline 对齐、边界隔离检查
conditions/baseline/likelihood.jsonl     无屏蔽 baseline 的 100 题分数
conditions/cutoff_03/likelihood.jsonl    对应条件的全部候选正文分数
conditions/*/summary.json               每个条件总/子集正确数和 baseline 差值
conditions/*/paired_changes.jsonl       新答对/新答错题目的题面、GT 和两组分数
condition_scores.json                   baseline-first 的全部 19 组汇总
accuracy_by_cutoff.csv                  按层排列的总体及四个子集曲线数据
ranking.json                            18 个屏蔽条件排名，同分并列
baseline.json                           本轮重新测得的完整模型 baseline
accuracy_curves.svg / .png              有 matplotlib 时自动生成两幅面板的曲线
analysis_report.json                    最终状态、完整覆盖数量及运行检查指纹
```

曲线由 Matplotlib 生成，虚线为各自 baseline，横轴为 1-based cutoff 层。若服务器环境缺少 Matplotlib，数值评测和 CSV 仍完整交付，报告明确标注绘图缺少依赖；可把 `condition_scores.json` 拷到有 Matplotlib 的环境，用下列独立入口绘图：

```bash
bash mizar_audio_cutoff/run.sh plot --output-dir /absolute/path/to/output
```

同一代码、manifest、精度和拓扑下重新提交同一输出目录可逐题续跑。完整 journal 坏行、身份改变、缺题/重复题会报错；末尾部分写入可恢复。不同拓扑使用各自目录，不合并进度。源码改变时使用新输出目录；旧剪层搜索产物保留。

## 验证边界

计算作业内，每张卡在正式 baseline 前执行：无屏蔽与原生 forward logits 逐 bit 一致；截止层 3、20 处固定文本状态、改变 audio＋SEP 状态，后续正文 logits 完全不变。每次实际前向核对 1～30 层执行顺序和应屏蔽的精确层范围。检查在同一个正式作业内完成，不额外提交 GPU smoke。

本地测试用随机初始化的小型真实 30 层 Llama，覆盖 eager/SDPA、全部 19 条件、causal attention、audio＋SEP 零 attention、边界状态隔离、早期音频信息保留、正文 likelihood 右移和参数不变。单卡/八卡条件调度及全量 1900 个决策汇总另外使用明确标记的合成数据。真实生产权重、数据、CUDA 和服务器结果须由远程作业验证。

```bash
# 从仓库根，在依赖齐全的 CPU 环境执行
python -B tests/validate_mizar_audio_cutoff.py --require-tensors
```
