# Mizar 单层 identity ablation：完整 MMAU test-mini 1000 题

这是与 CKA 分别提交、分别保存结果的任务。原始 Mizar 全部 30 层权重严格加载；先测完整 baseline，再分别跳过原始第 2～29 层，共 29 个条件、29,000 个题目决策。

## 提交

本地代码经 GitHub 同步到远程。一起同步 `mizar_layer_analysis_common/`、`mizar_layer_ablation/` 和顶层提交入口；原训练、剪层搜索、音频 cutoff 代码及结果保留。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_layer_ablation_8gpu_3090.sh
```

只提交一个完整作业：`pdgpu-3090 / 8 GPU / 64 CPU / 256G RAM / 1 node`，原 vc wrapper、镜像和 `mellow_c8204d8`。附加 `--check-args` 只查看资源与参数。

这轮不使用原 100 题清单代替全量 MMAU。默认来源为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini/mmau-test-mini.json
/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini/test-mini-audios/
/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini/test_mini.parquet
```

作业内 CPU 首先冻结完整 1000 行输入清单，默认产物为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/mizar_mmau1000_likelihood_v1/manifest.json
```

冻结包含原始 ID/行序/选项槽位、题面、候选正文、GT、实际 token IDs、音频来源/波形指纹、任务与难度分布及文件 SHA。每次都要求 1000 个唯一 ID、music334/sound333/speech333；缺音频或坏样本直接失败，不过滤、不补抽、不缩小分母。

音频优先按 ID 使用 WAV；缺失时按 ID 索引 parquet 音频，并验证 JSON/parquet ID 集合一致。SoundFile FP32 解码，实际采样率 resample32k→mono→first20→右补零，避免 TorchCodec 依赖。可用 `--mmau-metadata`、`--mmau-audio-root`、`--mmau-parquet` 覆盖输入路径；首次冻结后须用相同来源参数复用，改来源用新 manifest 路径。

```bash
bash run_mizar_layer_ablation_8gpu_3090.sh \
  --manifest /absolute/path/new_mmau1000/manifest.json \
  --output-dir /absolute/path/new_ablation_output
```

## 消融与评分定义

原始层号从 1 开始。条件 `ablate_L` 将第 L 层置为恒等映射：

```text
正常：h_out = Block_L(h_in)
消融：h_out = h_in
```

实现直接绕过这个 block，保留此前表示，不清零整个 hidden state，不去除其他层的残差连接，不修改源参数。baseline 实际执行 30 层；消融实际执行其余 29 个原始层，第 1、30 层保护。原始层号和每次实际执行轨迹均记录；无音频 cutoff，audio 在所有保留层正常可见。

主评分延续之前的完整候选正文 loglikelihood **求和**，FP32 log_softmax 和累加后 argmax。题面/候选正文小写，正文去字母标签、不添加 BOS/EOS/前导空格，首正文 token 使用 prefix 末位置的 logits，后续正确右移。不做长度平均主评分或生成文本评分，也不做自由生成。

原始 1000 题里可能存在重复选项正文；必须保留原题和所有原槽位，不能沿用小样本抽样时的排除规则。GT 先按唯一的原始选项文本（包含原标签时一起匹配）定位；否则固定取第一个匹配正文的槽位。同分预测取原序靠前的选项，保存 `option_score_tie`、重复正文标记及所有与 GT 正文相同的槽位，不私自改变分母或重标原始标签。该口径及有限选项的短答案偏好均会保存在协议中。

FP32 源参数、BF16 autocast、前处理 FP32、TF32 off、eval、seed1234、原可见 padding、原 position IDs/RoPE、无 KV cache保持。每卡预计算本卡125题的 encoder＋projector prefix，一次缓存后29组复用；不同条件不共享 LM 层间状态。

每个条件八卡共同运行，按全局索引模8各125题；rank0 验证精确1000题后，落盘/打印并 barrier，才进入下一条件。一个作业处理全部条件，不按层提交28个任务。

## 结果与日志

默认独立输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_ablation_bf16_8gpu3090_v1/
```

```text
conditions/baseline/likelihood.jsonl     完整模型1000题的全部选项分数
conditions/ablate_02/likelihood.jsonl    跳过第2层的1000题；一直到ablate_29
conditions/*/summary.json               accuracy、baseline差值、原任务/难度/类别统计
conditions/*/paired_changes.jsonl       新答对/新答错的题面、GT及两组候选分数
condition_scores.json                   baseline-first的29组汇总
accuracy_by_layer.csv                  按原始层号排列的主分数和差值
ranking.json                           28个消融条件排序，同分并列
baseline.json                          本次重新测得的1000题baseline
accuracy_by_layer.png / .svg            总体及music/sound/speech曲线、baseline差值
cache/rank00 … rank07/                  每卡125题的独立BF16 prefix缓存
ranks/                                 原始durable journals与每卡加载/等价/完成审计
runtime_checks.json / run_config.json   实际GPU数值检查与完整运行身份
analysis_report.json                   最终覆盖校验及状态
```

Matplotlib 缺失时矩阵/CSV与原始分数仍完整保留，报告明确标记绘图可选依赖缺失。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
ls -lt log/mizar-layer-ablation.*.log
tail -n 100 -F "$(ls -t log/mizar-layer-ablation.*.log | head -n 1)"
```

日志 `MMAU_INPUT_FROZEN` 为首次输入准备，`PREFIX_READY` 为每卡缓存，`BASELINE_DONE` 为完整模型1000题结果，`ABLATION_DONE` 为每层消融分数。accuracy差值保存为比例，例如0.01是提高1个百分点。

每题 journal flush/fsync，末尾未完整写入可恢复；完整坏行、输入/模型/代码/依赖变化均拒绝混用。相同身份重新提交同一目录可逐题续跑，不重算已完成分数；不与 CKA 合并进度。

本地验证使用真实小型30层Llama对照直接跳过与执行后输出替换输入的等价性，并核对所有28层的实际轨迹、BF16、源参数不变、1000题分母及29,000决策汇总；正式作业内每张GPU还会对第2、29层做真实BF16等价检查，无额外GPU smoke作业。生产权重和远程GPU结果仍由正式运行验证。

```bash
# 仓库根，依赖齐全的CPU环境
python -B tests/validate_mizar_layer_analysis.py --require-tensors
```
