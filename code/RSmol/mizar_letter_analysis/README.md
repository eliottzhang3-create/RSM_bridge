# 三个重做实验：小写选项字母 likelihood＋题面 EOT，八卡 5090

新包独立于原正文 likelihood 的 cutoff、MMAU ablation 和 MMAR ablation。
复用已验证的原始 Mizar strict loader、audio＋SEP mask、单层 bypass 和音频处理代码，旧源码和结果保留。
本地仅维护源码，模型权重、生产音频、manifest、缓存和输出都在远程 Linux；通过 GitHub 同步代码。

## 这次改变什么

用户确认只比较**纯小写字母** `a/b/c/d`，不含括号、空格、正文、BOS 或答案 EOS。
每题对同一 prefix 做一次 LM forward，取首答案位置 `logits[0,382,:]` 的 FP32 全词表 log_softmax，
读取各有效选项字母的分数，argmax 选答案。同分选原序靠前的有效选项。
每个字母必须编码为一个不同的普通 token；实际 release tokenizer 在生成 manifest 时严格核查。
题面仍包含原始问题和全部选项正文，不添加 only-letter 指令，不改变选项顺序。

将原保存的 compact lowercase 题面整体加上 ` <|endoftext|>`，然后按正式 `tokenize` 同样方式
tokenize、右截断/补齐到 256。不是在 padding 后添加 EOT，也不是把 EOT 放在问题与选项之间。
沿用 256-token 上限；极长题面可能把后追加的 EOT 截掉，
manifest 的 `prompt_eot_retained` 和 `input_audit.prompt_eot_retained_rows` 明确记录，不静默声称全部保留。

其他设置保持：

- 同一原始完整 Mizar final checkpoint、encoder＋已训练 projector，全部 30 个原始 block。
- FP32 源权重、encoder/projector/LM BF16 autocast，音频前处理和 log_softmax FP32，TF32 off。
- SoundFile 解码、先 resample32k 再 mono、**前 20 秒**截取、右补零，不改为中心裁剪。
- audio126＋SEP0＋prompt256＝prefix383×576，PAD17 可见，原 position IDs/RoPE，无 KV cache。
- 不自由生成；选出的字母映射回原有效选项，再按原各任务 GT 规则计算 accuracy。

## 三个任务与原题单

| task | 条件 | 题单 | 主要分组 |
| --- | --- | --- | --- |
| `cutoff` | 完整 baseline＋进入第 3～28 层起屏蔽 audio＋SEP，共 27 组 | 原四类各 250 的相同 1000 题 | MMAU/MMSU/MMAR/ReasonAQA 各 250 |
| `mmau` | 完整 baseline＋分别 bypass 第 2～29 层，共 29 组 | 原完整 MMAU test-mini 1000 题 | perception331／semantic230／reasoning439 |
| `mmar` | 完整 baseline＋分别 bypass 第 2～29 层，共 29 组 | 原完整 MMAR 1000 题 | Signal43／Perception404／Semantic412／Cultural141 |

总计 85,000 个题目决策，三个独立作业。
每个任务先重新计算自己的完整 30 层 baseline。消融层号从 1 开始。
cutoff 从选定层起直到第30层，文本 query≥127 不能读取 key0..126；不删除 token 或 block。
单层 ablation 是 `h_out=h_in`，不累计跳层，第1/30层保护；音频在保留层正常可见。

**不重新抽样、不改变题目 ID 或题序。** 新 manifest 从原冻结 manifest 转换，并保存原 manifest SHA、
每条原 record SHA、原正文 token IDs 供审计；原音频引用/波形指纹、题面正文、选项、GT 和异常规则保持。
已有原 manifest 时只读。若原 manifest 尚不存在，先调用对应原生成器生成，再转换；原生成器资格/seed不变。

原 manifest 默认分别为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/mizar_audio_cutoff_balanced1000_seed1234_v1/manifest.json
/hpc_stor03/sjtu_home/jinwei.zhang/data/mizar_mmau1000_likelihood_v1/manifest.json
/hpc_stor03/sjtu_home/jinwei.zhang/data/mizar_mmar1000_likelihood_v1/manifest.json
```

新 manifest 默认分别为 `data/mizar_{cutoff,mmau,mmar}1000_letters_eot_v1/manifest.json`。
每个任务独立生成加 EOT 的 prefix 缓存，身份与原正文实验分开；重提同一新任务可复用新缓存并逐题续跑。

MMAU 用户指定的 27 子类→8/8/11 三组精确映射复制在 `mmau_mapping.json`。
组分数按组内正确题数之和/题数之和计算，不平均子类 accuracy。
MMAU 原重复正文选项保留，GT 仍沿原唯一原文匹配优先的槽位规则；不同槽位现在对应不同字母。
cutoff 中 MMSU null 删除后的连续重标沿原 manifest 保持。
MMAR 不是全部四选题，实际保留 `a..b`、`a..c`、`a..f` 等真实选项范围，不凭空加选项或裁掉 e/f。
空选项保留原槽位且分数为 null，不参与 argmax；5 个无法映射 GT 的题继续保留在1000分母并计错。
MMAR 另外保存原官方 matcher 对所选选项正文的诊断分数，不用该诊断替换主字母 accuracy。

## 三条提交命令

本轮代码完整经 GitHub 同步到远程后：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

# 原四类各250、共1000题：audio＋SEP cutoff 3～28＋baseline
bash run_mizar_audio_cutoff1000_letters_eot_8gpu_5090.sh

# 完整MMAU1000：单层消融2～29＋baseline，三类能力曲线
bash run_mizar_layer_ablation_mmau_letters_eot_8gpu_5090.sh

# 完整MMAR1000：单层消融2～29＋baseline，四类能力曲线
bash run_mizar_layer_ablation_mmar_letters_eot_8gpu_5090.sh
```

每条命令只提交一个 `vc` 作业，固定 `pdgpu-5090 / 8 GPU / 64 CPU / 256G / 1 node`。
原镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`，自动激活 `mellow_c8204d8`。
`RSMOL_CONDA_BASE` 可覆盖默认 `/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3`。
八卡按全局索引模8每卡125题，共同完成同一个条件，合并完整1000题、落盘/打印/barrier后切换。
cutoff 仍由 rank0 预计算1000个prefix，各卡读取；两个 ablation 仍各卡预计算本卡125个prefix。
各题 encoder＋projector 在本任务缓存生成时计算一次，跨全部条件复用。

附加 `--check-args` 可只预览资源和协议，不访问资产、不提交。
自定义旧 manifest 必须通过 `--source-manifest` 指定；`--manifest` 是新字母版 manifest：

```bash
bash run_mizar_audio_cutoff1000_letters_eot_8gpu_5090.sh \
  --source-manifest /absolute/path/original1000/manifest.json \
  --manifest /absolute/path/new_letters_eot/manifest.json \
  --output-dir /absolute/path/new_cutoff_letters_output
```

也可先单独在远程 CPU 终端转换三个 manifest；此步骤不加载模型权重或使用 CUDA：

```bash
bash mizar_letter_analysis/run.sh cutoff manifest
bash mizar_letter_analysis/run.sh mmau manifest
bash mizar_letter_analysis/run.sh mmar manifest
```

已有原 manifest 时，来源路径以原 manifest 为准。源 model-dir 必须与新任务的 tokenizer model-dir 一致。
原 manifest 缺失时，可用 `--mmau-metadata`、`--mmau-parquet`、`--mmau-audio-root`、`--mmar-metadata`、
`--mmar-audio-root`、`--mmsu-dir`、`--reasonaqa-test`、`--audiocaps-root`、`--clotho-root`、`--evaluation-script`
覆盖原生成器的数据位置。

## 输出和日志

默认位于 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/` 下：

```text
mizar_audio_sep_cutoff03_28_1000_letters_eot_bf16_8gpu5090_v1
mizar_mmau_ablation_letters_eot_bf16_8gpu5090_v1
mizar_mmar_ablation_letters_eot_bf16_8gpu5090_v1
```

日志分别使用 `mizar-letters-cutoff`、`mizar-letters-mmau`、`mizar-letters-mmar` 前缀。
提交终端打印确切 `log_path`、输出目录和 `.submission.json`。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
ls -lt log/mizar-letters-*.log
tail -n 100 -F "$(ls -t log/mizar-letters-mmau.*.log | head -n 1)"
```

`LETTER_EOT_MANIFEST_READY` 显示来源 SHA、分组题数和 EOT/截断审计；`PREFIX_READY` 或 `BF16_PREFIX_READY` 是缓存。
`BASELINE_DONE` 为完整模型；每组 `CUTOFF_DONE` / `ABLATION_DONE` 打印整体和相应四子集/三能力/四能力分数。
主准确率步长为0.1个百分点；组内步长由分母决定。日志 `delta_vs_baseline` 为比例，CSV还显式保存百分点差值。

共同输出：

| 文件 | 内容 |
| --- | --- |
| `conditions/*/likelihood.jsonl` | 每题各有效字母 loglikelihood、字母/token ID、预测字母/正文、GT、实际层轨迹 |
| `conditions/*/summary.json` | 整体、主要分组和原字段子类统计 |
| `conditions/*/paired_changes.jsonl` | 相对baseline新答对/新答错题目的两组字母分数 |
| `condition_scores.json` / `baseline.json` / `group_baseline.json` | 全条件及完整模型基线 |
| `accuracy_by_layer.csv` / `accuracy_by_group.csv` | 按层及组排列的正确数/分母/accuracy/baseline差值 |
| `ranking.json` / `group_ranking.json` | 总体及各组并列排名 |
| `input_audit.json` | 同原题单证明、EOT保留、长题截断、空槽/未映射GT审计 |
| `accuracy_by_layer.png` / `.svg` | 总体准确率及相对baseline变化 |
| `cache/` / `ranks/` | 新prefix缓存、durable journal和各卡审计/失败记录 |
| `runtime_checks.json` / `run_config.json` / `analysis_report.json` | 数值检查、完整身份和最终覆盖状态 |

cutoff 另有 `accuracy_by_cutoff.csv`、`accuracy_by_group.png/svg`；
MMAU 另有 `psr_scores.csv`、`group_mapping.json`、**`psr_accuracy_by_layer.png/svg`**；
MMAR 另有 `official_diagnostic.csv`、`accuracy_by_group.png/svg`、
**`accuracy_by_category.png/svg`** 和 **`delta_by_category.png/svg`**。

缺 Matplotlib 时保留数值文件并明确标记缺依赖，可在有 Matplotlib 的 CPU 环境重画：

```bash
bash mizar_letter_analysis/run.sh mmau plot --output-dir /absolute/path/new_mmau_output
```

同新协议、输入、模型、精度、卡数、代码和依赖下重提同目录可续跑。
不复用旧正文实验的进度或prefix；新任务间也不合并进度。
完整坏行、错身份、漏题/重复题拒绝继续；仅恢复末尾未完成写入。

## 本地验证

`tests/validate_mizar_letter_analysis.py --require-tensors` 验证真实小型30层Llama eager/SDPA/BF16，
首答案位置383prefix的单次forward字母gather、无正文输入、全部27/29条件、原生baseline及隔离/identity等价、参数保持。
冻结输入转换/旧文件字节不变、三类映射与真实本地MMAR四类异常审计、两种真实BF16缓存复用；
8rank调度及三个任务完整85,000决策使用明确合成分数/运行证据夹具。
本地CPU通过不代表远程生产权重/音频/CUDA/5090运行通过；本轮不代为commit/push或提交GPU作业。
报告 `audit_reports/mizar_letter_analysis_validation.json`，精确同步清单在 `sync_files.txt`。

本轮本地 **19/19 通过、无跳过**，三个任务完整85,000决策的合成汇总及PNG/SVG生成通过。
真实 Bash 的三个入口和5090资源参数检查通过，没有实际调用 vc。
本地使用 Torch2.9.0+cpu / Transformers4.54.1 / Matplotlib3.11.2，未更改生产环境。
Windows沙箱无法正常读取隔离Torch依赖，获准在沙箱外复验；历史失败保留在报告。
合成图已经目视核对布局，不作为模型实测成绩。
