# 完整 Mizar 在 MMAR1000 上逐层消融：八卡 5090

复制并隔离已有 `mizar_layer_analysis_common` / `mizar_layer_ablation` 可用实现。
原 MMAU ablation、CKA、audio cutoff、训练代码和历史结果均保留。
本地只维护源码，权重/生产数据/缓存/结果留在远程 Linux，通过 GitHub 同步代码。

## 实验定义

使用原始完整 Mizar final checkpoint，包括 encoder、已训练 projector 和全部 30 个原始 decoder block。
首先重新测完整 30 层 baseline，然后分别消融原始第 2～29 层，每次只消融一层。
共 **29 组、29,000 个题目决策**；每题分别计算其全部非空候选正文的 likelihood。

消融定义为 `h_out = h_in`：直接绕过整个选中 block，保留此前表示。
每个条件从原始模型层集合出发，不累积跳层；第 1、30 层保护。
baseline 执行全部 30 层，消融执行其余 29 层；每次 forward 验证原始层号轨迹。
音频在所有保留层正常可见，没有音频 cutoff。

原推理设置保持：

- FP32 源权重、BF16 autocast 前向、FP32 音频前处理/log_softmax/正文累加、TF32 off。
- SoundFile 解码，实际采样率 resample32k→mono→前 20 秒→右补零，不使用 TorchCodec。
- compact lowercase 题面，原选项顺序和标签槽位；问题不加 EOT；prompt256 右 padding/truncate，PAD17 可见。
- audio126＋SEP0＋prompt256＝prefix383，原 position IDs/RoPE，无 KV cache。
- **完整候选正文 token log 概率之和**取最大值选答案；不加标签、BOS/EOS 或额外前导空格。
  保存均值供诊断，主分数不按长度平均。同分取原序靠前的有效选项。不自由生成。

## 完整 MMAR 与四类能力

不抽样、不筛掉异常题，保留原顺序和 1000 个唯一 ID。直接使用原 `category`：

| MMAR 能力类别 | 固定题数 |
| --- | ---: |
| Signal Layer | 43 |
| Perception Layer | 404 |
| Semantic Layer | 412 |
| Cultural Layer | 141 |

这些是题目类别，图的横轴仍为被消融的模型层号。
默认读取 `/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR/MMAR-meta.json`，
音频路径相对于 `/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR/mmar-audio`。
缺音频、重复 ID、分类题数改变、tokenize/音频失败直接报错，不缩小分母或补抽。

主准确率与官方 matcher 诊断分数分别保存：

1. **主准确率**：likelihood 选出的选项与唯一规范化 GT 选项匹配。
   当前本地原标注有 5 题无法映射到单个完整选项（原行号 0-based：43、517、836、894、983），
   `gold_index=null`，保留在分母并计错；不猜测、更改或模糊映射原答案。
2. **官方诊断**：把同一 likelihood 选出的正文交原 MMAR `string_match`，不生成额外答案。
   该函数实现与官方保持一致，并与服务器 `MMAR/code/evaluation.py` 的函数 AST 对照，不执行外部脚本。
   词集合匹配可能接受多个不同选项，主图/排名不使用这一诊断分数。

原标注另有 4 题带空选项（原行号 28、61、876、910）。保留原题面及空选项标签槽位，不重标，
空槽 `choice_token_ids=[]`、`option_loglikelihood=null`、`option_mean_loglikelihood=null`，不参与 argmax。
非空选项必须有非空 token 和有限分数。不能把空正文分数设成 0，否则它会错误地胜出。
所有异常保存在 manifest 的 `dataset_audit` 及最终 `dataset_audit.json`。
服务器实际文件重新审计，不用本地备份代替生产数据。

## 提交：一个八卡 5090 作业

代码经 GitHub 完整同步后，在远程执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_layer_ablation_mmar_8gpu_5090.sh
```

固定 `pdgpu-5090 / 8 GPU / 64 CPU / 256G / 1 node`，一次 `vc submit` 完成全部 29 组。
镜像仍为 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`，自动激活 `mellow_c8204d8`。
`RSMOL_CONDA_BASE` 可覆盖默认 `/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3`。

作业内先 CPU 冻结完整 manifest，再启动一次 8-rank torchrun。
按 `global_index % 8` 每卡 125 题，每卡各自预计算本卡的 encoder＋projector prefix。
各题仅编码一次，29 组复用 BF16 prefix；不同条件不共享 decoder 层间状态。
八卡共同完成一个条件，rank0 核对 1000 题、落盘、打印总体/四类分数，barrier 后进入下一个条件。

只查看参数/资源，不读取资产、不提交：

```bash
bash run_mizar_layer_ablation_mmar_8gpu_5090.sh --check-args
```

也可以先在远程 CPU 终端单独生成 manifest（无 CUDA、无模型权重加载）：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash mizar_layer_ablation_mmar/run.sh manifest
```

默认 manifest：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/mizar_mmar1000_likelihood_v1/manifest.json
```

默认输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_mmar_ablation_bf16_8gpu5090_v1
```

自定义路径（manifest 和提交阶段使用相同来源参数）：

```bash
bash run_mizar_layer_ablation_mmar_8gpu_5090.sh \
  --manifest /absolute/path/mmar1000/manifest.json \
  --output-dir /absolute/path/mmar_ablation_output \
  --mmar-metadata /absolute/path/MMAR-meta.json \
  --mmar-audio-root /absolute/path/mmar-audio \
  --evaluation-script /absolute/path/MMAR/code/evaluation.py
```

## 日志与结果

提交终端直接打印确切 `log_path`、`output_dir` 和 `.submission.json` 路径。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
ls -lt log/mizar-layer-ablation-mmar.*.log
tail -n 100 -F "$(ls -t log/mizar-layer-ablation-mmar.*.log | head -n 1)"
```

`MMAR_INPUT_FROZEN` / `MMAR_MANIFEST_READY` 表示输入冻结；`PREFIX_READY` 表示各卡缓存就绪。
`BASELINE_DONE` 是完整 30 层 1000 题成绩；每条 `ABLATION_DONE` 是一层消融的完整分数，
含总体及四类正确数/题数/accuracy/baseline 差值。日志差值为比例，`0.01` 表示提高 1 个百分点。

| 输出 | 内容 |
| --- | --- |
| `conditions/baseline/likelihood.jsonl` | 完整 30 层 1000 题分数 |
| `conditions/ablate_02…ablate_29/likelihood.jsonl` | 各消融条件 1000 题完整候选分数、预测、主/诊断正确性及执行轨迹 |
| `conditions/*/summary.json` | 总体/四类/子类/模态/语言统计 |
| `conditions/*/paired_changes.jsonl` | 相对 baseline 新答对/新答错题目的分数、GT 和分类 |
| `condition_scores.json` | baseline-first 的全部 29 组汇总 |
| `baseline.json` / `category_baseline.json` | 本次总体与四类 baseline |
| `accuracy_by_layer.csv` | 按模型层排列的总体主分数 |
| `accuracy_by_category.csv` | 29×4 行四类主分数、baseline、百分点变化及 paired 统计 |
| `official_diagnostic.csv` | 单独的总体/四类官方 matcher 诊断分数 |
| `ranking.json` / `category_ranking.json` | 总体及四类的 28 个消融条件并列排名 |
| `accuracy_by_layer.png` / `.svg` | 总体准确率及变化双面板 |
| `accuracy_by_category.png` / `.svg` | 四类各一个面板，含 baseline 虚线和题数 |
| `delta_by_category.png` / `.svg` | 四类相对各自 baseline 的百分点变化 |
| `dataset_audit.json` | 分类分布、空选项、无法映射 GT 及官方评分歧义审计 |
| `cache/rank00…rank07/` | 各卡 125 个 BF16 prefix 缓存 |
| `ranks/` / `runtime_checks.json` / `run_config.json` | durable journals、加载/跳层等价检查与身份 |
| `analysis_report.json` | 最终 29×1000 覆盖和验证状态 |

主图均使用严格选项准确率，不把官方诊断混入主曲线。
Signal 只有 43 题，单题约 2.33 个百分点；图中固定显示各类题数。
如果环境没有 Matplotlib，数值结果仍保存，并明确标记绘图依赖缺失。
在有 Matplotlib 的 CPU 环境可重新绘图：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash mizar_layer_ablation_mmar/run.sh plot \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_mmar_ablation_bf16_8gpu5090_v1
```

同源码、模型、manifest、依赖和卡数下重提相同目录可逐题续跑，复用缓存。
完整 journal 坏行、错身份、漏题/重复题拒绝继续；只截断末尾未完成写入。
已有不同实验输出不可混用；源码/模型/输入变更时选择新路径。

## 验证范围

作业内各卡对原始第 2、29 层比较“直接绕过”和“真实执行后把输出替换为输入”，要求 BF16 logits 逐值相同。
本地验证入口 `tests/validate_mizar_layer_ablation_mmar.py --require-tensors`，
覆盖实际小型 30 层 Llama 的 eager/SDPA/BF16、baseline、全部跳层轨迹、正文 likelihood shift、
空选项不进入 forward、真实 prefix 缓存恢复、SoundFile stereo/resample/first20/pad。
完整 29,000 决策、8-rank 调度和 manifest 生成使用明确标记的合成夹具；另只读核对本地真实 MMAR 元数据和官方 matcher。
真实生产权重、远程音频和八卡 CUDA 结果须由远程作业验证，不把本地 CPU 检查当作 GPU PASS。
精确同步清单在本目录 `sync_files.txt`，验证报告位于 `audit_reports/mizar_layer_ablation_mmar_validation.json`。

本轮本地 **19/19 通过、无跳过**；Python3.10 语法和真实 Bash 的参数/资源解析通过。
本地版本为 Torch2.9.0+cpu、Transformers4.54.1、Torchaudio2.9.0+cpu、SoundFile0.14.0。
早期 Windows 沙箱读取隔离 Torch 失败后，获准在沙箱外复验，历史保存在报告。
合成曲线已目视核对布局；它们只用于绘图验证，不是生产实验成绩。
