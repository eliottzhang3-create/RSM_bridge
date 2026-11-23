# Mizar 三十层 CKA：原冻结 100 题

独立任务，对原始完整 Mizar final 模型执行 prefix-only 前向，采集全部 30 个 decoder block 原始输出。没有删层或 audio cutoff；不输入候选正文续文、GT，不做生成。每题只提取一次，八卡合计 100 题。

## 提交

本地维护代码，经 GitHub 同步到远程。请一起同步 `mizar_layer_analysis_common/`、`mizar_layer_cka/` 和顶层入口；公共包依赖项目中原有的 Mizar 加载、SoundFile 音频和冻结 manifest 兼容实现。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_layer_cka_8gpu_3090.sh
```

单个 `vc submit` 作业：`pdgpu-3090 / 8 GPU / 64 CPU / 256G RAM / 1 node`，原镜像和 `mellow_c8204d8` 环境。附加 `--check-args` 可只预览路径与资源，不读取资产或提交。

默认读取之前成功冻结的同一份 manifest，不重新抽样、分词或改写：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/mizar_prune_calibration100_seed1234_v1/manifest.json
```

若原来使用了其他 manifest：

```bash
bash run_mizar_layer_cka_8gpu_3090.sh \
  --manifest /absolute/path/original100/manifest.json
```

历史 MMSU prompt-helper 整文件 SHA 的已知变更沿用 `mizar_audio_cutoff.frozen_manifest` 的精确版本兼容校验。原 100 个 ID、token IDs、音频指纹和 manifest 自身 SHA 保持。

## 表示与计算

每题模型输入仍为 `[audio126][SEP1][prompt256]`，共 383 个 token。原可见 padding、position IDs/RoPE、FP32 源权重、BF16 autocast、FP32 音频前处理和 TF32 off 均保持。参数冻结，模型 eval，无 KV cache。

三种表示各得到 `[30,100,576]` 的 FP32 特征：

| 视图 | 每层、每题的表示 |
| --- | --- |
| `audio_mean` | 126 个 audio 位置的均值 |
| `valid_text_mean` | prompt 内实际有效 token 的均值，统计时排除 padding 和 SEP |
| `answer_readout` | prefix 最后位置 382 的状态，正是预测首个答案 token 的位置 |

有效文本位置使用原 tokenizer 保存的有效位置 mask，而不是根据 token ID 17 猜测；这只作用于统计池化，不改变 forward 的可见 padding。位置 382 即使是 PAD 仍参与 readout 分析。第 30 层与其他层一样，取完整 block 输出、最终 LM norm 之前的状态，避免把两种取点混用。

每题的 30 层三种池化向量即时保存，不保存全部 token states。八卡按全局索引模 8 分题，四卡各 13 题、四卡各 12 题。合并精确 100 个样本并恢复原顺序后，CPU 用 FP64 计算；不平均各 GPU 上分别算出的 CKA。

输出 standard linear CKA 和基于无偏 HSIC U-centering 的 debiased linear CKA，各三张 30×30 矩阵。CKA 比值本身不宣称严格无偏；去偏值允许为负，不强制裁到零。退化特征导致分母为零会明确报错。

每张矩阵另保存按题目对应关系随机置换的 null mean/std，默认 `--permutation-repeats 100`、seed=1234。用于识别有限样本、高维表示下标准估计的偏高，不把其当作因果重要性或显著性结论。

方法参考：[Kornblith et al., ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html)、[作者官方实现](https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb)。本地测试与官方公式逐项对照。

## 输出、日志和恢复

默认输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_cka_bf16_8gpu3090_v1/
```

```text
run_config.json                         固定模型、输入、代码、依赖和实验协议
cache/rank00 … rank07/                  每卡分片的 BF16 prefix 缓存
features/features_000.pt …              每题 [3,30,576] 的 FP32 池化向量与校验元数据
pooled_features.npz                     三个 [30,100,576] 特征数组
cka_matrices.npz                        三视图×两估计器，共六个 [30,30] 矩阵
<view>_<standard|debiased>.csv           矩阵；行、列均是原始第1～30层
*_permutation_mean.csv / *_std.csv      对应置换 null 均值和标准差
*.png / *.svg                          六张热图及相邻层 CKA 曲线
ranks/                                 每卡加载、取点/native对齐、完成审计
runtime_checks.json / artifacts.json    运行与产物指纹
analysis_report.json                   最终状态与 100 题完整覆盖
```

绘图使用 Matplotlib；若远程缺少它，完整矩阵、特征和 CSV 仍保存，报告明确标记可选绘图依赖缺失。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
ls -lt log/mizar-layer-cka.*.log
tail -n 100 -F "$(ls -t log/mizar-layer-cka.*.log | head -n 1)"
```

日志 `PREFIX_READY` 为音频缓存，`CKA_FEATURE_READY` 为单题取层完成，`CKA_MATRIX_DONE` 为矩阵完成。中断后用同一代码、参数和目录重投，可复用已校验的 prefix/特征；不对已完成题重新编码。代码、依赖、模型或 manifest 改变时用新输出目录。

公共验证入口（仓库根、依赖齐全的 CPU 环境）：

```bash
python -B tests/validate_mizar_layer_analysis.py --require-tensors
```

真实 CPU 小 Llama 和 CKA 数值验证不代表生产权重/八卡 CUDA 已通过。正式作业内每张卡再确认取层 hooks 与原生 baseline 的 logits 逐 bit 一致，随后才提取正式样本。
