# Mizar 音频访问截断：四类各 250 题，层 3～28

独立复制原 `mizar_audio_cutoff` 评测实现。原 100 题包、manifest 和结果不改。
使用原始完整 30 层 Mizar final checkpoint（包括 encoder 和训练好的 projector）。
不使用先前删去两层的模型，不训练权重。

## 实验合同

- MMAU test-mini、MMSU、MMAR、ReasonAQA 各 250 题，共 1000 题。
- seed=1234，各数据集独立无放回抽样，按原始来源顺序保存；不按能力类别分层抽样。
- 不强制包含或排除旧 100 题。可用 `--compare-manifest` 记录题目交集。
- 先运行完整模型 baseline，再运行 `cutoff_03`～`cutoff_28`。
  共 **27 组、27,000 个题目决策**；每题分别计算各候选正文的 likelihood。
- 层号从 1 开始。`cutoff_L` 表示进入第 L 层前开始，直到第 30 层：
  query 位置 ≥127（prompt、原可见 padding、候选正文）不能访问 key 位置 0～126（audio126＋SEP1）。
- 使用 additive `-inf` attention mask，文本间保持 causal attention；audio/SEP 自己的 query 行
  仍保留合法 causal key，避免整行全被 mask。文本保留前 L−1 层已吸收的音频信息。
- 所有 30 层仍执行；不删除 token，不改变序列长度、position IDs 或 RoPE。
- 源权重 FP32，encoder/mapper/LM 前向 BF16 autocast，音频前处理及 log_softmax/累加 FP32，TF32 off。
- 音频沿原生 Mizar：SoundFile 解码、32k 重采样后转 mono、前 20 秒、右补零。
  不调用需要 TorchCodec 的 `torchaudio.load`。
- 原紧凑小写题面，无问题 EOT；prompt256 右 padding/truncate，PAD17 可见；
  audio126＋SEP0＋prompt256＝prefix383；无 KV cache。
- **主分数为完整选项正文 token log 概率之和**，argmax 判答案；不加标签、BOS/EOS、额外空格。
  保存 token 数和均值供检查；均值不参与主评分。并列选第一个原有效选项。
- 本轮不自由生成。

候选资格沿原 100 题：重复/空的规范化选项、GT 不唯一、空 token 或完整正文超过上下文限制的题目排除。
MMSU null 选项删去并连续重标，原槽号保留。ReasonAQA 只使用 test 中
`AudioCaps-MCQ.json`、`Clotho-MCQ.json` 的单录音选择题。
选中后若音频缺失/无法解码则报错，不重新抽样补题。

## 远程生成 manifest（CPU）

本地仅维护代码；经 GitHub 同步后，在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/mizar_audio_cutoff1000/run.sh manifest
```

脚本自动激活 `mellow_c8204d8`，只读离线 tokenizer/config、标注和音频，不加载模型权重或使用 CUDA。
首次会读取元数据、筛选/tokenize、抽样、审计全部 1000 段音频并原子发布 manifest；已存在则校验复用。
MMSU 等 parquet 音频按 shard 流式扫描一次，保留选中行，不逐题重新扫描整个 shard。
日志依次包含 `MANIFEST_BUILD_START`、`MANIFEST_POOLS_READY`、`MANIFEST_SAMPLED`、
1000 条 `MANIFEST_AUDIO_CHECKED`，最终 `PASS_MANIFEST`。

默认 manifest：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/mizar_audio_cutoff_balanced1000_seed1234_v1/manifest.json
```

可用 `--manifest /new/path/manifest.json` 覆盖保存位置。其他数据路径参数见
`bash code/RSmol/mizar_audio_cutoff1000/run.sh manifest --help`。
自定义路径时，后续提交命令传入相同的 manifest 和数据参数。
可选旧题交集审计应在首次生成时指定，并在后续提交时保留同一参数：

```bash
bash code/RSmol/mizar_audio_cutoff1000/run.sh manifest \
  --compare-manifest /hpc_stor03/sjtu_home/jinwei.zhang/data/mizar_prune_calibration100_seed1234_v1/manifest.json
bash code/RSmol/run_mizar_audio_cutoff1000_8gpu_3090.sh \
  --compare-manifest /hpc_stor03/sjtu_home/jinwei.zhang/data/mizar_prune_calibration100_seed1234_v1/manifest.json
```

新 manifest 使用独立 schema 和源码/资产内容身份，不冒充旧 100 题 manifest，不放宽历史文件的校验。
冻结后不会因更换卡数而改变抽样。已有 manifest 的源码或音频变更会明确报错，需检查来源。

## 提交八卡或单卡作业

八卡 5090：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/run_mizar_audio_cutoff1000_8gpu_5090.sh
```

八卡 3090：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/run_mizar_audio_cutoff1000_8gpu_3090.sh
```

单卡 4090：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/run_mizar_audio_cutoff1000_4090.sh
```

每条命令只调用一次 `vc submit`。八卡 5090 使用 `pdgpu-5090 / 8GPU / 64CPU / 256G / 1node`；
八卡 3090 使用 `pdgpu-3090 / 8GPU / 64CPU / 256G / 1node`；
单卡使用 `pdgpu-4090 / 1GPU / 8CPU / 32G / 1node`。
镜像仍为 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`。
没有预先生成 manifest 时，作业内会先生成再启动评测。
如果准备同时提交两个拓扑，先用 CPU 命令生成共享 manifest，避免同时争用生成锁。

rank0 预计算每题 encoder＋projector 输出及完整 BF16 prefix，共 1000 份，跨 27 组复用。
各卡读取同一缓存；八卡按 `global_index % 8` 每卡 125 题，共同完成同一条件，
汇总/持久化/打印后 barrier，再切下一个条件。单卡每条件覆盖 1000 题。
原精度和缓存身份检查保留；单卡、八卡的输出和缓存独立，不合并进度。

只查看参数与提交资源，不触碰资产、不提交作业：

```bash
bash code/RSmol/mizar_audio_cutoff1000/run.sh manifest --check-args
bash code/RSmol/run_mizar_audio_cutoff1000_8gpu_3090.sh --check-args
bash code/RSmol/run_mizar_audio_cutoff1000_8gpu_5090.sh --check-args
```

## 日志、输出与恢复

提交时终端会打印真实日志路径、输出目录和 `.submission.json` 路径。
日志位于 `code/RSmol/log/mizar-audio-cutoff1000.<tag>.1.log`（JOB=1）。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
ls -lt log/mizar-audio-cutoff1000.*.log
tail -n 100 -F "$(ls -t log/mizar-audio-cutoff1000.*.log | head -n 1)"
```

`BF16_PREFIX_READY` 表示缓存就绪；`BASELINE_DONE` 和每条 `CUTOFF_DONE` 都表示该条件完整 1000 题已合并。
输出总体/四子集正确数、准确率、baseline 差值和新答对/新答错数量。
总分每题为 0.1 个百分点，子集每题为 0.4 个百分点。
差值字段为比例，`0.01` 表示 1 个百分点。

默认输出目录分别为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_audio_sep_cutoff03_28_1000_bf16_v1_8gpu3090
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_audio_sep_cutoff03_28_1000_bf16_v1_8gpu5090
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_audio_sep_cutoff03_28_1000_bf16_v1_1gpu4090
```

| 文件 | 内容 |
| --- | --- |
| `run_config.json` | 模型、数据、代码、依赖及卡数身份 |
| `cache/` | 1000 份 BF16 prefix 与 SHA 库存 |
| `ranks/rankXX/` | 每卡逐题 journal、模型/运行检查、完成或失败审计 |
| `runtime_checks.json` | 所有卡的原生 baseline 对齐与第 3、28 层隔离检查 |
| `conditions/{baseline,cutoff_03…cutoff_28}/likelihood.jsonl` | 各条件 1000 题完整候选分数 |
| `conditions/*/summary.json` | 总/四子集分数与 baseline 差值 |
| `conditions/*/paired_changes.jsonl` | 新答对/新答错题目、GT 与两组候选分数 |
| `condition_scores.json` / `accuracy_by_cutoff.csv` | 按层排列的 27 组汇总 |
| `ranking.json` / `baseline.json` | 26 个 cutoff 并列排名及 baseline |
| `accuracy_curves.png` / `.svg` | 总体/四子集准确率与相对 baseline 差值曲线 |
| `analysis_report.json` | 最终状态与 27×1000 覆盖检查 |

同代码、manifest、依赖、精度和卡数下，重新提交相同输出目录会从每卡逐题 journal 续跑，复用已完成 prefix。
仅恢复未完成的末尾写入，完整坏行/错身份/漏题/重复题报错。源码或卡数改变时使用新输出目录。
启动早期错误看作业日志，worker 错误同时写入 `ranks/rankXX/failure.json`。

数值评测不强制安装 Matplotlib；若缺少，报告会标记缺依赖，CSV/JSON 保留。绘图可在已有 Matplotlib 的 CPU 环境执行：

```bash
bash mizar_audio_cutoff1000/run.sh plot --output-dir /absolute/path/to/output
```

## 本地验证

验证入口 `tests/validate_mizar_audio_cutoff1000.py --require-tensors`。
真实小型 30 层 Llama 覆盖 eager/SDPA、27 条件、BF16 prefix383、mask/因果性、
第 3/28 层状态隔离、原生 baseline 逐值对齐、正文 sum likelihood 右移及参数保持。
抽样、manifest builder、1/8 卡调度、恢复及完整 27,000 决策汇总使用明确的合成夹具。
本地 CPU 验证不代表生产权重、远程数据、CUDA 或八卡作业已验证。

本轮验证 26/26 通过、无跳过；真实 Bash 解析/单卡与八卡资源检查通过。
报告为 `audit_reports/mizar_audio_cutoff1000_validation.json`，保留早期测试夹具错误及修复后的复验记录。
5090 支持只调整提交层，复用已有 1000 题 manifest；指定 `--output-dir` 时保留用户路径。
Git 同步文件清单为本目录 `sync_files.txt`，包含新包、三个 launcher、测试、报告和 RM 记录。
