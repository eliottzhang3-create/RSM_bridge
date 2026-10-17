# formal30 三种文本骨干的统一 MMAU / MMAR 评测

只用于 `run_mellow_official_reasonaqa_gbs256_formal30_{,5_10x2_5_mesh_,5_10x4_5_mesh_}5090.sh` 三条路线的完整 30-epoch schema-v2 音频 checkpoint。新增代码在此目录，提交 wrapper 为上一级 `submit_mellow_formal30_eval_5090.sh`。训练代码和历史 evaluator 不作修改。

## 正式提交

代码通过 GitHub 同步到远端后，在远端 `code/RSmol/` 目录执行。以下每条命令提交两个独立作业：MMAU test-mini、MMAR；合计六个作业。没有 smoke gate，直接遍历完整数据并运行两套评分。

```bash
bash submit_mellow_formal30_eval_5090.sh both smollm2
bash submit_mellow_formal30_eval_5090.sh both mesh_x2
bash submit_mellow_formal30_eval_5090.sh both mesh_x4
```

三种别名在 [checkpoints.tsv](checkpoints.tsv) 中对应用户指定的 `model--epo-30.ckpt` 绝对路径。x2/x4 必须先实际产出完整 checkpoint，wrapper 不等待尚未完成的训练，也不会改用较早 epoch。

固定提交资源：`pdgpu-5090 -c 8 -m 32G -g 1 -n 1`，沿用训练容器与 `mellow_c8204d8` conda 环境。`MELLOW_CONDA_BASE`、`MELLOW_CONDA_ENV` 可指定已有环境；不自动安装依赖。所有模型比较时使用同一软件环境。单任务或自定义 checkpoint 的语法：

```bash
bash submit_mellow_formal30_eval_5090.sh mmau_test_mini smollm2
bash submit_mellow_formal30_eval_5090.sh mmau_test_mini smollm2_epo25
bash submit_mellow_formal30_eval_5090.sh mmau_test_mini smollm2_epo15

`smollm2_epo25` and `smollm2_epo15` evaluate the intermediate checkpoints from
the same 30-epoch run. The wrapper prints the checkpoint argument and the
resolved absolute path; the report stores both values. The unified MMAU audio
adapter averages channels 0 and 1 as the training dataset does, and sends a
zero waveform through the native second audio slot.

These changes use the fresh default output namespace
`eval_formal30_unified_v3`, so reports from the previous audio-slot protocols
cannot be resumed under this protocol.
bash submit_mellow_formal30_eval_5090.sh mmar mesh_x4
bash submit_mellow_formal30_eval_5090.sh mmar /absolute/path/to/model--epo-30.ckpt \
  --runtime-config /same/run/root/runtime_30epochs.yaml \
  --output-dir /absolute/path/to/new/eval_output \
  --dataset-dir /hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR
```

`both --output-dir PATH` 时 PATH 下分别创建 `mmau_test_mini/`、`mmar/`。`--dataset-dir` 仅适用于单任务。

## 共同评测合同

| 项目 | 三种模型共同设置 |
|---|---|
| 精度、设备 | 全部浮点参数/buffer、音频、prefix、logits 均 FP32；CUDA 0；关闭 autocast 和 TF32 |
| 模式 | 全模块 eval / inference mode；每次一条样本 |
| 音频 | 32 kHz、10 秒；两个原生槽分别编码；第二槽 waveform 为全零静音，静音特征仍正常通过 HTSAT/c2l/投影 |
| MMAU audio1 | 优先使用官方 `<id>.wav`；若镜像没有该 convenience WAV，则沿用旧 evaluator 回退到 parquet 内嵌 `context.audio`；随后使用 Mellow 解码/通道 reshape 和短音频重复、长音频随机裁剪；随机种子由固定 seed 0 + 样本 ID 决定，断点继续不会换片段 |
| MMAR audio1 | 原有 mono / resample / 首 10 秒 / 短音频右补零 |
| 音频 prefix | 129 + 分隔符 token 0 + 129 + 分隔符 token 0 = 260 个有效位置 |
| 文本 | 保留 question 和选项正文大小写及问题标点；固定原始选项顺序；按选项数使用小写 `a)` 至 `j)`；不加 `Choices:` |
| prompt 分词 | 直接调用对应 formal30 dataset 的 `_tokenize_text(..., max_len=129, padding=False)`；附加 EOS；截断后最后有效 token 强制为 EOS；不在 prompt EOS 与答案间补齐 |
| 注意力 | prefix + 实际 prompt 全有效；首个答案 token 由 prompt EOS 的最后位置预测 |
| 生成 | 两个 benchmark 均最多 300 tokens；temperature 1.0、top-p 0.8 过滤后 argmax；不随机采样；每步全量重算，显式 use_cache=False；遇到新生成的 EOS 停止 |
| 分母 | 每个任务官方 1000 条；单条 skip/空输出保留且计错；模型生成异常使报告不可用于正式比较 |

这套输入协议遵循本项目 formal30 的槽位和 prompt EOS 合同；与历史全小写、prompt 补齐、第二槽复用音频的评测输出有协议差异。三个新结果应使用本目录的统一协议一起比较。

## 模型加载与隔离

从 checkpoint 内嵌的 `route_contract` / `text_model_contract`、层和 router state keys 自动识别，不能通过文件名猜架构。严格验证 schema、30/30 epochs、GBS256、global token loss、4e-5 → 4e-6、完整 scheduler、compact attention、静音第二槽和 HTSAT 冻结合同。SmolLM2 为普通 30 物理层；x2 为 20 物理 / 30 逻辑层、5 memory slots、6 routers；x4 为 20 物理 / 50 逻辑层、7 memory slots、10 routers。

每个作业只加载一种架构，并按原训练的 overlay 顺序导入对应 route，完整严格加载音频 checkpoint 的 `state_dict`。不恢复 optimizer、scheduler、训练 RNG、数据游标，也不创建训练 staging。原生构造器仍需要 runtime YAML 中的 HTSAT、canonical SmolLM2 tokenizer/权重和相应递归文本初始化目录可读，然后以音频 checkpoint 覆盖整个模型权重。不要清理这些依赖。

训练 runtime YAML 中已经失效的 `/dev/shm` 数据路径、节点/PID、时间戳、输出目录和训练 provenance 不参加评测启动 gate，也不需要重建。数据/scorer、实际模型、tokenizer、软件/精度和评测 Python 语义才参加评测恢复身份。shell wrapper、注释字节哈希不作为恢复门槛。

## 两套评分与输出

每条样本只生成一次：

- `choice_label_prefix_evaluation.{json,txt}`：使用原始 `generated_text.split(')')[0].lower()` 与带正确标签的答案比较；不额外 trim，不猜测句子里的选项。标签大小写不敏感，` b)`、`answer is b)` 不会被宽松纠正。answer 无法映射到选项时保留分母、计错并记录 `record_errors`。
- `official_evaluation.txt`：运行数据目录中原本的官方 `evaluation.py --input ...`。仅移除生成开头的 `a)` 到 `j)` 标签，剩余正文保持原样。MMAR 的 `answer_prediction` / `model_prediction` 字段由官方 scorer 自动探测。官方 scorer 的原始日志完整保留。

共同文件包括 `summary.json`、`evaluation_report.json`、`run_config.json`、`progress.jsonl`、`raw_generations.jsonl`、`skipped.jsonl`、`predictions_choice_label_prefix.json`。MMAU 官方输入为 `predictions_fixed_order.json`，MMAR 为 `predictions_official.json`。原始 token IDs、prompt IDs、音频片段 hash、静音断言、停止原因都保留在 JSONL 中。不会用官方去标签后的文本计算字母分数。

默认输出：

```text
<checkpoint run root>/eval_formal30_unified_v3/<mmau_test_mini|mmar>/<checkpoint job dir>/model--epo-{15,25,30}/full/
```

同级 `.full.formal30_eval_identity.json` 防止不同 checkpoint、数据或协议混写；Linux advisory lock 防止同一个输出并发写入。中断后重复原提交命令，会恢复已经完成的样本。若产生模型生成异常，该行按既有遍历规则保留为 terminal skip，修复后应指定新的输出目录重跑。完整分母、artifact 审计、官方 scorer 和生成无异常均通过后，报告才写 `comparable: true`。

## 三模型结果核对

两个 benchmark 分别汇总一次。在已有 conda 环境中：

```bash
python mellow_formal30_eval/compare.py \
  /path/to/smollm2/evaluation_report.json \
  /path/to/mesh_x2/evaluation_report.json \
  /path/to/mesh_x4/evaluation_report.json \
  --output-csv /path/to/mmau_or_mmar_comparison.csv
```

汇总前要求三份报告均 comparable PASS，且协议、数据/scorer、tokenizer、软件环境、非架构 LM 配置及 attention implementation 一致；逐样本核对实际 prompt token IDs、audio1 waveform SHA256 和生成覆盖集合。模型权重和递归结构分别记录，不要求其相同。

## 本地验证边界

依赖轻量的回归测试：`python -m unittest discover -s tests -p test_mellow_formal30_eval.py -v`。另执行 Python compile、三个骨干路线 source path 检查、Bash `-n` 和 mock `vc submit` 参数/路径转义检查。本地没有 PyTorch、数据或权重；这些检查不能证明远端 FP32/CUDA 加载及两个官方 scorer 的实际执行成功。实际结果以远端正式作业输出为准。
