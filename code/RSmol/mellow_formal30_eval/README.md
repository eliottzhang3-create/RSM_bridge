# Mellow checkpoint 的统一 MMAU / MMAR 评测

支持三条 formal30 路线的 schema-v2 checkpoint，以及 `run_mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090.sh` 产出的任意 1–10 epoch checkpoint。10-epoch 路线只接受标准 SmolLM2 权重；任意正 max LR 均可，但 checkpoint 与同 run root 的 `runtime_10epochs.yaml` 必须一致且 `min_lr=0.1*max_lr`。提交 wrapper 为 `submit_mellow_formal30_eval_5090.sh`。

## 正式提交

代码通过 GitHub 同步到远端后，在远端 `code/RSmol/` 目录执行。以下每条命令提交两个独立作业：MMAU test-mini、MMAR；合计六个作业。没有 smoke gate，直接遍历完整数据并运行两套评分。

```bash
bash submit_mellow_formal30_eval_5090.sh both smollm2
bash submit_mellow_formal30_eval_5090.sh both mesh_x2
bash submit_mellow_formal30_eval_5090.sh both mesh_x4
```

formal30 的别名在 [checkpoints.tsv](checkpoints.tsv) 中对应指定 checkpoint 绝对路径。自定义绝对路径可评测任意受支持路线的已保存 epoch；wrapper 不等待尚未完成的训练，也不会猜测 checkpoint。

固定提交资源：`pdgpu-5090 -c 8 -m 32G -g 1 -n 1`，沿用训练容器与 `mellow_c8204d8` conda 环境。`MELLOW_CONDA_BASE`、`MELLOW_CONDA_ENV` 可指定已有环境；不自动安装依赖。所有模型比较时使用同一软件环境。自定义 checkpoint 的示例：

```bash
bash submit_mellow_formal30_eval_5090.sh mmau_test_mini \
  /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090/<run-root>/checkpoints/<job-dir>/model--epo-10.ckpt \
  --runtime-config /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090/<run-root>/runtime_10epochs.yaml
```

将 `model--epo-10.ckpt` 替换为该 job 下任意已保存的 `model--epo-<n>.ckpt`。未指定 `--runtime-config` 时，wrapper 只在 run root 恰好发现一个 `runtime_10epochs.yaml` 或 `runtime_30epochs.yaml` 的情况下自动选择；否则必须显式传入。wrapper 会打印 checkpoint 参数和解析后的路径。`both --output-dir PATH` 时 PATH 下分别创建 `mmau_test_mini/`、`mmar/`；`--dataset-dir` 仅适用于单任务。

## 共同评测合同

| 项目 | 所有受支持 checkpoint 的共同评测设置 |
|---|---|
| 精度、设备 | 全部浮点参数/buffer、音频、prefix、logits 均 FP32；CUDA 0；关闭 autocast 和 TF32 |
| 模式 | 全模块 eval / inference mode；每次一条样本 |
| 音频 | 32 kHz、10 秒；两个原生槽分别编码；第二槽复用第一槽的同一 waveform tensor |
| MMAU audio1 | 优先使用官方 `<id>.wav`；若镜像没有该 convenience WAV，则沿用旧 evaluator 回退到 parquet 内嵌 `context.audio`；随后使用 Mellow 解码/通道 reshape 和短音频重复、长音频随机裁剪；随机种子由固定 seed 0 + 样本 ID 决定，断点继续不会换片段 |
| MMAR audio1 | 原有 mono / resample / 首 10 秒 / 短音频右补零 |
| 音频 prefix | 129 + 分隔符 token 0 + 129 + 分隔符 token 0 = 260 个有效位置 |
| 文本 | 保留 question 和选项正文大小写及问题标点；固定原始选项顺序；按选项数使用小写 `a)` 至 `j)`；不加 `Choices:` |
| prompt 分词 | 直接调用对应 formal30 dataset 的 `_tokenize_text(..., max_len=129, padding=False)`；附加 EOS；截断后最后有效 token 强制为 EOS；不在 prompt EOS 与答案间补齐 |
| 注意力 | prefix + 实际 prompt 全有效；首个答案 token 由 prompt EOS 的最后位置预测 |
| 生成 | 两个 benchmark 均最多 300 tokens；temperature 1.0、top-p 0.8 过滤后 argmax；不随机采样；每步全量重算，显式 use_cache=False；遇到新生成的 EOS 停止 |
| 分母 | 每个任务官方 1000 条；单条 skip/空输出保留且计错；模型生成异常使报告不可用于正式比较 |

当前使用 v5 评分协议：两个评分器接收同一份完整 `generated_text`，保留选项字母、`)`、大小写及前后空白。旧版静音槽或官方输入去标签协议的报告不可与新报告直接比较；默认独立输出目录避免续写旧结果。

## 模型加载与隔离

从 checkpoint 内嵌的 `route_variant`、`route_contract` / `text_model_contract`、层和 router state keys 识别训练路线与架构，不通过文件名猜测。formal30 仍严格要求 30 epochs 和固定 `4e-5 → 4e-6`；10-epoch LR 消融要求 route 标识正确、epoch 在 1–10、GBS256、global token loss、AdamW/warmup/cosine/梯度裁剪及 runtime LR 一致，并允许任意正 max LR 且固定 `min_lr=0.1*max_lr`。两个路线都验证 compact attention、HTSAT 冻结和模型结构。10-epoch 只接受普通 30 物理层 SmolLM2。

每个作业只加载一种架构，并按原训练的 overlay 顺序导入对应 route，完整严格加载音频 checkpoint 的 `state_dict`。不恢复 optimizer、scheduler、训练 RNG、数据游标，也不创建训练 staging。原生构造器仍需要 runtime YAML 中的 HTSAT、canonical SmolLM2 tokenizer/权重和相应递归文本初始化目录可读，然后以音频 checkpoint 覆盖整个模型权重。不要清理这些依赖。

训练 runtime YAML 中已经失效的 `/dev/shm` 数据路径、节点/PID、时间戳、输出目录和训练 provenance 不参加评测启动 gate，也不需要重建。数据/scorer、实际模型、tokenizer、软件/精度和评测 Python 语义才参加评测恢复身份。shell wrapper、注释字节哈希不作为恢复门槛。

## 两套评分与输出

每条样本只生成一次：

- `choice_label_prefix_evaluation.{json,txt}`：使用原始 `generated_text.split(')')[0].lower()` 与带正确标签的答案比较；不额外 trim，不猜测句子里的选项。标签大小写不敏感，` b)`、`answer is b)` 不会被宽松纠正。answer 无法映射到选项时保留分母、计错并记录 `record_errors`。
- `official_evaluation.txt`：将完整模型输出直接传入数据目录中原本的官方 `evaluation.py --input ...`；不移除开头的 `a)` 到 `j)`，不截取正文、不 trim，也不额外改变大小写。官方 scorer 自身的评分逻辑保持原样。MMAR 的 `answer_prediction` / `model_prediction` 字段由官方 scorer 自动探测。官方 scorer 的原始日志完整保留。

共同文件包括 `summary.json`、`evaluation_report.json`、`run_config.json`、`progress.jsonl`、`raw_generations.jsonl`、`skipped.jsonl`、`predictions_choice_label_prefix.json`。MMAU 官方输入为 `predictions_fixed_order.json`，MMAR 为 `predictions_official.json`；官方与前缀评分输入中的预测文本完全一致，并逐行与原始 `generated_text` 核对。原始 token IDs、prompt IDs、audio1 waveform hash、第二槽复用断言和停止原因都保留在 JSONL 中。

默认输出：

```text
<checkpoint run root>/eval_mellow_unified_v5/<mmau_test_mini|mmar>/<checkpoint job dir>/model--epo-<n>/full/
```

同级 `.full.mellow_eval_identity.json` 防止不同 checkpoint、数据或协议混写；Linux advisory lock 防止同一个输出并发写入。中断后重复原提交命令，会恢复已经完成的样本。若产生模型生成异常，该行按既有遍历规则保留为 terminal skip，修复后应指定新的输出目录重跑。完整分母、artifact 审计、官方 scorer 和生成无异常均通过后，报告才写 `comparable: true`。

## 三模型结果核对

两个 benchmark 分别汇总一次。在已有 conda 环境中：

```bash
python mellow_formal30_eval/compare.py \
  /path/to/smollm2/evaluation_report.json \
  /path/to/mesh_x2/evaluation_report.json \
  /path/to/mesh_x4/evaluation_report.json \
  --output-csv /path/to/mmau_or_mmar_comparison.csv
```

汇总器仍只用于三条 formal30 架构线。汇总前要求三份报告均 comparable PASS，且协议、训练家族、数据/scorer、tokenizer、软件环境、非架构 LM 配置及 attention implementation 一致；逐样本核对实际 prompt token IDs、audio1 waveform SHA256 和生成覆盖集合。训练家族不含 LR 数值，因而不同 LR 的 10-epoch checkpoint 可以共享协议身份，但不会与 formal30 混为同组。模型权重和递归结构分别记录，不要求其相同。

## 本地验证边界

依赖轻量的回归测试：从 `code/RSmol/` 执行 `python -m unittest discover -s mellow_formal30_eval/tests -p test_mellow_formal30_eval.py -v`。本地没有远端数据或权重；这些检查不能证明远端 FP32/CUDA 加载及两个官方 scorer 的实际执行成功。实际结果以远端正式作业输出为准。
