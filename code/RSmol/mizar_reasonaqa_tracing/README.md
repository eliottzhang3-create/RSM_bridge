# ReasonAQA1000 silence activation patching — original full Mizar

本实验独立于剪层搜索、audio cutoff、CKA、MMAU/MMAR layer ablation；不修改旧代码、旧 manifest 或旧输出。
参考 [Causal Tracing of Audio-Text Fusion in Large Audio Language Models, §3](https://arxiv.org/html/2603.13768v2#S3)。
这是逐层恢复隐藏状态的 activation patching；所有实验都执行原始 30 个 Transformer block。

## 固定协议

- 模型：`models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt`，原 encoder、已训练 projector 和完整 30 层 LM。
- 使用原 `data/reasonaqa/test_close_ended_sample5000_seed1234_v1/manifest.json` 的全部且仅这 5000 题；不会重新抽取候选池。
- 原 5000 应为 AudioCaps-MCQ2108、Clotho-MCQ2369、ClothoAQA yes/no523；远程原 manifest/test.json 是权威来源。
- 独立 seed1234 打乱 5000 个索引，保存完整 permutation 及 SHA。按打乱顺序筛选，满足以下三个条件才收录：
  clean 分类答对；静音分类答错；正确目标 token 的 `Pclean > Pcorrupted`。
- 收满 1000 立即停止筛选前向。每轮窗口大小 `min(8, 还差的合格题数, 候选池剩余数)`；按 shuffle 顺序提交，避免最后一轮多做推理。
- 若 5000 耗尽仍不足 1000，输出 `selection_report.json: INSUFFICIENT_QUALIFYING` 和完整筛选日志，作业失败退出，**不启动正式 patch 阶段**。不补抽、不放宽条件。
- 合格 manifest 按原 test.json 行号升序，记录原5000序号、原行号、ID、shuffle位置、入选顺序、新 selected_index 及音频/源码/token/模型指纹。
- MCQ 比纯小写 `a/b/c/d`，yes/no 比纯 `yes/no`。每个候选必须是 tokenizer 的单个普通 token；真实 release tokenizer 启动前核对。
- 原紧凑全小写题面末尾追加空格及 `<|endoftext|>`，然后右截断/补齐256。MCQ保留原选项槽位，GT只用于评分；yes/no保留原问题，不额外添加指令。
- 输入为126 audio＋1 SEP(id0)＋256文本=383；PAD17仍可见；首答案读取0-based位置382。
- 源权重FP32、BF16前向、FP32音频前端与log_softmax；TF32关闭；SDPA math/eager；不生成、不用KV cache。
- SoundFile解码，先重采样32k再转mono，**前20秒**、不足右补零。不使用 TorchCodec。

## 三种运行与 RR

Clean run使用真实音频；Corrupted run将同样640000采样点的波形置零，正常通过encoder/projector。
不把projected audio token直接清零；静音经过网络后的表示一般不为零。每个rank在当前设备编码一次静音音频并跨题复用，八卡校验一致。

每个入选样本保存30个block的原始输出，位于各block残差之后、下一层/最终LM norm之前。
缓存为精确 `30×256×576` 文本状态，保留实际dtype，不池化、不强制转换、不复用CKA特征。
BF16状态1000题约8.24GiB，另含两份prefix和元数据；若运行实际输出FP32，则状态约16.48GiB，并如实记录。

对第1～30层分别做一次新的静音完整前向。在指定层的block输出处执行：

```python
patched_hidden[:, 127:383, :] = clean_hidden_at_layer[:, 127:383, :]
```

这里包括全部256个文本位置、EOT及可见PAD，尤其包括最终readout位置382。
不替换audio `[0:126]` 与SEP `[126:127]`；每个层条件从原静音prefix重新启动，干预不累积。

概率是正确目标token在**完整词表softmax**中的概率，不是候选集内重新归一化。分类预测则是候选token间argmax，平局取原先候选。
先FP32 `log_softmax`，再用FP64指数、差值和除法：

```text
RR_i,L = (Ppatched_i,L - Pcorrupted_i) / (Pclean_i - Pcorrupted_i)
```

不加epsilon、不截断RR；保留负数和大于1。主指标为每题RR的均值，不是总体概率均值之差的比值。
另报median、分位数、按题bootstrap2000次的点态95%区间、三来源类别、patched acc、目标概率、小gap数量及阈值敏感性。
小gap不被自动删除。所选子集clean100%/corrupt0%来自筛选条件，不能当作全5000题准确率。

每卡第一道入选题都检查：只读capture与native结果逐值一致；输入层文本替换等于corrupted；
第30层文本替换等于clean完整词表logits，RR=1；缓存与重新capture一致。
最后一层是实现的正对照，其高恢复率不代表最后一层独自完成音频理解。

## 执行

本地改代码，经GitHub同步到Linux。以下在远程仓库运行；无需在本地持有数据/模型。
`sync_files.txt`给出本次文件；`reference_guard.json`列出的已有依赖也必须已经同步。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

# 可选：CPU终端预先冻结候选5000题的题面/token/音频路径及打乱顺序。
# 此步不做模型前向，不能在CPU上选出baseline答对的1000题。
bash mizar_reasonaqa_tracing/run.sh prepare

# 默认：一个8x5090作业内完成筛选1000 + 全30层patch + 汇总作图。
# 若省略上面的prepare，作业会自动生成/核验pool manifest。
bash run_mizar_reasonaqa_tracing_8gpu_5090.sh
```

单次 `vc submit` 使用 `pdgpu-5090 / 8GPU / 64CPU / 256G / 1node`，原镜像和 `mellow_c8204d8`。
第二阶段每卡负责 `selected_index%8` 的125题；八卡共同完成同一patch层，打印汇总并barrier后切下一层。
入选题的encoder/projector不在patch阶段重复编码；约30000次额外完整LM前向，另有筛选与少量对照。

可选预览、分阶段与续跑：

```bash
# 只打印资源和参数，不调用vc、不读取模型/数据。
bash run_mizar_reasonaqa_tracing_8gpu_5090.sh --check-args

# 仅筛选1000题并保留完整clean状态；后续同目录默认命令继续patch。
bash run_mizar_reasonaqa_tracing_8gpu_5090.sh --stage select

# 更换独立输出目录；失败重跑用完全相同目录和参数续跑。
bash run_mizar_reasonaqa_tracing_8gpu_5090.sh \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_reasonaqa_tracing_repeat1
```

不要同时对相同目录提交多个作业；输出锁会拒绝。`--stage select`和默认命令是可选的分阶段方案，不要与默认全流程同时提交。
同身份断点恢复：按题fsync日志，截断最后一条未完成记录；已经完整的候选outcome/patch结果复用。
缓存/题目/代码/环境不同会拒绝续跑，完整日志篡改不会被静默忽略。

## 日志和输出

提交时JSON打印 `log_path`、`output_dir`、`selected_manifest` 和提交报告路径。
日志固定前缀 `code/RSmol/log/mizar-reasonaqa-tracing.<tag>.1.log`：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
ls -lt log/mizar-reasonaqa-tracing.*.1.log
# 用上面列出的确切文件名：
tail -f log/mizar-reasonaqa-tracing.<tag>.1.log
```

`SCREENED`逐题打印扫描数、入选数、原序号、标签、clean/silence预测和目标概率。
`PATCH_LAYER_DONE`每层打印mean/median RR、patched acc及三来源分组；不打印300个生成token。

默认输出目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_reasonaqa_tracing1000_letters_eot_bf16_8gpu5090_v1/
  run_config.json, shuffle.json
  screening/journal.jsonl, screening/source_XXXX.json
  selection_report.json
  selected/manifest.json, selected_records.jsonl, sampled_test.json
  cache/source_XXXX/{entry.json,tensors.pt}
  ranks/rankXX/{model_audit.json,runtime_check.json,cache_inventory.json,patch_XX.jsonl,rank_report.json}
  runtime_checks.json
  layers/patch_XX/{per_question.jsonl,summary.json}
  layer_scores.json, layer_scores.csv
  recovery_rate.{png,svg}, recovery_by_category.{png,svg}
  target_probability.{png,svg}, recovery_heatmap.{png,svg}
  analysis_report.json
```

完成以 `analysis_report.json` 的 `PASS` 为准，`SUBMITTED`不代表推理通过。筛选不足1000时查看`selection_report.json`；
各rank出错查看`ranks/rankXX/failure.json`，torchrun尾部错误可能只是连带退出。
纯CPU重新画图：

```bash
bash mizar_reasonaqa_tracing/run.sh plot \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_reasonaqa_tracing1000_letters_eot_bf16_8gpu5090_v1
```

本地验证入口：`tests/validate_mizar_reasonaqa_tracing.py`；报告在`audit_reports/mizar_reasonaqa_tracing_validation.json`。
真实生产Mizar/CUDA及八卡运行须在远程完成；本地数值验证使用真实随机小Llama和CPU BF16，不冒充生产分数。
