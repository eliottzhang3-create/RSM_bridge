# 完整 Mizar 在 MMAU test-mini / MMAR 上的静音 activation tracing

本实现独立于 ReasonAQA tracing、cutoff、剪层搜索和 MMAU/MMAR ablation，不修改旧代码、题单或输出。
两个数据集各自完整1000题，全部筛选结束后，用**所有合格题**做原第1～30层文本状态恢复。
没有500/1000入选配额，不会因合格题数量不足某个固定值而退出。

## 输入、评分、筛选与分类

模型固定原始Mizar final完整encoder＋已训练projector＋30个独立LM block：
`models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt`。
FP32源参数，BF16前向，FP32音频前端/log_softmax，TF32关闭。
SoundFile解码→先resample32k再mono→**前20秒**→不足右补零；无需TorchCodec。
紧凑全小写原题面/选项末尾加空格EOT，再右截断/补齐256，PAD17可见。
126audio＋SEP0＋256text=383，首答案logits在0-based382；无KV、无自由生成。

分类比较各有效选项的单个小写字母logprob；MMAU为a/b/c/d，MMAR保留原槽位和实际选项数，可到f。
空选项对应null分数/概率，不进入argmax，不重新编号；平局取最先有效原槽位。
真实release tokenizer核对单普通token，MMAU重复正文/既有GT槽位保持。
MMAR原5题无法唯一映射GT，继续在完整1000题分母中计错，目标概率null，按unmapped_gold排除RR。
此前官方matcher信息保留在题目元数据中，主分类沿用字母口径。

完整1000题均计算clean和silence，两者每题各一次LM前向。静音是640000点零波形正常通过原encoder/projector，
每rank/device编码一次，复用其126个audio token；不直接把投影token归零。
独立seed1234打乱执行顺序，每卡负责125题，最后核对完整覆盖。入选同时满足：

1. clean字母分类正确；
2. silence字母分类错误；
3. 正确目标token的完整词表概率Pclean>Pcorrupted。

选中题按原数据行号排序，不按概率排序、不重抽题。保存原ID、类别、shuffle位置、全部字母分数/概率及排除原因。
筛选报告另统计clean_wrong、silence_correct、nonpositive_probability_gap、unmapped_gold。

| 数据集 | 分析组 | 完整1000题分母 |
|---|---|---:|
| MMAU | perception | 331 |
| MMAU | semantic | 230 |
| MMAU | reasoning | 439 |
| MMAR | Signal Layer | 43 |
| MMAR | Perception Layer | 404 |
| MMAR | Semantic Layer | 412 |
| MMAR | Cultural Layer | 141 |

MMAU沿用用户的27子类→三类映射，见`mmau_mapping.json`；MMAR沿用题目元数据的四层级标签。
分组名不限制干预的Transformer深度：所有组都测完整1～30层。
筛选后各组n按实际结果变化，图和CSV明确显示。某组n=0的指标为null，图例注明无合格样本；不伪造为0分。
整个数据集n=0时保存完整基线、空题单和解释性空图，最终status=NO_QUALIFYING，进程成功退出，不做patch。

## 激活恢复与统计

Clean前向捕获每个block残差后的原始文本状态，位于下一block/最终LM norm之前。
仅为合格题落盘精确`30×256×576` tensor，保留实际dtype，不池化；BF16满1000题约8.24GiB，另有两份prefix。
每题encoder/projector计算一次，跨全部patch条件复用。未入选题的临时状态释放。

每个目标层L都从原静音prefix重新启动完整30层前向；L层结束时仅替换text：

```python
hidden_corrupted[:,127:383,:] = clean_state_at_layer_L
```

全部256位置包含EOT/可见PAD及readout382，audio/SEP保持静音运行状态，干预不累积。
每卡按selected_index%8分配实际N题；空分片参与集体通信。若N<8，空分片rank用共享入选probe完成同样的正确性对照。
八卡完成同一层后汇总全部N题、打印日志、barrier，再切换下一层。

概率取全词表FP32 log_softmax，沿ReasonAQA tracing在CPU readout上计算，再FP64 exp；不在候选字母内归一化。

```text
RR_i,L = (Ppatched_i,L - Pcorrupted_i) / (Pclean_i - Pcorrupted_i)
```

主指标为逐题RR的平均值，不是总体平均概率差之比，也不直接平均类别均值。
另报median/分位数/2000次按题bootstrap点态95%CI、三/四组的RR/准确率/目标概率、小gap敏感性及负值/>1数量。
不加epsilon、不clip；保留所有正分母合格题。
完整1000题的clean/silence准确率、入选比例独立输出；所选子集clean100%和silence0%是筛选条件。

每卡执行capture/native、输入层文本替换=corrupted、第30层恢复=clean、缓存=重新capture的逐值对照。
full-vocabulary logits/raw状态保持bitwise；派生FP64 exp只允许8ULP，标签/logprob等严格相同。
第30层RR实际计算而不强写1；其恢复是readout正对照，不能解释成最后一层独自完成音频融合。

## 远程执行与提交

本地修改代码，经GitHub同步到Linux；权重/音频/正式输出仍只在远程。
新文件清单见`sync_files.txt`，已有依赖须匹配`reference_guard.json`；旧实验源码保持原版本。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

# 可选：CPU预先冻结各自完整1000题输入，不做模型前向。
bash mizar_benchmark_tracing/run.sh prepare --task mmau
bash mizar_benchmark_tracing/run.sh prepare --task mmar

# 两个独立作业，每个内部完成筛选→全部合格题30层patch→汇总作图。
bash run_mizar_mmau_tracing_8gpu_5090.sh
bash run_mizar_mmar_tracing_8gpu_5090.sh
```

每条提交命令只提交一个vc作业：`pdgpu-5090 / 8GPU / 64CPU / 256G / 1node`，原镜像，自动激活`mellow_c8204d8`。
默认Miniconda为`/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3`，可用`RSMOL_CONDA_BASE`覆盖。
省略CPU prepare也可以，作业内自动ensure输入。

准备时依次只读复用：

1. 旧`data/mizar_{mmau,mmar}1000_letters_eot_v1/manifest.json`；
2. 若字母manifest不存在，读取旧`data/mizar_{mmau,mmar}1000_likelihood_v1/manifest.json`并追加EOT/字母目标；
3. 若两者均不存在，调用已有完整数据集builder，结果只写新tracing输入目录，旧路径不发布文件。

新输入默认`data/mizar_{mmau,mmar}1000_tracing_inputs_v1/manifest.json`。
已存在但不匹配的原manifest明确报错，不静默跳过校验或换源。`--source-manifest`指定字母题单、`--body-manifest`指定正文题单。
原标注/音频位置可用`--mmau-metadata/--mmau-audio-root/--mmau-parquet`或`--mmar-metadata/--mmar-audio-root/--evaluation-script`覆盖。

预览、仅筛选和续跑示例：

```bash
# 无模型/数据/CUDA/vc访问，仅打印提交计划。
bash run_mizar_mmau_tracing_8gpu_5090.sh --check-args
bash run_mizar_mmar_tracing_8gpu_5090.sh --check-args

# 可选：只筛选并保存所有合格题/状态，之后同目录默认命令继续patch。
bash run_mizar_mmau_tracing_8gpu_5090.sh --stage select

# 自定义新输出，重试用相同目录和参数。
bash run_mizar_mmar_tracing_8gpu_5090.sh \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_mmar_tracing_repeat1
```

默认全流程与仅筛选是可选方案，不要同时对同一目录提交。原子缓存、每题fsync日志、partial-tail恢复和输出锁保持。
完整题目不重复筛选；缺失/部分记录按原shuffle分片继续。模型/源码/环境/题目/音频/缓存身份变化会拒绝混用。

## 日志与输出

提交JSON返回确切`log_path`、`input_manifest`、`selected_manifest`、`output_dir`和submission报告。
日志位于远程`code/RSmol/log/`：

```bash
ls -lt log/mizar-tracing-mmau.*.1.log
ls -lt log/mizar-tracing-mmar.*.1.log

# 将<tag>换为提交打印的确切tag：
tail -f log/mizar-tracing-mmau.<tag>.1.log
tail -f log/mizar-tracing-mmar.<tag>.1.log
```

`SCREENED`逐题打印rank/完成数/原题ID/分类/原因/clean和silence预测/目标概率。
`SCREENING_DONE`打印完整1000题总体和分组基线、入选数量。
`TRACING_CACHE_READY`/`TRACING_CONTROLS_DONE`显示缓存和对照；`PATCH_LAYER_DONE`每层打印总体及三/四组RR与acc。
失败时`ranks/rankXX/failure.json`和`analysis_report.json.worker_failures`包含真实error、traceback和phase。

默认两个输出目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_mmau_tracing_all_eligible_letters_eot_bf16_8gpu5090_v1
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_mmar_tracing_all_eligible_letters_eot_bf16_8gpu5090_v1
```

各自包含：

- `run_config.json`、`shuffle.json`：完整身份和打乱顺序。
- `screening/per_question.jsonl`：原序号排列的完整1000题clean/silence结果；`shuffled_journal.jsonl`保留执行顺序。
- `selection_report.json`、`screening_summary.csv`：完整1000题总体/分组基线与入选比例。
- `selected/manifest.json`、`selected_records.jsonl`：所有合格题最终manifest，含分类和原始序号。
- `cache/source_XXXX/{entry.json,tensors.pt}`：合格题两份prefix和完整clean状态。
- `runtime_checks.json`及`ranks/rankXX/`：实际模型/缓存/对照/逐题恢复日志。
- `layers/patch_XX/{per_question.jsonl,summary.json}`：全部30层逐题RR和统计。
- `layer_scores.csv`、`layer_scores.json`：总体与三/四分组的完整逐层指标。
- `recovery_rate`：总体mean/median RR和CI；`recovery_by_category`：分组RR及patched acc。
- `target_probability`：入选子集clean/silence/patched目标概率；分组概率在CSV/JSON中。
- `recovery_heatmap`：按分类分组、组内原序号排列的逐题RR；`heatmap_row_map.jsonl`对应原题ID。
- 四类图各PNG＋SVG；`analysis_report.json`的`PASS`才表示正式完成，`NO_QUALIFYING`表示完整筛选完成但无合格题。

纯CPU重画图：

```bash
bash mizar_benchmark_tracing/run.sh plot \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_mmau_tracing_all_eligible_letters_eot_bf16_8gpu5090_v1

bash mizar_benchmark_tracing/run.sh plot \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_mmar_tracing_all_eligible_letters_eot_bf16_8gpu5090_v1
```

本地验证入口`tests/validate_mizar_benchmark_tracing.py`，报告`audit_reports/mizar_benchmark_tracing_validation.json`。
真实release tokenizer/本地标注和CPU小Llama数值验证与合成分布式流程分别记录，不冒充远程生产权重或八卡CUDA通过。
