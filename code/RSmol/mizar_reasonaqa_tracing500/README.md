# ReasonAQA500：复用已完成筛选的缓存继续 activation patching

用户回传原任务的`selection_report.json`：5000题全部扫描，clean正确3921题，其中静音仍正确3190题，
另1题概率差不为正，最终合格730题，未满足原1000题目标而退出。
这次按用户要求改为500题，保持原clean正确／静音错误／Pclean>Pcorrupted三项条件。

新增独立`mizar_reasonaqa_tracing500`包，**原`mizar_reasonaqa_tracing`及其所有源文件不修改**，
因此可以严格核对旧`run_config.json`身份并读取已有概率、prefix和30层干净隐藏状态。
不对原5000题重跑筛选、不重抽500、不重建原pool、不改旧manifest或缓存。

## 样本、推理和结果

1. 只读原`screening/journal.jsonl`，逐行核对seal、原题索引、shuffle顺序、acceptance_order及三项资格条件。
2. 取原打乱顺序中**最先合格的500题**，即acceptance_order0～499；再按原test.json行号升序输出新manifest。
   不是直接取原test.json前500题，也不按概率高低选题。
3. 保留原candidate token、题面/EOT/PAD、clean/corrupt完整分数和概率、音频SHA、来源/入选序号及缓存SHA。
4. 从旧目录只读加载对应`cache/source_XXXX/tensors.pt`；不复制约4GiB的500题干净状态、不重新编码各题音频。
   每卡额外编码一次真实静音波形作对照，原每卡首题clean/native/layer0/layer30/cache重现对照保持。
5. 完整Mizar encoder＋训练projector＋30层LM，FP32源参数/BF16前向、前20秒、全词表目标token概率保持。
   每次patched从静音prefix重新前向，逐一在原1～30层post-block替换全部text[127:383]，不替换audio/SEP，不累积。
6. 每卡62／63题，8卡共同完成同一层后合并500题、打印并barrier，再进行下一层；共15000次patched决策。
7. RR为逐题概率空间比值，不加epsilon、不clip；均值/median/CI/三来源组/准确率/概率/小gap敏感性和八份PNG/SVG保持。
   第30层恢复属于readout正对照，所选clean100%与静音0%属于筛选后的条件子集。

新旧包固定源码保护不被绕过。若修改过原1000题包的源码，应恢复为生成旧结果时的同一版本；
不能仅把原TARGET从1000改成500，否则旧pool/model/source身份及缓存会不一致。
此续跑只支持已完成且保存了原sealed selection_report的筛选；少于500合格题会明确失败。

### 读取旧概率的末位校验修复

原校验将保存的整份decision字典与当前CPU用`math.exp(logprob)`复算的字典直接判相等，
因此即使预测和logprob完全相同，仅FP64派生概率相差1ULP也会报
`saved prediction/probability disagrees with candidate logprobs`。
500题包的`validation.py`只对`candidate_probabilities`与`target_probability`允许最多8个FP64 ULP，
不使用固定绝对epsilon；0和1仍要求完全相同。类型、标签、argmax/tie/正确性、logprob、目标与其候选概率的内部对应保持严格。
超过容差时错误会显示具体题ID、字段、保存值/复算值、十六进制float和误差，便于区分真实不一致。

只校验，不重写或归一化旧概率、不改变筛选成员、clean/corrupted基准值、gap或缓存SHA。
同类校验同时用于GPU缓存重现与第30层；完整词表logits、raw hidden states仍逐值相等。
第30层按实际概率计算RR并保留其末位偏差，不强制写成1。
本次用户报错发生在新500题manifest写入前，更新整个500题包后可重执行下面原命令；无需重建原5000题pool或重跑筛选。

## 远程执行

本地通过GitHub同步新增包、顶层入口；原包和既有依赖保持原版本。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

# 可选：CPU预检原结果/缓存并导出500题manifest，不做模型前向。
# 验证旧模型资产和500份缓存SHA可能需要一些时间。
bash mizar_reasonaqa_tracing500/run.sh manifest

# 一个8卡5090作业，直接跑这500题的30层patch及汇总。
# 不先执行上面的manifest也可以，作业会自动导入和核对。
bash run_mizar_reasonaqa_tracing500_8gpu_5090.sh
```

沿用`pdgpu-5090 / 8GPU / 64CPU / 256G / 1node`、原镜像与`mellow_c8204d8`环境。
包内run.sh自动激活环境，`RSMOL_CONDA_BASE`可覆盖原Miniconda根目录。

默认来源目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_reasonaqa_tracing1000_letters_eot_bf16_8gpu5090_v1
```

默认新输出目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_reasonaqa_tracing500_letters_eot_bf16_8gpu5090_v1
```

如果旧任务用了自定义输出目录，CPU导入和提交两处使用同样的`--source-output-dir`，可同时指定新的`--output-dir`：

```bash
bash run_mizar_reasonaqa_tracing500_8gpu_5090.sh \
  --source-output-dir /absolute/path/to/old_screening \
  --output-dir /absolute/path/to/new500_output
```

输出锁和身份保持；失败后同参数同新目录重提可续跑已保存的逐题patch结果。不要同时对同一新目录提交两个任务。

## 日志与输出文件

提交打印确切`log_path`、`source_output_dir`、`selected_manifest`与`output_dir`。

```bash
ls -lt log/mizar-reasonaqa-tracing500.*.1.log
# 使用提交返回的确切tag：
tail -f log/mizar-reasonaqa-tracing500.<tag>.1.log
```

日志先显示`REUSED_FIRST500_READY`（source_qualifying730、selected500、new_screening_forwards0），
再显示模型、缓存、对照进度；每层输出`PATCH_LAYER_DONE`及RR/acc/分组分数。
新的协调器会将rank真实error、traceback及phase收集进`analysis_report.json.worker_failures`，减少只看到torchrun汇总错误的情况。

新目录中主要文件：

- `selected/manifest.json`：500题最终manifest；另有`selected_records.jsonl`与`sampled_test.json`。
- `run_config.json`、`selection_report.json`：原任务身份、原5000/730统计、first500规则和新协议。
- `ranks/rankXX/patch_XX.jsonl`：各卡逐题日志；原隐藏状态仍引用旧目录cache。
- `runtime_checks.json`：实际模型/八卡/静音/原始缓存重现对照。
- `layers/patch_XX/per_question.jsonl`、`summary.json`：30层全部500题的概率、RR和统计。
- `layer_scores.csv`、`layer_scores.json`。
- `recovery_rate`、`recovery_by_category`、`target_probability`、`recovery_heatmap`各PNG和SVG。
- `analysis_report.json`：最终`PASS`才表示完成，`SUBMITTED`只是提交成功。

CPU重画图：

```bash
bash mizar_reasonaqa_tracing500/run.sh plot \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_reasonaqa_tracing500_letters_eot_bf16_8gpu5090_v1
```

本地检查入口`tests/validate_mizar_reasonaqa_tracing500.py`，报告`audit_reports/mizar_reasonaqa_tracing500_validation.json`。
真实GPU、旧730份实际生产缓存及其CUDA重现由远程验证；本地只验证CPU真实张量/小Llama和明确的合成流程。
