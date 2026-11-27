# 完整 Mizar 连续三层窗口消融：起点6～25，步长1，八卡3090

本实验独立于既有单层消融、cutoff、剪层搜索与tracing；旧代码、题单和输出保持。
本地Windows修改核心源码，经GitHub同步到Linux，权重/音频/正式缓存/结果在远程。

## 条件与消融定义

每个数据集先测完整30层baseline，再依次测试20组三层窗口：

```text
[6,7,8], [7,8,9], [8,9,10], ... , [24,25,26], [25,26,27]
```

所有层号均为**原始1-based层号**，窗口起点6～25、步长1，结束层=起点+2。
消融为选中3个完整decoder block的残差增量归零，即每个block的`h_out=h_in`。
实现直接路由到其余27个原始模块；保留此前残差流、所有audio/SEP/text token、原位置/RoPE和原模块权重。
不把hidden本身清零、不只改某个attention或MLP分支；不重新训练、重编号模块或累积删层。
每个条件都从完整30层的原模块集合重新选取27层，前1～5与末28～30层始终执行。

例如跳过[6,7,8]后轨迹为`1..5,9..30`；跳过[25,26,27]为`1..24,28..30`。
每个任务21组×1000题=21000次正式题目决策，两个任务合计42000次，另有少量同作业内的正确性对照。

## 模型、输入和评分

- 原始完整Mizar final：encoder、已训练projector、30个独立LM block，严格加载
  `models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt`。
- FP32源参数、BF16前向、音频前端/log_softmax FP32，TF32关闭，SDPA math/eager，全部eval。
- SoundFile解码→先resample32k再mono→**前20秒**→不足右补零，无TorchCodec。
- 原紧凑小写题面加空格EOT，再右截断/补齐256，PAD17可见；audio126+SEP0+text256=383，首答案位置382。
- 每题只进行一次prefix LM前向，原GPU FP32全词表log_softmax，比较纯小写候选字母token的loglikelihood；无生成、无KV cache。
- MMAU使用a/b/c/d，MMAR保持原实际选项数与槽位，可到e/f等；空选项null且不参与argmax，平局取最先有效原槽位。
- 原MMAU重复正文/GT字母槽位保持；MMAR原5题无法唯一映射GT仍在完整1000分母计错。
  原MMAR matcher只作为单独诊断，不替换主字母准确率。

每个数据集都使用**完整原1000题**，不重新采样、不使用tracing筛选子集、不输入静音。

| 数据集 | 分组 | 原完整题数 |
|---|---|---:|
| MMAU test-mini | perception | 331 |
| MMAU test-mini | semantic | 230 |
| MMAU test-mini | reasoning | 439 |
| MMAR | Signal Layer | 43 |
| MMAR | Perception Layer | 404 |
| MMAR | Semantic Layer | 412 |
| MMAR | Cultural Layer | 141 |

MMAU按此前用户27子类映射，见`mmau_mapping.json`；MMAR按原category。
总体和分组分数均按各自正确题数/题数计算，不平均各子类准确率。

## 执行与提交

同步本包和两个顶层入口，精确交付清单见`sync_files.txt`；已有依赖须匹配`reference_guard.json`。
无需修改旧单层消融或tracing源码。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

# 可选：CPU预先冻结两个完整1000题输入；不做模型前向。
bash mizar_triplet_ablation/run.sh manifest --task mmau
bash mizar_triplet_ablation/run.sh manifest --task mmar

# 两条各提交一个8卡3090作业；作业内自动ensure输入也可以。
bash run_mizar_mmau_triplet_ablation_8gpu_3090.sh
bash run_mizar_mmar_triplet_ablation_8gpu_3090.sh
```

资源固定`pdgpu-3090 / 8GPU / 64CPU / 256G / 1node`，原镜像，自动激活`mellow_c8204d8`。
默认conda根`/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3`，可用`RSMOL_CONDA_BASE`覆盖。
每卡固定`global_index%8`的125题；各题encoder/projector一次，新独立BF16 prefix缓存跨21个条件复用。
8卡一起完成baseline或同一窗口，完整合并1000题并打印，再barrier切下一条件；不按窗口把任务各派给单卡。

输入源优先只读已有字母EOT manifest：`data/mizar_{mmau,mmar}1000_letters_eot_v1/manifest.json`。
若其不存在，读取原正文manifest `data/mizar_{mmau,mmar}1000_likelihood_v1/manifest.json`，原题追加EOT/字母目标。
若两者均不存在，调用已有完整数据集builder，只把返回数据写新输入路径，不在旧目录发布题单。
存在但身份/文件不匹配的旧manifest明确拒绝，不静默跳过验证换源。
新输入目录默认`data/mizar_{mmau,mmar}1000_triplet_inputs_v1/manifest.json`；完整原题/ID/题序/audio/GT/分类保持。

预览与自定义输出示例：

```bash
# 仅打印资源和参数，不访问模型/数据/CUDA/vc。
bash run_mizar_mmau_triplet_ablation_8gpu_3090.sh --check-args
bash run_mizar_mmar_triplet_ablation_8gpu_3090.sh --check-args

# 自定义独立输出目录；失败续跑用相同目录和参数。
bash run_mizar_mmau_triplet_ablation_8gpu_3090.sh \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_mmau_triplet_repeat1
```

`--source-manifest`覆盖字母题单，`--body-manifest`覆盖正文题单，`--manifest`覆盖本实验新输入题单。
原标注/音频路径可用`--mmau-metadata/--mmau-parquet/--mmau-audio-root`或
`--mmar-metadata/--mmar-audio-root/--evaluation-script`覆盖。
旧manifest存在时，仍使用其原音频/元数据指纹，不靠这些参数重写旧题目。

## 输出、图和日志

默认新输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_mmau_triplet06_25_letters_eot_bf16_8gpu3090_v1
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_mmar_triplet06_25_letters_eot_bf16_8gpu3090_v1
```

主要文件：

- `analysis_report.json`：最终PASS/完整21条件21000决策、baseline、并列最佳窗口及运行审计。
- `conditions/baseline/`、`conditions/drop_06_08/`…`conditions/drop_25_27/`：每题字母分数/预测、summary及paired_changes。
- `condition_scores.json`、`baseline.json`、`group_baseline.json`。
- `accuracy_by_window.csv`：每个窗口原始起止层、27层执行、准确率和相对baseline变化；baseline执行30层。
- `accuracy_by_group.csv`：三类/四层级的正确率、baseline、百分点变化、newly_correct/newly_wrong/prediction_changed。
- `ranking.json`、`group_ranking.json`：按准确率降序、并列同rank；高排名表示删该窗口后保留性能较好，不表示窗口最重要。
- `accuracy_by_window.{png,svg}`：总体准确率及相对baseline的百分点变化。
- `accuracy_by_group.{png,svg}`：分组绝对准确率及相对各组baseline的变化。
- MMAR另有`accuracy_by_category`和`delta_by_category`四宫格，各PNG/SVG；`official_diagnostic.csv`独立诊断。
- `ranks/rankXX/`：逐题可续跑日志、模型审计、runtime_check、rank完成报告。
- `cache/rankXX/`：本实验独立BF16 prefix缓存；`runtime_checks.json`：真实路由/等价性和缓存库存。

每完成一组打印`BASELINE_DONE`或`TRIPLET_DONE`，含drop_layers、准确率、baseline差值、各组分数和耗时。
提交JSON返回确切`log_path`、`manifest`、`output_dir`及submission报告路径。

```bash
ls -lt log/mizar-triplet-mmau.*.1.log
ls -lt log/mizar-triplet-mmar.*.1.log

# 将<tag>换成提交返回的确切tag：
tail -f log/mizar-triplet-mmau.<tag>.1.log
tail -f log/mizar-triplet-mmar.<tag>.1.log
```

`SUBMITTED`只代表提交成功；最终看`analysis_report.json`的PASS。
错误保存`ranks/rankXX/failure.json`，协调器也收集到`analysis_report.json.worker_failures`。
同身份逐题续跑，不重复已完成决策；partial尾行恢复、原子缓存/fsync/输出锁保持，不要同时提交同目录。
源码/模型/环境/输入/缓存身份变化会拒绝混用，换卡型的旧缓存不直接套用本实验。

CPU重画图：

```bash
bash mizar_triplet_ablation/run.sh plot \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_mmau_triplet06_25_letters_eot_bf16_8gpu3090_v1

bash mizar_triplet_ablation/run.sh plot \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_mmar_triplet06_25_letters_eot_bf16_8gpu3090_v1
```

## 验证与解释边界

同作业内每卡首题检查[6,7,8]、[15,16,17]、[25,26,27]的真实27层路由，
与完整执行30层但三个block输出分别替换为其输入的独立实现逐值比较完整词表logits；恢复baseline后logits保持。
每次推理读前钩子验证输入/无KV，读后钩子核对精确原层轨迹，不靠config字段声称删层成功。
本地验证另用真实30层小Llama eager/SDPA/BF16测试全部20窗口、native baseline、参数不变/不累计；
真实release tokenizer和本地2000题元数据核对既有分类与选项异常；完整42000结果与8rank调度为明确合成夹具。

这是三层一起移除的联合影响；相邻窗口高度重叠，不能把窗口掉分直接归因于其中单层，
也不能默认三层掉分等于旧三个单层掉分之和。
测试入口`tests/validate_mizar_triplet_ablation.py`，报告`audit_reports/mizar_triplet_ablation_validation.json`。
实际远程Mizar/CUDA/3090显存与8卡通信仍需远程运行，不把本地CPU验证当生产评测PASS。
