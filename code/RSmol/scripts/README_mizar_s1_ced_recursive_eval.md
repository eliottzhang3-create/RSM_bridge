# 四条非 MeSH CED 5+10×2+5 路线的 MMAU / MMAR 评测

两个既有入口 `run_mizar_s1_ced_small_mmau_eval_5090.sh` 和
`run_mizar_s1_ced_small_mmar_eval_5090.sh` 通过 `RSMOL_MIZAR_CED_EVAL_VARIANT`
选择四条新增路线。默认 `base` 和已有 `recursive_mesh` 继续调用各自原实现。
新增实现位于训练目录之外：

```text
scripts/evaluate_mizar_s1_ced_recursive_benchmark.py
scripts/mizar_s1_ced_recursive_eval_runtime.py
scripts/evaluate_mizar_s1_ced_recursive_benchmark.sh
```

四条训练路线、原普通 CED / MeSH Python evaluator 和共享递归实现均未为此次评测修改。
本次不包含 layer36912 cross attention 的评测适配。

## 默认 checkpoint 与模型

所有路径均位于远程输出根目录
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/`。
每项读取对应 run 的 `formal/checkpoints/model--step-17313.ckpt`
和同一 run 的 `formal/runtime.json`。

| variant | 相对输出根目录的 run | 实际行为 |
|---|---|---|
| `recursive_5_10_5` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_5090/formal_20261028_160823559722647` | 普通非 MeSH 递归，无额外注入 |
| `recursive_5_10_5_layer8_4_inject` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_5090/formal_20261030_000807991327585` | 逻辑索引5注入A8、15注入A4 |
| `recursive_5_10_5_layer4_8_inject` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer4_8_inject_5090/formal_20261029_210116592758429` | 逻辑索引5注入A4、15注入A8 |
| `recursive_5_10_5_audio_reinject` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_audio_reinject_5090/formal_20261029_110929631468450` | 原mapper音频prefix经共享投影、共享gate，在索引5和15注入 |

训练目录对应
`mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small`
及其 `_layer8_4_inject`、`_layer4_8_inject`、`_audio_reinject` 后缀。
四者均为 hidden576、20个独立物理层、30次逻辑执行，轨迹精确为
`0..14,5..14,15..19`，没有MeSH routers或memory slots。

两条层注入路线每个样本只编码一次音频，保持训练接口返回的 `(A4,A8)` 原顺序，
每一步完整LM重算均显式传递这对特征；8→4的选择在训练模型内部完成。
audio_reinject沿用训练模型，每次LM forward从原始前126个输入embeddings计算一次A，
再在两处调用同一gate。所有样本特征都是本次调用的局部变量，不保存跨样本cache。

## 与原 CED 一致的协议

| 项目 | 固定设置 |
|---|---|
| 数据 | MMAU test-mini、MMAR各完整1000条；官方metadata/scorer校验、原遍历和选项顺序 |
| 音频 | 32 kHz mono；fixed20中心裁剪或右补零后拆为两个10秒，重新拼成20秒送入CED；MMAU保持ID WAV优先和parquet内嵌音频回退 |
| Prompt | 原固定选项模板、整体小写、训练tokenize/EOS后缀；截断/右补`!`至256 tokens |
| Prefix | 126 audio + 1 SEP + 256 prompt = 383；第一个answer logit位于382 |
| 精度 | 单GPU、batch1、FP32、全部eval、inference mode；seed1234，TF32关闭，原cuDNN设置 |
| 解码 | `attention_mask=None`、`labels=None`、`use_cache=False`；逐token完整重算、纯argmax |
| 停止 | 两个benchmark均最多300个新token，第一次生成EOS后停止 |
| 文本 | `skip_special_tokens=False`、`clean_up_tokenization_spaces=False`；原始EOS、大小写、空白完整保留，不strip、不截标签、不提取字母 |
| 评分 | 原官方scorer和预测字段；完整1000条生成、无skip后才成功评分 |
| 资源 | `pdgpu-5090 / 8 CPU / 32G / 1 GPU / 1 node`；原容器与`mellow_c8204d8` |

协议字典与普通CED完全相同。runner的18个函数中，15个与原普通CED保持AST一致；
变化仅在参数解析、逐样本特征传递、输出身份。评测不混入其他路线的32-token上限、
top-p、去标签评分或10秒单段输入。

## 严格加载与执行审计

先检查checkpoint schema、formal模式、step17313、3-epoch horizon、训练geometry
`8,8,4`、loss标识、route和runtime合同/SHA，再核对训练源码、递归实现和资产身份。
用对应训练路线构建模型，先转FP32，再strict加载完整音频checkpoint的state_dict，
覆盖CED、mapper、LM以及投影/MLP/gate。仅规范化一致的DDP `module.`前缀；
missing/unexpected keys或混合前缀均拒绝。评测不恢复optimizer/scheduler/训练RNG。

tokenizer从原文本checkpoint读取，沿用训练校验注册已有`!`为PAD，EOS/SEP=0，
要求无词表增长、不resize embeddings。加载后的模型结构调用真实训练validator核验。
4→8、8→4的参数名称和形状相同，仍须通过route、源码、合同及实际轨迹区分。

只读hooks核对每次LM forward实际30次物理层调用及无KV cache，另外核对：

- 8→4：`[[5,8],[15,4]]`。
- 4→8：`[[5,4],[15,8]]`。
- audio_reinject：`[5,15]`。

hooks返回None，不修改隐藏状态或logits。保留训练模型默认关闭的数值统计模式。
源码SHA不符时应找回匹配训练时的代码版本，不能关闭校验来加载。
预训练CED和文本资产仍须存在；完整音频checkpoint不替代构建阶段对这些资产的依赖。

## 八项正式提交

通过GitHub将代码同步到远程后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol

# 普通非MeSH递归CED
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5 bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5 bash run_mizar_s1_ced_small_mmar_eval_5090.sh

# CED 8→4
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer8_4_inject bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer8_4_inject bash run_mizar_s1_ced_small_mmar_eval_5090.sh

# CED 4→8
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer4_8_inject bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer4_8_inject bash run_mizar_s1_ced_small_mmar_eval_5090.sh

# mapper音频重注入
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_audio_reinject bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_audio_reinject bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

每条命令分别提交一个单卡full评测作业，共八项，无smoke前置任务。
提交wrapper打印variant、checkpoint、runtime与output，未知variant在vc提交前拒绝。
compute wrapper激活`mellow_c8204d8`并设置HF离线模式，不安装或升级依赖。

## 结果、覆盖路径与续跑

各模型结果分别位于各自run root下：

```text
eval/mmau_test_mini_step17313/
eval/mmar_step17313/
```

每项保存`run_config.json`、`progress.jsonl`、`raw_generations.jsonl`、
`predictions_fixed_order.json`（MMAU）或`predictions_official.json`（MMAR）、
`evaluation_report.json`与`official_evaluation.txt`。
日志写入远程仓库`code/RSmol/log/`下各variant独立的JOB.log。
成功需同时满足report的`status=PASS`、1000条generated/零skip，且官方scorer确认1000条分母。
实际结构与执行轨迹见`model_audit.recursive_audit`、`model_audit.recursive_forward_audit`。
完全复用既有1000条结果续跑时，不产生新的forward，后一个字段可为PENDING/0。

保留原环境变量覆盖接口：
`RSMOL_MIZAR_CED_RUN_ROOT`、`RSMOL_MIZAR_CED_CHECKPOINT`、
`RSMOL_MIZAR_CED_RUNTIME_CONFIG`、`RSMOL_MIZAR_CED_ROUTE_ROOT`、
`RSMOL_MIZAR_CED_EXPECTED_STEP`、`RSMOL_MIZAR_CED_EXPECTED_GEOMETRY`、
`RSMOL_MMAU_DATASET_DIR`、`RSMOL_MMAR_DATASET_DIR`、
`RSMOL_MIZAR_CED_MMAU_OUTPUT_DIR`和`RSMOL_MIZAR_CED_MMAR_OUTPUT_DIR`。
额外CLI参数仍直接传入runner。仅覆盖RUN_ROOT时checkpoint/runtime/output同步从该root推导；
手动覆盖单个checkpoint时应同时提供其配对runtime、正确variant及独立output。

同一checkpoint、数据、协议、评测代码和输出目录下，重新提交同一命令可继续已有记录，
已落盘样本不重新推理。run_config含variant、训练route和评测源码身份，
拒绝不同模型或代码继续写入同一目录，也拒绝非空且无有效身份的目录。
修正代码导致identity变化时使用新的输出目录。

## 本地验证与远程边界

汇总入口（仓库根目录）：

```bash
python -B tests/validate_mizar_s1_ced_recursive_eval.py
```

本地65项实际检查通过，9项真实CPU tensor检查因缺Torch/Transformers跳过。
65项包含新增28项合同/加载/mock解码/完整1000条流程/Bash模拟，以及原CED/MeSH37项回归；
另通过Python与三份Bash语法检查。报告位于
`audit_reports/mizar_s1_ced_recursive_eval_validation.json`。

新增7项真实CPU tensor检查使用真实20层/30次执行的模型实现、hidden32和合成输入，
仅在测试中缩小宽度和解码长度：非零gate及不同A4/A8的实际选择、逐步logits/token与直接模型调用一致、
hook不改变logits/参数、重注入每次投影一次/gate两次、跨样本隔离、strict权重roundtrip，
以及训练decoder与prefix评测首个答案logit对齐。另包含原MeSH的2项tensor回归。
这些测试无需远程大权重或数据，不下载、不运行CUDA、不提交作业。

可在远程手动执行要求tensor实际运行的验证，缺依赖或出现跳过会失败：

```bash
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate mellow_c8204d8
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
python -B tests/validate_mizar_s1_ced_recursive_eval.py --require-tensors
```

这不是正式评测的自动前置门禁。本地不具备四份真实checkpoint、数据和CUDA，
远程资产一致性、完整GPU加载、1000条生成与官方评分尚未确认，未登记远程PASS。
