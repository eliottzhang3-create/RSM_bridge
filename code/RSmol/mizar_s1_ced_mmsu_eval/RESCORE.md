# MMSU：完整答案文本包含评分与已有五份结果的 CPU 离线重评

用户已确认评分口径：忽略模型生成的选项字母，只检查生成中是否包含归一化后的完整 `answer_gt` 文本，
再将该判定转换为未改动官方 scorer 所需的格式。已有结果离线重评和后续新 MMSU 评测共用同一实现。
评分版本为 `mmsu_answer_text_contains_v2`。音频、题面、模型加载、FP32、greedy、token 上限均保持。

## 直接重评这五份已有结果

本地修改 commit/push 后，在远程执行以下命令。该入口直接运行 CPU 评分，**不使用 vc，不申请 GPU，
不读取 checkpoint、音频或 Parquet**；使用已经存在的 `mellow_c8204d8` 环境和外部官方 scorer。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/rescore_mizar_mmsu_selected5.sh all
```

也可单独重评，和上面的 `all` 二选一：

```bash
bash code/RSmol/rescore_mizar_mmsu_selected5.sh official_mizar
bash code/RSmol/rescore_mizar_mmsu_selected5.sh base
bash code/RSmol/rescore_mizar_mmsu_selected5.sh recursive_5_10_5
bash code/RSmol/rescore_mizar_mmsu_selected5.sh serial84_lora_r8
bash code/RSmol/rescore_mizar_mmsu_selected5.sh recursive_mesh
```

固定源目录均位于 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/`：

| TARGET | 已有 MMSU 目录 |
|---|---|
| `official_mizar` | `mizar_official_stage3_q4_seed20260905_matched_ced_v1/mmsu` |
| `base` | `mizar_s1_ced_small_gbs256_5090/formal_20261026_193830617147305/eval/mmsu_step17313` |
| `recursive_5_10_5` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_5090/formal_20261028_160823559722647/eval/mmsu_step17313` |
| `serial84_lora_r8` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8_5090/formal_20261103_165410438279511/eval/mmsu_step17313` |
| `recursive_mesh` | `mizar_s1_ced_small_gbs256_recursive_mesh_5090/formal_20261027_081123150934256/eval/mmsu_step17313` |

每份新结果都写入其源目录下的 `rescore_mmsu_answer_text_v2/`，不会覆盖旧 `evaluation_report.json`、
`official_evaluation.txt`、原始预测或 raw generations。源目录的 `.mmsu.lock` 用于和生成进程互斥。
`all` 会尝试全部五份；某份失败仍处理其余目标，最终返回非零状态并标出失败目标。

仅检查五个绑定路径，不读数据、不评分：

```bash
bash code/RSmol/rescore_mizar_mmsu_selected5.sh all --check-args
```

任意其他 MMSU 已有目录也可用通用离线入口：

```bash
bash code/RSmol/mizar_s1_ced_mmsu_eval/rescore.sh \
  --source-output-dir '/path/to/another/run/eval/mmsu_step17313'
```

`--evaluation-script` 可指定官方 scorer 位置（默认原含空格路径），`--output-dir` 可选独立评分目录。
禁止将输出设为源目录本身或其祖先；五目标入口的 `all` 不接受同一个 `--output-dir`。

## 精确评分定义

1. 模型生成和 token IDs 始终原样保存。仅在评分副本的文本比较中，去首尾空白、末尾一个
   `<|endoftext|>` 以及开头的 `a. `、`D) `、`(b) `、`[C] ` 等选项格式标记；这些字母不参与判分。
2. 生成文本和完整 `answer_gt` 均 Unicode `casefold`，连续空白折叠为一个空格；保留标点，不作模糊或语义匹配。
3. 检查完整标准答案文本是否出现在生成中。以字母/数字/下划线开头或结尾的答案有对应词边界检查，
   所以 `no` 不会命中 `noise`，`3` 不会命中 `13`。
4. 命中时才从原始选项元数据找到文本精确等于 `answer_gt` 的字母，写入独立评分副本；
   多个选项文本相同则确定地使用第一个。选项仅承担官方输入格式转换，不与模型输出的字母比较。
5. 未命中、空回答、仅输出字母而没有答案文本时，评分副本 `response` 写为空字符串。
   官方 scorer 将其计错并保留分母。5000题始终完整参与评分，不靠排除错误格式缩小分母。

例如标准答案是 `Hong Kong`，模型输出 `d. Hong Kong <|endoftext|>` 或解释中包含 `Hong Kong` 均命中；
即使 `Hong Kong` 在原题对应 A，也给官方副本写 A。只生成 `A`、但标准答案文本是 `Hong Kong` 则不命中。
这是用户选择的**答案文本包含指标**，不是模型选项字母准确率；解释、否定句或多候选回答中提到标准答案也会命中。

官方脚本保持原 SHA256：`b6d5d35d2755244019a78e601495ea74e722681bf6f7e1fd0179bab7cabca3eb`。
仍执行其 CLI，再用同一官方函数核对未四舍五入结果；另核验官方分母=5000、官方总体正确率=文本命中数/5000。

## 源结果校验和输出

优先读取完整 `raw_generations.jsonl`，核验5000条连续行号、唯一ID、原始元数据及 record/generation SHA。
有 `predictions_official.jsonl` 时还要求其元数据和原始 response 与 raw 逐行一致。
只有 raw 文件不存在时，才支持只凭原始预测 JSONL 重评；raw 存在但损坏时不会降级绕过检查。
读者不会截断或修补源文件，完整性不足、重复ID或损坏均明确失败。

新评分目录包含：

| 文件 | 内容 |
|---|---|
| `run_config.json` | 源文件、原报告等摘要，评分协议、代码与官方脚本身份 |
| `predictions_normalized.jsonl` | 官方格式：文本命中对应字母，未命中为空字符串 |
| `response_parsing.jsonl` | 逐题原始回答、归一化文本/GT、匹配位置、命中判定及转换后的 response |
| `parsing_report.json` | 命中/未命中数量、格式/EOS处理统计、未命中示例、文件摘要 |
| `official_evaluation.txt` | 未改官方 scorer 的 stdout/stderr |
| `evaluation_report.json` | 最终官方分数、5000题分母、文本匹配统计、来源和错误 |

重复离线评分须满足源内容、原报告、代码、协议和 scorer 身份一致；否则选新的评分输出目录。
源目录已有生成进程运行时会因锁冲突停止，不与其同时读取或覆盖文件。

## 后续还没跑的 MMSU

更新代码后继续使用原来的五目标提交、四目标三数据集提交或通用 MMSU 提交命令，无需额外参数。
它们都会自动调用上述 v2 文本包含评分。新推理仍保留 `raw_generations.jsonl` 和原始
`predictions_official.jsonl`，另写 `predictions_normalized.jsonl` 和解析审计给官方 scorer。

评分及代码身份已改变，已有 v1 输出不要直接用新生成入口覆盖；完整旧结果用本页离线入口即可。
未完成的旧生成应先沿用旧代码完成，再离线重评；保持原有断点身份规则。

## 本地验证与 Git 同步

```powershell
python -B tests/validate_mizar_s1_ced_reasonaqa_eval.py --report audit_reports/mizar_mmsu_answer_text_v2_validation.json
```

覆盖真实官方 scorer、5000条合成结果、原始内容保留、源损坏/重复/部分文件拒绝、五个固定路径、
真实CPU shell、无模型/音频/Parquet导入、后续完整生成流程，以及原MMAU/MMAR/MMSU/ReasonAQA回归。
缺真实Torch等依赖的检查明确跳过，不宣称远程生产推理已验证。

2026-11-13 本地验收：180项中170通过，10项因缺少Torch/音频/Parquet依赖跳过，报告PASS。
新增16项重评回归全部通过；真实官方scorer在5000条合成结果上验证分母和文本命中总数一致。
历史132份快照保持，其中4份授权MMSU改动单列前后摘要，其余128份字节一致。
验收报告：`audit_reports/mizar_mmsu_answer_text_v2_validation.json`。未访问远程生产资产、提交GPU作业或代为commit/push。

本地自行提交推送（不会由工具代为执行）：

```powershell
git add -- RM.md code/RSmol/mizar_s1_ced_mmsu_eval/parsing.py code/RSmol/mizar_s1_ced_mmsu_eval/rescore.py code/RSmol/mizar_s1_ced_mmsu_eval/rescore.sh code/RSmol/mizar_s1_ced_mmsu_eval/evaluate.py code/RSmol/mizar_s1_ced_mmsu_eval/runtime.py code/RSmol/mizar_s1_ced_mmsu_eval/README.md code/RSmol/mizar_s1_ced_mmsu_eval/SELECTED5.md code/RSmol/mizar_s1_ced_mmsu_eval/RESCORE.md code/RSmol/mizar_s1_ced_mmsu_eval/normalization_reference.json code/RSmol/rescore_mizar_mmsu_selected5.sh tests/test_mizar_mmsu_rescore.py tests/mmsu_v2_audit.py tests/test_mizar_s1_ced_reasonaqa_eval.py tests/validate_mizar_s1_ced_reasonaqa_eval.py audit_reports/mizar_mmsu_answer_text_v2_validation.json
git commit -m "Add MMSU answer-text scoring and offline rescoring for five targets"
git push
```
