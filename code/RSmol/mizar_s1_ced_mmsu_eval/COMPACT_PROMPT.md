# 2026-11-22 MMSU正式题面修正

用户授权本轮只修正实现，后续待评测模型/checkpoint由用户另行指定。
没有修改历史targets/excluded清单，没有提交真实作业，也没有把旧生成重评分标记为新模板结果。

正式输入示例：

```text
which country? a) india b) none c) china d) japan
```

`dataset.prompt`精确遵循既有MMAU `build_fixed_order_prompt(question, choices).lower()`。
保留四个选项的位置，null占位为`none`，不添加Question/Choices前缀、模板换行或only A/B/C/D指令。
GT、任务名和分类不进入输入；问题/选项仅外侧strip、整体lower，原始元数据保持。
这与用户回传三题中恢复选项回答的compact_lowercase逐字符相同，不保证所有题目答对。

通用MMSU的38个已有variant与text71427六条路线共享同一默认模板及协议标识。
模型加载/训练类、音频、FP32、PAD!/EOS0、prompt256/prefix383、首logit382、
全序列重算/无KV/argmax300保持；评分继续原答案文本包含v2及固定官方scorer，完整5000题分母保持。
原答案解析、scorer、rescore、数据读取/校验和greedy函数未改；MMSU runtime生成函数AST保持。

题面协议：`mmsu_compact_lowercase_ced_fixed20_raw_greedy_answer_text_v3`。
评分协议：`mmsu_answer_text_contains_v2`（独立于题面版本，仍忽略生成选项字母）。

新默认输出：

| 入口 | 默认输出 |
|---|---|
| 通用MMSU / 历史训练快捷入口 | `<run-root>/eval/mmsu_compact_lowercase_v1/step<step>/` |
| 历史官方release快捷入口 | `<release-output-root>/mmsu_compact_lowercase_v1/` |
| text71427 MMSU | `<run-root>/eval/text71427_ced_v2/mmsu_compact_lowercase_v1/<checkpoint-stem>/` |
| text71427显式output-root | `<output-root>/<model>/mmsu_compact_lowercase_v1/<checkpoint-stem>/` |

旧模板目录与新协议身份不符，显式指定旧目录时在模型加载前拒绝，不覆盖raw/报告。
同版新协议续跑保持。既有旧raw离线rescore路径保持，可复核历史分数，但不能据此推断新题面表现。
旧官方模板保留为`dataset.official_prompt`，仅给两套诊断工具做明确对照；生产默认不再使用。

本地联合验收134个独立用例最终125通过、9缺ML/音频/Parquet依赖明确跳过。
最初一项旧源码保护检查未包含此前selected4/init0904授权注册，已按其已有变更记录接入，并定向复验通过；初次失败及修复后记录均保留。
包含真实Bash参数链/模拟vc、通用5000题和六路线各5000题合成流程、题面逐行检查、续跑与旧身份拒绝、原官方scorer和旧raw离线兼容。
2886份本次范围外生产源码normalized-LF摘要保持，12份生产实现/入口发生明确授权变更（含诊断调用与说明注释）。
原训练/模型、MMAU/MMAR/ReasonAQA数值实现和评分器未改。历史源码保护快照保留，新增compact_prompt_reference.json明确记录前后摘要。
本地模型/音频/Parquet为明确fixture、bulk fsync模拟；不是新模板全量真实权重GPU验证。
报告：`audit_reports/mmsu_compact_lowercase_validation.json`。

针对性复验：

```powershell
python -B -m unittest discover -s tests -p test_mmsu_compact_lowercase.py -v
python -B -m unittest discover -s tests -p test_mizar_s1_ced_mmsu_eval.py -v
```

代码仍在本地，通过GitHub同步到远程后才可运行。当前base训练源码已改为重训配方，
后续若选择历史checkpoint，仍需其匹配源码；本轮没有放宽strict source检查或改动任何训练路线。
