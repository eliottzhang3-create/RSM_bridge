# MMSU 离线细分类报告

原官方 `mmsu_evaluation.py` 的函数名虽然包含 task，但实际只读取 `category` 和
`sub-category` 分组，所以原报告只列出 Perception/Reasoning × Linguistics/Paralinguistics
四组。官方数据更细的字段已经保存在我们所有架构的原始结果中，没有在生成时丢弃。

官方资料：
- [MMSU README：三层分类、47项任务与榜单](https://github.com/dingdongwang/MMSU)
- [官方评分脚本](https://github.com/dingdongwang/MMSU/blob/main/mmsu_evaluation.py)
- [官方数据字段](https://huggingface.co/datasets/ddwang2000/MMSU)

本入口只做 CPU 离线统计，不重新推理，不读权重/音频/Parquet，不需要 GPU。
复用当前项目 `mmsu_answer_text_contains_v2` 判分和固定 SHA 的官方 scorer；
原始回答、正式总分、推理协议和原评测代码均不修改。**采用官方分类字段，不表示改为
官方原生字母回答协议；与公开榜单比较时仍需注明本项目现行答案文本判分方式。**

## 输出的七种分类视图

每组都有 `correct`、`incorrect`、`total`、`accuracy`（0到1）和 `accuracy_percent`（0到100）。

| view | 分组字段 | 用途 |
| --- | --- | --- |
| category | category | Perception / Reasoning |
| category_subcategory | category + sub-category | 保留原来的四组 |
| taxonomy | category + sub-category + sub-sub-category | 细分 Semantics、Phonology 及副语言分类 |
| discipline | 上述三层 + linguistics_sub_discipline | 如 Prosody 等，保留完整上级路径 |
| task | task_name | 按实际任务逐项列出，官方 README 描述47项 |
| task_taxonomy | 上述四层 + task_name | 同名任务在不同分类下分别列出 |
| leaderboard_domain | category + 衍生的 domain | Per./Rea.各 Semantics、Phonology、Paralinguistics，便于查看六个榜单维度 |

所有准确率均按该组完整样本数计算；总分为5000题的微平均，不把任务百分比再次平均。
每个 view 的样本数之和必须为5000、正确数之和必须等于总正确数，且原四组必须与官方
scorer 在原 v2 评分副本上的结果一致。空分类字符串单列，未知榜单维度放入 Unmapped
并警告；不丢题，不强行把观测到的任务数量凑成47。不读取 `id` 前缀猜测任务标签。

## 单个模型：直接处理已有结果目录

代码经 GitHub 同步到远程后，在仓库根执行。例如四循环 MeSH：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/mizar_mmsu_detailed_report/run.sh \
  --source-output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_recursive_mesh_5_10x4_5_5090/formal_20261103_093224843011222/eval/mmsu_step34626
```

如果要分析 rerun，改为实际 `mmsu_step34626_rerun_...` 目录。
必须指向保存原始 `raw_generations.jsonl` 或 `predictions_official.jsonl` 的目录，
不要指向 `rescore_mmsu_answer_text_v2` 或仅含规范化字母的评分目录。

默认在该目录下独立创建 `detailed_mmsu_answer_text_v2/`：

- `detailed_report.json`：完整细分类、计数、分类覆盖和来源/历史总分核对。
- `detailed_scores.csv`：可用 Excel 查看，按 `view` 筛选不同层次。
- `detailed_scores.md`：完整表格，便于直接阅读。
- `predictions_normalized.jsonl`、`response_parsing.jsonl`、`parsing_report.json`、
  `official_evaluation.txt`：本次核对使用的独立 v2 评分副本及审计。
- `run_config.json`：本次来源、策略、源码和 scorer 身份。

允许同身份重复运行；来源或策略改变时拒绝覆盖已有报告，可指定新 `--output-dir`。
不覆盖原目录及其祖先目录。默认子目录不进入生产评测源码身份，现有作业仍可按原代码续跑。
若已存在成功的 v2 总分，会核对它与本次逐题重算是否一致；不一致时失败，不静默改旧分数。
旧评分协议或未成功报告会注明未比较。本工具不验证权重是否正确，只验证所提供完整结果的统计。

## 多个模型与批量处理

可重复传入 `--source-output-dir`，每项写自己的子目录；一项失败后继续其余项，任何失败使最终退出码非零。

```bash
bash code/RSmol/mizar_mmsu_detailed_report/run.sh \
  --source-output-dir /absolute/path/model1/eval/mmsu_step17313 \
  --source-output-dir /absolute/path/model2/eval/mmsu_step34626
```

批量处理 outputs 下既有正式和 rerun 结果（也包含未完整项；未完整项会明确失败并保留原文件）：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
failed=0
while IFS= read -r -d '' prediction; do
  bash code/RSmol/mizar_mmsu_detailed_report/run.sh \
    --source-output-dir "$(dirname "$prediction")" || failed=1
done < <(find /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol \
  -type f -path '*/eval/mmsu_step*/predictions_official.jsonl' -print0)
echo "failed=$failed"
```

官方发布模型如保存在其他命名的目录，可单独传入其完整原始结果目录；实现与架构无关。
仅参数预览可加 `--check-args`，不读输入文件或写报告。

## 边界

只接受完整5000条、唯一ID、原分类字段齐全的结果；优先验证原 raw 的逐行哈希。
原 raw 存在但损坏时不会退回另一个文件。源目录忙锁时拒绝；不恢复、不截断源JSONL。
对方直接改写文件时，发布前再次检查来源哈希，发现变化则不发布成功表格。
本工具不会解决模型回答退化，也不重新解释 v2 已声明的文本包含判分规则。
