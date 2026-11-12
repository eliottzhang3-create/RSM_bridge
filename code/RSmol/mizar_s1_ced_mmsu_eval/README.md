# MMSU：官方题面、官方 scorer、项目 CED 推理协议

正式入口是 `code/RSmol/run_mizar_s1_ced_small_mmsu_eval_5090.sh`。本实现只新增隔离文件；原 MMAU test mini、MMAR 的入口、runner、runtime、模型和训练文件均保持原样。

本地 Windows 维护代码，通过 GitHub 同步到远程 Linux；权重、真实推理和输出仍在远程。通用入口不预选架构、训练 run、checkpoint 或 step，也没有默认触发 smoke、额外测试作业或训练。提交入口默认执行完整 5000 条评测。

2026-11-13 用户指定的官方 Mizar 与四个 step17313 checkpoint，已准备独立的五目标提交入口：`code/RSmol/run_mizar_mmsu_selected5_5090.sh`。固定映射、五条可直接执行的提交命令及本地 Git 同步命令见 [SELECTED5.md](SELECTED5.md)。它只组装参数并调用本通用入口，模型加载、生成和评分实现保持原样。

## 正式运行

在远程仓库根 `/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM` 执行下列模板。`<...>` 必须换成后续确认的真实目标，不能原样执行：

```bash
bash code/RSmol/run_mizar_s1_ced_small_mmsu_eval_5090.sh \
  --eval-variant '<已有 runtime 支持的 variant>' \
  --run-root '<训练 run 根目录，包含 formal/>' \
  --expected-step '<checkpoint 的 total_step>' \
  --expected-geometry '<world_size,microbatch,gradient_accumulation>'
```

`--run-root` 自动推导三个位置：

```text
<run-root>/formal/checkpoints/model--step-<step>.ckpt
<run-root>/formal/runtime.json
<run-root>/eval/mmsu_step<step>/
```

checkpoint 不在标准布局时，显式提供完整参数：

```bash
bash code/RSmol/run_mizar_s1_ced_small_mmsu_eval_5090.sh \
  --eval-variant '<variant>' \
  --checkpoint '<完整训练 checkpoint 路径>' \
  --runtime-config '<配套 runtime.json>' \
  --expected-step '<total_step>' \
  --expected-geometry '<world,micro,accum>' \
  --output-dir '<独立 MMSU 输出目录>'
```

`--route-root` 可显式指定对应训练实现；省略时只按 variant 选择已有实现目录，不套用任何历史 checkpoint、step 或 geometry。支持 base、原始 recursive_mesh、已有四条简单递归路线，以及现有 multilayer runtime 注册的交互、memory、LoRA 和 official-final S1 路线；完整 variant 列表见 `--help`。尚未接入这些 runtime 的新训练架构需后续单独适配，不能仅凭 checkpoint 名字自动加载。

独立官方最终发布权重也可复用已有严格 release loader，参数为：

```bash
bash code/RSmol/run_mizar_s1_ced_small_mmsu_eval_5090.sh \
  --eval-variant official_mizar \
  --model-dir '<官方发布目录，包含配套 assets/>' \
  --checkpoint '<原始官方最终权重文件>' \
  --output-dir '<独立 MMSU 输出目录>'
```

此分支会检查既有 `mizar_official_eval/release_manifest.json`，使用完整原始发布权重，不能传训练用 runtime/step/geometry。它与两条 `official_final_s1_...` 训练 checkpoint 分支不同。

入口使用既有资源设置：`pdgpu-5090`、1 GPU、8 CPU、32G、1 node，镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`，conda `mellow_c8204d8`。保持 HF/Transformers 离线。提交前仅在 CPU 解析显式参数；不加载权重、不读取数据、不额外提交 GPU 作业。`RSMOL_CONDA_BASE` 可覆盖 conda 安装目录。

已在计算节点或自己的 GPU allocation 中时，直接用同一计算入口：

```bash
bash code/RSmol/mizar_s1_ced_mmsu_eval/run.sh <同一组目标参数>
```

只检查参数而不提交或评测时，在正式提交命令中增加 `--check-args`。

## 数据与推理口径

默认读取用户已上传的目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/MMSU/
  MMSU data/data/train-00000-of-00003.parquet   (1667)
  MMSU data/data/train-00001-of-00003.parquet   (1667)
  MMSU data/data/train-00002-of-00003.parquet   (1666)
  MMSU code/mmsu_evaluation.py
```

`--mmsu-root` 可整体改位置；`--dataset-dir` 指向包含 `data/` 的 HF 目录，`--evaluation-script` 指向官方脚本，二者也可单独覆盖。包含空格的路径始终作为单个参数传递。不访问 Hugging Face 网络，不解包或批量导出音频。

按 shard 文件名及其原始行序建立全局 row_index；模型加载前核验三片的 schema、1667/1667/1666 行数、5000 个唯一 ID、字段类型，以及 `answer_gt` 与原选项的文本精确匹配。四个选项的位置和 JSON null 均保持。官方 f-string 题面中的空选项显示为 `None`。

经用户确认，题面采用 MMSU 官方英文指令与原始大小写：

```text
Choose the most suitable answer from options A, B, C, and D. You must respond with only A, B, C, or D.

Question: <原始 question>

A. <choice_a>
B. <choice_b>
C. <choice_c>
D. <choice_d>
```

GT、任务名和分类不进入模型输入。其他推理设置沿用当前 CED：

| 项目 | 设置 |
| --- | --- |
| 音频 | 优先 Parquet bytes；SoundFile FP32 解码，以真实头部采样率为准；先单声道，再 torchaudio resample 至 32 kHz |
| 时长 | 使用训练 `fixed20.fixed_len`：整段居中截到 20 秒或右补零，再连续分成两个 10 秒输入；不复制短音频 |
| Prompt | 使用训练 tokenize：EOS suffix、右侧 PAD `!`、固定 256 tokens；保留 MMSU 大小写 |
| Prefix | 126 audio + 1 SEP + 256 prompt = 383；首个答案 logit 在 382 |
| 数值 | FP32、eval/frozen、无 autocast、TF32 关闭；沿用已有 seed 与运行时审计 |
| 生成 | 逐 token argmax、每次完整 forward、attention_mask=None、无 KV cache；首个 EOS 或 300 新 tokens 停止 |
| Decode | `skip_special_tokens=False`、`clean_up_tokenization_spaces=False`；原始空白、大小写、EOS 全部保留 |

直接调用现有严格 `load_runtime` 和 `greedy_decode`。简单递归路线传入该样本的 A4/A8 context；multilayer 路线调用原 `generate_prefix_inference`、`sample_context`，再将原 context 传给解码器。没有修改或 monkeypatch 原 runtime 的 prompt/生成行为。

这是 **MMSU 在项目 CED 协议下的结果**。固定 20 秒和 256-token 截断可能影响题目信息，不等同于复现其他模型原生推理设置的排行榜结果。逐样本及汇总记录音频截断/补零、原始 prompt token 数和截断数。

## 输出、续跑与官方评分

| 输出 | 内容 |
| --- | --- |
| `run_config.json` | checkpoint/runtime、三份数据、scorer、评测和模型源码、协议的身份与 SHA256 |
| `raw_generations.jsonl` | 按全局行号追加的完整原始回答、token IDs、元数据、逐样本处理统计及校验摘要 |
| `progress.jsonl` | 已完成样本事件；恢复以 durable raw 记录为准 |
| `model_audit.json` | 现有严格 loader 和实际 forward 的审计，完整结果重评分时保留 |
| `predictions_official.jsonl` | 原官方元数据 + 原始 `response`，含真实 null 选项 |
| `official_evaluation.txt` | 未改的官方 scorer 标准输出与错误输出 |
| `evaluation_report.json` | 推理覆盖、评分执行状态、实际分母、原始精度 overall/category 分数、处理统计和错误 |

每个输出目录持有进程锁，进程异常退出后锁自动释放。再次用相同参数、相同输出目录启动，会校验 run identity，恢复已经落盘的连续样本，只推理剩余样本。raw 写成功但 progress 未写的样本也能恢复。只修复最后一个缺少换行的未完成写入；已完整换行但 JSON 损坏、样本顺序/ID/内容不匹配均报错。更换权重、runtime、数据、scorer 或源码时使用新输出目录。完整 5000 条 raw 已存在时只重评分，不再加载模型。

遇到音频、数据、模型或生成错误立即停止并保留结果，不跳过样本。严格完成 5000 条唯一预测后才调用：

```bash
python '/.../MMSU code/mmsu_evaluation.py' '/.../predictions_official.jsonl'
```

脚本源文件固定为调研时版本，归一 LF 后 SHA256：
`b6d5d35d2755244019a78e601495ea74e722681bf6f7e1fd0179bab7cabca3eb`。
只容许 Git/上传产生的 CRLF/LF 差异；不修改脚本，不自动下载替换。测试 fixture 是该原文件的原样副本。

官方 scorer 以选项文本与 `answer_gt` 精确相等判断正确；它会排除部分格式错误的回答，空回答和字符串 `None` 则计错。保留这种官方行为，不提取答案字母、不转大写、不删除 EOS。结果中的 accuracy 是 0～1 的比例。执行官方 CLI 后，再调用同一个官方函数取得未四舍五入的分数/计数，并核对 CLI 输出。

三个状态分开报告：5000 条推理覆盖、scorer 进程执行、官方实际计分分母。即使 scorer 返回 0，只要实际分母小于 5000，仍保留官方分数与输出，但整轮 `status=FAILED`、退出码为 1；这不表示推理记录丢失，也不会偷偷改写回答以提高覆盖。官方现成统计按 category/sub-category，47 个 task 的元数据覆盖另有记录。

## 验证与代码同步

可选 CPU 回归不属于正式入口的自动前置步骤：

```bash
python -B tests/validate_mizar_s1_ced_mmsu_eval.py
# 可额外只读核验实际上传数据和音频；不需要 GPU/权重。
python -B tests/validate_mizar_s1_ced_mmsu_eval.py \
  --assets-root '/hpc_stor03/sjtu_home/jinwei.zhang/data/MMSU'
```

覆盖真实三片 Parquet 的顺序、空选项、异常 schema/重复 ID；官方 scorer 的原始 EOS、大小写及分母行为；完整 5000 条 fixture 流程、中断续跑、身份变化和损坏记录；真实 Bash + 模拟 vc 的资源/路径参数/离线环境；现有 CED MMAU/MMAR 相关回归。`reference_audit.json` 固定了原有 28 个评测文件的原始字节和 LF SHA256。

真实 tensor/resampling 检查缺依赖时明确 skip；不将 fixture 推理或 CPU 检查宣称为远程 checkpoint 评测成功。无需安装本地权重或训练数据。

2026-11-13 本地验收：122 项中 119 通过、3 项真实 Torch/torchaudio 检查缺依赖跳过，原有 28 个评测文件的原始字节校验全部通过。使用独立临时目录中的 PyArrow/SoundFile 核验了实际三片 5000 条元数据，并实际解码全部 5000 段音频。元数据为 47 个任务；音频包含 8 种真实采样率、450 段双声道，不能统一假设为 HF 元数据标注的 16 kHz；实际时长约 0.302～35.915 秒，145 段会被固定 20 秒策略截断，4855 段需补零。298 段的头部帧数与实际解码长度不同，因此统计与推理均以实际解码结果为准。完整记录见 `audit_reports/mizar_s1_ced_mmsu_eval_validation.json`。

本次同步的新增文件为：

```text
code/RSmol/run_mizar_s1_ced_small_mmsu_eval_5090.sh
code/RSmol/mizar_s1_ced_mmsu_eval/           # 全目录，含 README 与 reference_audit.json
tests/test_mizar_s1_ced_mmsu_eval.py
tests/test_mizar_s1_ced_mmsu_dependencies.py
tests/test_mizar_s1_ced_mmsu_shell.py
tests/validate_mizar_s1_ced_mmsu_eval.py
tests/fixtures/mmsu/mmsu_evaluation.py
audit_reports/mizar_s1_ced_mmsu_eval_validation.json
RM.md                                      # 仅前置本次记录
```

用户在本地自行选择并提交这些文件、push 后，远程 `git pull --ff-only`，再用上述模板填入实际目标运行。本次实现没有代为 commit/push 或提交真实评测作业。
