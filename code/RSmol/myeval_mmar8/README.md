# MyEval MMAR8：三模型对照与 A4/A8 置零实验

本地 Windows 修改代码，经 GitHub 同步到远程 Linux；生产权重、音频和作业留在服务器。
本包从既有 `ced_eval_20261122` 复制后隔离实现，保留既有正式评测和训练文件。
提交入口为 `code/RSmol/run_myeval_mmar8_4090.sh`，默认四个独立 4090 作业。
本轮仅交付代码和本地验收，未 commit/push、连接服务器或提交真实作业。

## 1. 固定实验

两个 checkpoint 都是 `formal/checkpoints/model--step-34626.ckpt`，runtime 为同 run 的 `formal/runtime.json`。
以下 run 路径前缀均为 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/`：

| 别名 | run | 本次任务 |
|---|---|---|
| `base` | `mizar_s1_ced_small_gbs128_text71427_mesh_3090/formal_20261119_141531507533438` | 自定义 8 音频题 |
| `read2` | `mizar_s1_ced48_audio7_read2_lora_r8_gbs128_text71427_3090/formal_20261120_111448563391250` | 自定义 8 音频题 |
| `read2_zero48` | 与 `read2` 完全相同 checkpoint | 自定义 8 音频题、完整 MMAR 各一作业 |

模型、checkpoint、step、训练几何与干预绑定在 `registry.py`，入口不能替换其他权重。
每作业 `pdgpu-4090`、1 GPU、8 CPU、32G、1 node，复用原镜像与 `mellow_c8204d8` 环境。
本包不运行 MMAU，也不把自定义四维分类自动等同于 MMAR 原题分类。

### 置零的精确定义

严格加载完整 checkpoint 并完成模型审计之后，将本题 CED 第 4、8 层经过各自 MLP 得到的
`A4/A8` 两个只读 seed 张量替换为 `zeros_like`。它们对应七槽 audio memory 的第 5、6 槽（从 0 起）。
每道题独立编码；每个自回归 token 的完整 LM forward 都显式使用本题的零张量，拒绝陈旧或替换的 context。
每次 forward 的 memory 仍从输入重新构建。

保留 CED 主干及最终音频 prefix、两套 LoRA 的 392 个矩阵、全部训练参数、两个七路 audio read softmax、
原五槽读写和 28 物理层/42 次执行。七路 softmax 分母保留，不删除槽或重新归一化，也不重新训练。
因此 `read2` 与 `read2_zero48` 比较表示这两路 seed 值在已训练系统中的作用；置零后的模型仍有 LoRA 等改动，
不能当作重新训练的 `base`。

## 2. 自定义题与计分

远程标注固定为 `/hpc_stor03/sjtu_home/jinwei.zhang/data/MyEval/mmar8selected.jsonl`。
本地原件位于 `C:\Xlance\assets\MMAU analysis\MyEval\mmar8selected.jsonl`，不打包进代码库。
文件 normalized-LF SHA256：`a670c97eef24859c6dc015ba137e723fa2e7205d198a0b4d234ced0e642b7815`。
更换或修改标注会被拒绝，应另开版本，不能与本轮结果混用。

共 8 音频、67 个原始自定义问题：53 个 closed-ended、14 个 open-ended。
只使用标注 `questions` 中的题目/选项；旧 `original` 问题、旧 `models` 回答、GT、维度和 caption 等元数据均不进入模型题面。
GT/分类保留在输出清单供评分和核查。open-ended（含 caption）仅输出，不评分。

每道选择题枚举选项**索引**的所有排列，重新标为 `a)` 至 `e)`；每个排列均独立生成。
同一道题按下表压成一个对/错结果，再计算 ACC：

| 选项数 | 题数 | 每题排列数 | 本题通过门槛 |
|---:|---:|---:|---:|
| 2 | 28 | 2 | 2/2 |
| 3 | 1 | 6 | 6/6 |
| 4 | 23 | 24 | 至少 21/24 |
| 5 | 1 | 120 | 至少 115/120 |

每模型 734 个选择题排列 + 14 个开放题 = **748 次生成**；三模型合计 2,244 次。
屏蔽模型另跑原始完整 MMAR 1,000 题，全部四作业共 3,244 次。
额外完整 MMAR 使用原问题与原固定选项顺序，不做全排列，沿用原官方完整评分。

| 自定义维度 | closed-ended ACC 分母 |
|---|---:|
| P / Perception | 29 |
| S / Semantic Understanding | 11 |
| W / World Knowledge | 7 |
| R / Reasoning | 6 |
| 总 ACC | 53 |

总 ACC 是通过题数 / 53，不是四维 ACC 的简单平均，也不是正确排列数 / 734。
734 次排列准确率仅作为附加诊断；保留每题正确排列数/门槛及每次原始回答。
`source_category/source_sub_category` 是这段音频对应的原 MMAR **原问题**分类，保留作来源信息，
不用于给新问题自动贴能力标签；本次四维 ACC 按自定义题的 `dim` 统计。

### 已知评分边界

按约定，每个排列把原始生成原样交给官方 MMAR `string_match(answer, prediction, choices)`，
不新增字母提取器、不删除 EOS、不清理或改写模型回答。只答字母不自动映射为答案文本。
官方匹配器使用小写词集合，忽略词序、重复次数和词语间关系。
当前 6 道题存在错误选项与正确答案词集合相同的情况：

- `v2oCIDFP4oU_00-00-29_00-00-59::q3`
- `H3ixLqnHqCg_00-00-00_00-00-13::q2`、`::q5`
- `kJkMJJMoVIc_00-00-00_00-00-30::q5`、`::q6`
- `e-Q-lcBY89c_00-00-00_00-00-25::q5`

这些题可能将词序/角色错误算对；全排列门槛不能修复这个局限。`scores.json` 和本地验收报告逐题列出冲突选项，
请结合 `permutation_scores.jsonl` 看实际回答。未来若换严格选择解析规则，应另立评分版本，不能把两种口径混作一组 ACC。

## 3. 与既有评测一致的推理

沿用 `run_text71427_ced_eval_4090.sh` 的 CED 数值协议，经已有统一评测的复制版本加载模型：

- FP32 / eval、seed 1234、TF32 关闭、无 autocast；严格完整 state_dict，校验 dtype/finite/tied 与训练资产。
- mono 32 kHz，中心裁剪或右补零至 20 秒，再拆 10+10 秒。
- 选择题 `question a) ... b) ...` 紧凑整体小写；开放题只给本题文本并小写，无新增输出字母指令。
- 原 EOS 后缀、PAD `!`、原分词及截断/右补齐顺序；prompt 256，audio126+SEP1+prompt256=prefix383，首答案 logit382。
- attention_mask=None、无 KV cache，每 token 全序列重算，greedy argmax、最多 300 token，首 EOS 停止。
- 28 物理层按 `0..6, 7..20, 7..20, 21..27` 执行；只读 hooks 核对 router 边界、七槽 read 和两轮 LoRA 调用。

保留原 prompt 256 token 上限；raw 记录 `prompt_original_token_count/prompt_truncated` 及实际输入 token，
用于识别过长题面的截断。比较程序核对三模型实际 prompt token、原 mask 及两段解码波形指纹一致。

8 段音频中，按标注时间长度有 5 段超过 20 秒；保留原评测裁剪，不改标注：

| item_no | 原片段秒数 | 相对该片段保留区间（秒） |
|---:|---:|---|
| 1 | 22 | 1–21 |
| 3 | 30 | 5–25 |
| 10 | 30 | 5–25 |
| 16 | 25 | 2.5–22.5 |
| 26 | 24 | 2–22 |

表格按标注推算，实际采样数依解码；若问题需要被裁掉的开头/结尾，模型无法获得该证据。
`dataset_audit.json` 保存此审计，实际输入波形指纹另存 raw。8 段同源多题是探索性诊断集，不表示完整 benchmark 的总体能力估计。

## 4. 源码与输出隔离

两个 checkpoint 的训练源码均固定到 Git revision `8a9f1f76722e77cba4be429a578c63fdcf6ee527`。
当前同名 base 训练目录已演进，不能直接用当前训练代码冒充历史版本。
本包只通过本地 `git archive` 提取需要的训练后端和共享模型到
`code/RSmol/log/myeval_mmar8_sources/<revision>/`，逐文件和训练 `code_digest` 精确校验。
不会切分支、改 checkout 或使用网络；远程仓库需保留该 Git 对象，浅克隆缺失时应先补齐历史。
新缓存与原正式缓存分开。评测器仍取当前本包，不从旧 Git 恢复旧评测题面。

默认输出为：

`<对应run>/eval/myeval_mmar8_v1/<base|read2|read2_zero48>/<myeval8|mmar>/model--step-34626/`

提交报告/GPU 日志在 `code/RSmol/log/myeval_mmar8/`。
`--output-root /path/to/new-root` 可将四份结果一起放到新根，子路径仍为 `<alias>/<task>/model--step-34626/`。
该参数在提交、续跑时须保持一致；比较时相应传入三个 `--result-dir`。
不要把新输出指定到正式评测或历史其他协议的目录。

## 5. 同步并在远程提交

本地仓库根可按明确清单暂存；以下命令交由用户执行，尚未代为运行：

```powershell
git add -- RM.md code/RSmol/myeval_mmar8 code/RSmol/run_myeval_mmar8_4090.sh tests/test_myeval_mmar8.py tests/test_myeval_mmar8_runtime.py tests/validate_myeval_mmar8.py tests/fixtures/myeval_mmar8_official_scorer.py audit_reports/myeval_mmar8_validation.json
git diff --cached --stat
git commit -m "Add isolated MyEval MMAR8 three-model evaluation"
git push
```

须同步整个包和当前已提交的共享评测依赖，不能只复制入口 Shell。
标注文件已由用户单独上传到上述远程数据路径，不通过 Git 传输权重/音频。

远程仓库根执行以下预览，确认三模型/四任务和路径：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_myeval_mmar8_4090.sh all --check-args
```

`--check-args` 不读取模型或数据，不提交。
正式入口在提交前自动进行标注/八音频路径预检，并对两个 checkpoint 各做一次 CPU mmap 合同/历史源码预检。
若要只做这一预检，可执行：

```bash
bash code/RSmol/run_myeval_mmar8_4090.sh all --preflight-only
```

一次提交四项：

```bash
bash code/RSmol/run_myeval_mmar8_4090.sh all
```

或者按需逐项提交；与上面的 `all` 择一：

```bash
bash code/RSmol/run_myeval_mmar8_4090.sh base myeval8
bash code/RSmol/run_myeval_mmar8_4090.sh read2 myeval8
bash code/RSmol/run_myeval_mmar8_4090.sh read2_zero48 myeval8
bash code/RSmol/run_myeval_mmar8_4090.sh read2_zero48 mmar
```

`read2_zero48` 不写任务名会提交它的两项；`all myeval8` 只提交三个自定义评测。
某 checkpoint 预检失败则不提交它的任务，其余目标继续；某次 vc 失败也继续其他项，整体返回非零并保存原因。
`READY_FOR_GPU` 仅代表 CPU 预检通过，`SUBMITTED` 仅代表 vc 接收，最终查看各输出 `evaluation_report.json`。
没有附加 GPU smoke 作业。

中断后确认原作业已退出，再执行相同单项提交命令。只续跑未生成项，保留已有 raw；
末行不完整时恢复，完整坏行/身份冲突拒绝。源码、权重、runtime、标注、音频文件身份、scorer、门槛和干预必须一致。
同输出同时运行有进程锁；不要在原作业仍排队/运行时重复提交。修改代码/标注后的新实验须用新输出根，不手改 `run_config.json`。

## 6. 看分数和开放题

每个自定义输出目录中：

- `evaluation_report.json`：状态、生成覆盖、四维/总 ACC 与运行身份。
- `accuracy.csv`、`scores.json`：按 53 道原题统计，后者含逐题排列通过数与词集合冲突。
- `task_manifest.json`、`dataset_audit.json`、`run_config.json`：每次排列、源题对应、裁剪与协议身份。
- `raw_generations.jsonl`：追加保存完整原始生成、token、输入指纹和干预。
- `permutation_scores.jsonl`：734 次排列逐次判定。
- `open_answers.jsonl`：14 个开放题输出，无评分。
- `model_audit.json`：完整权重/架构审计与执行 hooks 计数；零 seed/每次 forward 的 context 检查计数。
- `official_scorer.py`：本次固定官方匹配源码，供 CPU 重评分。

额外完整 MMAR 的目录使用原 runner 的输出：`predictions_official.json`、`official_evaluation.txt`、
`raw_generations.jsonl` 和含模型/原官方分类评分状态的 `evaluation_report.json`，完整分母 1,000。

三份自定义评测结束后，从远程仓库根运行 CPU 对比，无需 GPU、不重新读取权重或音频：

```bash
bash code/RSmol/myeval_mmar8/run.sh compare
```

默认写到仓库 `outputs/RSmol/myeval_mmar8_comparison_v1/`：
`comparison.csv/json` 为四维和总 ACC；`open_answers_comparison.md/jsonl` 为同题三模型并排输出。
先核对完整 raw、保存 scorer、相同任务/实际输入和代码身份，再重算评分；拒绝半程、混数据或不一致的两份 read2 权重。
源评测目录只读，不覆盖它们的分数、进度或报告。

自定义输出根时，显式指定三份目录（以下先设置 `RESULT_ROOT` 为提交时 `--output-root` 的值）：

```bash
bash code/RSmol/myeval_mmar8/run.sh compare \
  --result-dir "$RESULT_ROOT/base/myeval8/model--step-34626" \
  --result-dir "$RESULT_ROOT/read2/myeval8/model--step-34626" \
  --result-dir "$RESULT_ROOT/read2_zero48/myeval8/model--step-34626" \
  --output-dir "$RESULT_ROOT/comparison"
```

单个自定义结果只读重评分：

```bash
bash code/RSmol/myeval_mmar8/run.sh rescore --result-dir "$RESULT_ROOT/base/myeval8/model--step-34626"
```

以上比较/重评分需保持实际推理时的评测代码版本。它们只使用已完成结果，不代表重新验证生产权重/音频。

## 7. 本地验收及边界

```powershell
python -B tests/validate_myeval_mmar8.py --annotations 'C:\Xlance\assets\MMAU analysis\MyEval\mmar8selected.jsonl' --metadata 'C:\Xlance\assets\MMAR\MMAR-meta.json' --include-reference-regressions
```

本地最终 33 项：32 通过，1 项真实微型 CPU Torch 模型测试因缺 Torch/Transformers/Safetensors 明确跳过。
报告 `audit_reports/myeval_mmar8_validation.json` 保留此前失败和后续通过的运行记录。
覆盖三组各 748 次及额外 1,000 次的完整尺寸**合成**流程、中断/尾行恢复、全部四档门槛、原题分母、
数据泄漏防护、真实 Git 历史源码/合同、原生成核心 AST、七槽/softmax 数值及真实 Bash 参数链。
原有 checkpoint loader 与只读 hooks 的 5 项回归一并纳入 33 项；3,529 份既有 tracked Python/Shell/JSON 的 normalized-LF SHA 保持。

七槽数值测试以 NumPy 张量 API 执行实际生产 `_mixed_interaction/_read/_write` 方法；不冒充完整 Torch/CUDA 模型。
完整 MMAR 测试调用真实未改官方评分 CLI，仅用测试目录补齐它未调用的 `tqdm` 显示导入依赖，评分逻辑未替换。
`tests/fixtures/myeval_mmar8_official_scorer.py` 来自本地官方资产 `C:\Xlance\assets\MMAR\code\evaluation.py`，
仅换行为 LF，SHA256 `fb046e16b9c0d2482b5b0698bc242e75a34d24a8165ba2a8f344572e56ee7918`；原始空白保留。
生产执行始终使用服务器数据目录的官方脚本，并核对支持的固定内容摘要。

合成流程中模型生成、音频解码/CUDA 和 bulk fsync 明确模拟。真实生产 checkpoint、真实音频解码、4090 推理、
完整生成数值与远程 vc 尚未验证；须以服务器本次输出为依据，不能把本地 PASS 或提交预检当生产评测 PASS。
