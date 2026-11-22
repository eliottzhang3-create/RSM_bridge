# 2026-11-22 指定七个 checkpoint 的统一评测

本地修改代码，经 GitHub 同步到 Linux 后提交。此入口固定本次用户指定的完整 checkpoint 路径，不搜索 latest、不更改训练。
清单见 `targets.json`，全部 step=34626。合计 **19 项：MMSU 7 项，ReasonAQA5000、MMAU test-mini、MMAR 各 4 项**。
audio7_read2 在两组需求中的 MMSU 去重，只提交一次；其四项全部使用本批次新目录重新生成。

| 提交别名 | 输出 run（均位于 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/`） | 本次任务 |
|---|---|---|
| mesh_original | `mizar_s1_ced_small_gbs128_text71427_mesh_3090/formal_20261119_141531507533438` | MMSU |
| audio7_read2 | `mizar_s1_ced48_audio7_read2_lora_r8_gbs128_text71427_3090/formal_20261120_111448563391250` | 四项 |
| listen | `mizar_s1_ced48_listen_lora_r8_gbs128_text71427_mesh_3090/formal_20261119_141724655549596` | MMSU |
| loop_lora | `mizar_s1_ced48_loop_lora_r8_gbs128_text71427_mesh_3090/formal_20261119_141712678007968` | MMSU |
| wfuse | `mizar_s1_ced48_wfuse_lora_r8_gbs128_text71427_mesh_3090/formal_20261120_124016246076398` | 四项 |
| mesh_reinit | `mizar_s1_ced_small_gbs128_text71427_mesh_3090/formal_reinit34626_lr2e4_20261121_140510343409202` | 四项 |
| smol_gbs128 | `mizar_s1_ced_small_gbs128_3090/formal_20261121_150129461846142` | 四项 |

每个 run 后接 `/formal/checkpoints/model--step-34626.ckpt`，同 run `/formal/runtime.json` 自动绑定。

## 推理与评分口径

所有模型使用当前同一套评测流程。MMSU 统一最新紧凑全小写：
`which country? a) india b) none c) china d) japan`。
不附加只答字母指令，四个原选项槽不删除、不重排，GT 不进入输入。
评分沿用 `mmsu_answer_text_contains_v2`：规范化完整答案文本包含，忽略生成字母，外部原 scorer 和 5000 分母不变。
新协议 `mmsu_compact_lowercase_ced_fixed20_raw_greedy_answer_text_v3`。

共同数值设置保持既有实现：

- FP32、eval、seed1234、TF32 关闭、无 autocast，batch1；不改模型 forward。
- 单声道32k；fixed20 中心裁剪/右补零，然后10+10拆分；CED 按原训练实现拼回整段处理。
- audio126 + SEP1 + prompt256 = prefix383；首答案取 logit382；原 EOS 后缀、PAD `!`、截断/补齐顺序保持。
- attention_mask=None，无 KV cache，每 token 全序列重算，argmax，最多300 tokens，首 EOS 停止。
- 每题只编码一次音频，分支每次 LM forward 显式接收该题 A4/A8；memory 每次 forward 重建。
- ReasonAQA 使用原 seed1234 固定5000 manifest/完整答案精确匹配；MMAU/MMAR 各1000，原始输出直接交原官方 scorer。
- 四个评测保留各自原指标，不能横跨 benchmark 比较数值。所有模型在同一 benchmark 内使用相同数据/预处理/评分。
- 完整分母校验、坏行拒绝和 append-only 续跑保持；未完成推理不评分。

架构分别调用保存 checkpoint 时对应的训练实现，严格覆盖全部模型权重，不恢复 optimizer/scheduler/RNG/游标。
MeSH 为28物理层/42次执行，保留训练后的 routers/LoRA/音频分支；plain 为标准 HF LlamaForCausalLM，30个独立物理层，0..29轨迹，无MeSH/LoRA。
mesh_reinit 额外核对原音频 checkpoint 仅权重初始化来源及2e-4/2e-5配方；不再执行初始化或重置参数。

## 固定训练源码，使用当前评测器

旧 MeSH 与重训 MeSH 路线名相同，但训练源码不同。不能关闭 code SHA 检查，也不能用历史整套评测器退回旧 MMSU 题面。

- 前五项训练源码固定 `8a9f1f76722e77cba4be429a578c63fdcf6ee527`。
- 重训/普通 SmolLM2 固定 `75dcda3a7bcd8765044b14ecd4e408fe33dfdda6`。
- 只从本地 Git object 通过 `git archive` 提取所需训练后端/共享模型到 `code/RSmol/log/ced_eval_20261122_sources/<revision>/`。
- `source_manifest.json` 固定逐文件 SHA、源码库存和七个训练 code_digest。每个计算进程再次核对；缺 Git 历史、源码不同、分支错配都明确拒绝。
- 本批次评测器及当前共享 MMSU dataset/parsing/scorer 始终来自当前 checkout。不会切换分支或修改现有训练目录。
- 申请 GPU 前每个 checkpoint 做一次 CPU mmap 合同预检。`READY_FOR_GPU` 仅表示源码和合同预检通过；真实资产/张量/架构仍由 GPU loader 严格审计。

## 提交命令

先将本地已交付代码经 GitHub 同步，再在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# 只打印19项绑定与vc参数，不读权重、不提交
bash code/RSmol/run_ced_eval_20261122_4090.sh all --check-args

# 一次提交本次全部19项
bash code/RSmol/run_ced_eval_20261122_4090.sh all
```

全部 `pdgpu-4090 / 1 GPU / 8 CPU / 32G / 1 node`，沿用原容器、`mellow_c8204d8`。
默认逐项提交，某目标预检失败不申请其 GPU，其他目标继续；任何失败最终退出非零并记录报告。

也可按模型提交以下七条，**与上面的 all 方式择一，避免重复排队**：

```bash
# 以下3条各1项：修正版MMSU
bash code/RSmol/run_ced_eval_20261122_4090.sh mesh_original
bash code/RSmol/run_ced_eval_20261122_4090.sh listen
bash code/RSmol/run_ced_eval_20261122_4090.sh loop_lora

# 以下4条各4项：ReasonAQA5000 / MMAU test-mini / MMAR / 修正版MMSU
bash code/RSmol/run_ced_eval_20261122_4090.sh audio7_read2
bash code/RSmol/run_ced_eval_20261122_4090.sh wfuse
bash code/RSmol/run_ced_eval_20261122_4090.sh mesh_reinit
bash code/RSmol/run_ced_eval_20261122_4090.sh smol_gbs128
```

如需19项逐条控制，下面也是等价替代，不能与上述重复提交：

```bash
bash code/RSmol/run_ced_eval_20261122_4090.sh mesh_original mmsu
bash code/RSmol/run_ced_eval_20261122_4090.sh listen mmsu
bash code/RSmol/run_ced_eval_20261122_4090.sh loop_lora mmsu
bash code/RSmol/run_ced_eval_20261122_4090.sh audio7_read2 reasonaqa5000
bash code/RSmol/run_ced_eval_20261122_4090.sh audio7_read2 mmau
bash code/RSmol/run_ced_eval_20261122_4090.sh audio7_read2 mmar
bash code/RSmol/run_ced_eval_20261122_4090.sh audio7_read2 mmsu
bash code/RSmol/run_ced_eval_20261122_4090.sh wfuse reasonaqa5000
bash code/RSmol/run_ced_eval_20261122_4090.sh wfuse mmau
bash code/RSmol/run_ced_eval_20261122_4090.sh wfuse mmar
bash code/RSmol/run_ced_eval_20261122_4090.sh wfuse mmsu
bash code/RSmol/run_ced_eval_20261122_4090.sh mesh_reinit reasonaqa5000
bash code/RSmol/run_ced_eval_20261122_4090.sh mesh_reinit mmau
bash code/RSmol/run_ced_eval_20261122_4090.sh mesh_reinit mmar
bash code/RSmol/run_ced_eval_20261122_4090.sh mesh_reinit mmsu
bash code/RSmol/run_ced_eval_20261122_4090.sh smol_gbs128 reasonaqa5000
bash code/RSmol/run_ced_eval_20261122_4090.sh smol_gbs128 mmau
bash code/RSmol/run_ced_eval_20261122_4090.sh smol_gbs128 mmar
bash code/RSmol/run_ced_eval_20261122_4090.sh smol_gbs128 mmsu
```

按评测类型整组提交也可：`all mmsu`为7项，`all reasonaqa5000`、`all mmau`、`all mmar`各4项。
只查合同用 `all --preflight-only`，不提交。不要在前一次作业仍排队/运行时重复提交。

## 日志、结果与重跑

仓库根查看：

```bash
ls -lt code/RSmol/log/ced_eval_20261122/
```

- `<时间>_<pid>.submission.json`：每目标预检结果、19项 checkpoint/output、完整vc参数、stdout/stderr和提交返回码。
- `ced22-<target>-<task缩写>-<hash>.JOB.log`：GPU stdout/stderr，`tail -f <实际日志路径>`实时查看。
- `SUBMITTED` 只表示 vc 接受提交；完整评测以各输出目录 `evaluation_report.json` 的 PASS 为准。

新结果目录：
`<run>/eval/ced_eval_20261122_v1/<task>/model--step-34626/`。
task 为 `reasonaqa5000`、`mmau`、`mmar` 或 **`mmsu_compact_lowercase_v1`**。
目录中有 `run_config.json`、`model_audit.json`、`raw_generations.jsonl`、`evaluation_report.json`及各任务评分产物。

旧评测目录及 audio7_read2 旧的半程输出不读取、不覆盖。本次第一次提交从新命名空间重新生成；
本批次若再发生中断，确认作业退出后重新运行同一条命令，在源码/权重/数据/协议身份不变时续跑剩余题。
完整运行会复用 raw 重评分。修改代码或数据后不要强行复用身份不匹配的目录。

`compare.py`可对同 benchmark 的完成目录做只读口径检查与CSV汇总；会拒绝不完整覆盖、MMSU旧题面协议、数据/源码/顺序不同及raw/评分矛盾：

```bash
bash code/RSmol/ced_eval_20261122/run.sh compare --benchmark mmsu \
  --report-dir /absolute/completed/run1 \
  --report-dir /absolute/completed/run2 \
  --output /absolute/new/comparison.csv
```

## 本地验证与同步文件

运行 `python -B tests/validate_ced_eval_20261122.py`，证据写入 `audit_reports/ced_eval_20261122_validation.json`。
测试执行真实 Git 解包/源码合同、Bash→Python参数链、mock vc、19项完整尺寸合成工作流、最新MMSU题面/原scorer、
断点续跑、错架构/同route错源码拒绝、FP32完整权重覆盖顺序与30/42层hooks。
模型/CUDA/音频/Parquet及bulk fsync在合成工作流中明确模拟；可选真实CPU小模型测试按依赖是否存在执行。
本地验收不代表远程生产 checkpoint 加载或4090 GPU评测已经通过。

本次最终新包验收35项：28通过、7项真实CPU小模型因缺Torch/Transformers/Safetensors明确跳过；
另7项既有MMSU紧凑题面/旧输出拒绝回归全部通过，总计42项中35通过、7跳过。
2898份本轮开始时已有生产源码摘要保持，Python3.10、两份Bash入口、文档5个Bash代码块和whitespace检查通过。
本轮未commit/push或远程提交。

同步须包含本包、顶层入口、tests/报告/RM，以及此前已交付的紧凑小写MMSU改动。
本包复用当前 `mizar_s1_ced_mmsu_eval`、`mizar_s1_ced_reasonaqa5000_eval`、`text71427_ced_eval/observers.py`和既有scripts。
仅同步本包而遗漏上一轮未提交的MMSU实现，将得到错误题面或协议；不应只拷贝单份Shell。
本工具不自动commit/push，不提交训练或评测真实作业。
