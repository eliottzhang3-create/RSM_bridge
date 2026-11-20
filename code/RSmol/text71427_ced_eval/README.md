# 7–14×2–7 CED：六条路线、四项隔离评测

本地 Windows 维护核心代码，通过 GitHub 同步；checkpoint、数据和实际 GPU 评测位于远程 Linux。
本包位于训练目录之外，不修改六条训练路线、共享模型或任何旧评测源码。

## 支持的模型

| `MODEL` | 对应训练入口（`code/RSmol/` 下） | 模型交互 |
|---|---|---|
| `base` | `run_mizar_s1_gbs128_text71427_mesh_ced_small_3090.sh` | 原五槽 MeSH |
| `loop_lora` | `run_mizar_s1_gbs128_text71427_mesh_ced48_loop_lora_r8_3090.sh` | read 后 CED4→8 注入、两轮 LoRA |
| `listen` | `run_mizar_s1_gbs128_text71427_mesh_ced48_listen_lora_r8_3090.sh` | 原 write→listen→原 read、两轮 LoRA |
| `audio7_read2` | `run_mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_lora_r8_3090.sh` | 音频 read0/1 七槽、只读 A4/A8 seeds |
| `wfuse` | `run_mizar_s1_gbs128_text71427_mesh_ced48_wfuse_lora_r8_3090.sh` | 原 write value 拼接融合、两轮 LoRA |
| `audio7_read2_ar6` | `run_mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_ar6_lora_r8_3090.sh` | 七槽 read0/1、六个独立音频 routers |

全部为 hidden576、28物理层、42次执行：`0..6,7..20,7..20,21..27`。
后五条保留物理7..20的两轮独立 LoRA r8/alpha16/scale2/dropout0，392个 A/B 矩阵。
加载各自真实训练构造器，并用完整音频 checkpoint 严格覆盖全部权重；不会在加载后重置 gate、router、fusion 或 LoRA。
普通模型无额外 context；后五条每题生成一次 `(A4,A8)`，在每次完整 LM 重算中显式传入。
每次 forward 重建 memory，audio7 两条路线只写前五槽，槽5/6只读。

## 固定推理与评分协议

直接复用原数据、分词、评分和 durable storage；必要的外层 runner 隔离复制。
测试逐函数比较实际生成、音频处理、评分输入构造和 greedy 循环的 AST。

- FP32、eval、seed1234、关闭 autocast/TF32，单卡 batch1。
- mono/32k；单文件 fixed20 中心裁剪或右补零后分10+10秒；ReasonAQA 显式双文件各固定10秒。
- `126 audio + SEP1 + prompt256 = 383`，首答案 logit 位置382。
- 按原训练分词追加 ` <|endoftext|>` 再截断/填充，已有 `!` 为 PAD，不扩词表。
- `attention_mask=None`、`use_cache=False`，完整重算、纯 argmax、最多300新 token、首次 EOS 停止。
- 原始输出保留 EOS、空白、选项标记及 token IDs。

| 任务参数 | 范围 | 题面与判分 |
|---|---|---|
| `reasonaqa5000` | 已有同一份 seed1234 的5000条 manifest | 原 input 小写，无新增指令；完整答案规范化精确匹配，保留字母/括号/标点 |
| `mmau` | test-mini 全1000条 | 原固定选项顺序、小写题面；完整 raw generation 交原官方 scorer |
| `mmar` | 全1000条 | 原固定选项顺序、小写题面；完整 raw generation 交原官方 scorer，预测字段按其代码探测 |
| `mmsu` | 全5000条、三份原 parquet | 官方大小写题面；原 `mmsu_answer_text_contains_v2`，再执行固定官方 scorer |

MMSU 沿用原方案：生成字母不参与判分，归一化后包含完整 GT 文本才计对。
解释、否定或多个候选中出现 GT 也可能命中；没有切换为字母准确率。
空回答计错；任务失败不跳过题目、不缩小分母，不对未完整覆盖的结果输出成功分数。

## checkpoint 必须显式传入

没有默认训练 run、默认 checkpoint、latest 扫描或强制最终步数。
可评测任何已完整保存的中间 checkpoint，以及最终 checkpoint。

- 标准路径：`<run-root>/formal/checkpoints/model--step-N.ckpt`。
- 从标准路径推导外层 run root 和 `<run-root>/formal/runtime.json`，不会重复追加内层 formal。
- `N` 与 payload 的 `total_step` 必须一致。
- `model--epo-3.ckpt` 的 step 从 payload 获取，不根据文件名臆定。
- 非标准路径须同时给 `--runtime-config` 与 `--output-dir`。
- 训练 geometry 固定核对 `8,4,4`、GBS128、3epochs/34626 horizon；推理仍单卡 batch1。
- 校验 route、contract/runtime SHA、训练源码和共享源码、来源文本与 CED 资产、完整 state 张量库存/shape/dtype/finite。
- 两套 tied embedding/head 别名必须值一致；混合 DDP 命名空间拒绝。
- 评测不恢复 optimizer、scheduler、训练 RNG 或数据游标，也不使用训练 resume validator 阻止最终 checkpoint 评测。

## 提交单项评测

代码经 GitHub 同步到服务器后，在远程仓库根执行。将 `MODEL` 设为上表之一，`CKPT` 替换为该模型实际产物：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

MODEL=base
CKPT='/实际run路径/formal/checkpoints/model--step-5771.ckpt'

bash code/RSmol/run_text71427_ced_eval_5090.sh "$MODEL" reasonaqa5000 --checkpoint "$CKPT"
bash code/RSmol/run_text71427_ced_eval_5090.sh "$MODEL" mmau --checkpoint "$CKPT"
bash code/RSmol/run_text71427_ced_eval_5090.sh "$MODEL" mmar --checkpoint "$CKPT"
bash code/RSmol/run_text71427_ced_eval_5090.sh "$MODEL" mmsu --checkpoint "$CKPT"
```

四条命令各提交一个独立完整作业；也可用下面一条提交该模型的四项：

```bash
bash code/RSmol/run_text71427_ced_eval_5090.sh "$MODEL" all --checkpoint "$CKPT"
```

六模型的四项完整评测分别提交：

```bash
# 先将六个变量分别设为对应路线实际产生的完整 checkpoint 绝对路径。
BASE_CKPT='/实际base路径/formal/checkpoints/model--step-5771.ckpt'
LOOP_CKPT='/实际loop路径/formal/checkpoints/model--step-5771.ckpt'
LISTEN_CKPT='/实际listen路径/formal/checkpoints/model--step-5771.ckpt'
READ2_CKPT='/实际read2路径/formal/checkpoints/model--step-5771.ckpt'
WFUSE_CKPT='/实际wfuse路径/formal/checkpoints/model--step-5771.ckpt'
AR6_CKPT='/实际ar6路径/formal/checkpoints/model--step-5771.ckpt'

bash code/RSmol/run_text71427_ced_eval_5090.sh base all --checkpoint "$BASE_CKPT"
bash code/RSmol/run_text71427_ced_eval_5090.sh loop_lora all --checkpoint "$LOOP_CKPT"
bash code/RSmol/run_text71427_ced_eval_5090.sh listen all --checkpoint "$LISTEN_CKPT"
bash code/RSmol/run_text71427_ced_eval_5090.sh audio7_read2 all --checkpoint "$READ2_CKPT"
bash code/RSmol/run_text71427_ced_eval_5090.sh wfuse all --checkpoint "$WFUSE_CKPT"
bash code/RSmol/run_text71427_ced_eval_5090.sh audio7_read2_ar6 all --checkpoint "$AR6_CKPT"
```

示例5771仅说明文件格式，可换成实际要比较的任意已保存 step；跨模型比较时自行选择相同训练进度。
不需要重新生成已有的 ReasonAQA5000 manifest，也没有新增 smoke 前置。

默认沿用参考入口当前实际资源：ReasonAQA5000 为 `pdgpu-4090`，其余为 `pdgpu-5090`，均1GPU/8CPU/32G/1node。
使用原镜像与 `mellow_c8204d8`，强制离线；可显式 `--queue pdgpu-4090` 等覆盖队列。
`_5090.sh` 为兼容命名，不表示 ReasonAQA 的默认队列。
完整 job 名少于60个 ASCII 字符，包含 checkpoint/模型/任务/批次摘要。

先检查有效参数、输出和完整 vc 参数，不读资产、不提交：

```bash
bash code/RSmol/run_text71427_ced_eval_5090.sh "$MODEL" all --checkpoint "$CKPT" --check-args
```

单任务支持原数据路径覆盖，如 ReasonAQA `--manifest`/`--test-json`/三个音频根，
MMAU/MMAR `--dataset-dir`/`--metadata-json`/`--audio-root`/`--parquet`/`--evaluation-script`，
MMSU `--mmsu-root`/`--dataset-dir`/`--evaluation-script`；任务专属覆盖请按单任务提交。
`--parquet-batch-size` 只影响 I/O 批次，不改变推理 batch1。

## 批量清单

可准备一个 JSON list，每项显式填写 `model`、`checkpoint`，可选 `runtime_config`、`expected_step`：

```json
[
  {"model": "base", "checkpoint": "/实际base/formal/checkpoints/model--step-5771.ckpt"},
  {"model": "wfuse", "checkpoint": "/实际wfuse/formal/checkpoints/model--step-5771.ckpt"}
]
```

```bash
bash code/RSmol/run_text71427_ced_eval_5090.sh batch all --targets-json /实际路径/targets.json --check-args
bash code/RSmol/run_text71427_ced_eval_5090.sh batch all --targets-json /实际路径/targets.json
```

每条清单记录提交四项；某项文件缺失或 vc 失败仍继续其余任务，整体返回失败并保留每项状态。
`SUBMITTED` 仅表示提交成功，不代表推理或评分完成。
日志与 submission report 位于 `code/RSmol/log/text71427_ced_eval/`。

## 输出、续跑与汇总

默认输出：

```text
<run-root>/eval/text71427_ced_v1/<reasonaqa5000|mmau|mmar|mmsu>/<checkpoint文件名去掉.ckpt>/
```

每项保留 run_config、raw_generations、模型/42层执行审计、评分输入及 evaluation_report。
ReasonAQA 另有 accuracy/per_sample_results/input fingerprints；MMSU 保留原始及 normalized predictions/parsing audit。
重提相同命令、相同身份仅续跑剩余题目；ReasonAQA/MMSU 完整 raw 重评分不构造模型。
进程锁拒绝同时写同一输出；完整坏行、错误身份或非空无主目录拒绝，只有未完成末尾写按原 storage 恢复。
不覆盖旧评测目录。`--output-root DIR` 可让批量结果放到 `DIR/MODEL/TASK/CHECKPOINT_STEM`；单项可用 `--output-dir`。

显式列出同一任务的已完成输出，CPU 汇总为 CSV/JSON：

```bash
bash code/RSmol/text71427_ced_eval/run.sh compare --benchmark reasonaqa5000 \
  --report-dir /实际base评测目录 --report-dir /实际wfuse评测目录 \
  --output /实际新目录/reasonaqa_comparison.csv
```

其他任务将 `--benchmark` 换为 `mmau`、`mmar` 或 `mmsu`。
比较拒绝不完整记录、不同数据/协议/评分代码、错序和分母变化。
ReasonAQA 还核对每题输入指纹、重算分数，可对齐原固定5000评测结果与新结果；原/新复用的数据与评分源码摘要必须一致。
其他三项汇总核对原始记录和评分产物，不加载模型。

## 本地验收与同步

```powershell
python -B tests/validate_text71427_ced_eval.py
```

验收保护全部原有 tracked Python/Shell/JSON 源码的 normalized-LF SHA；检查关键函数 AST 与协议字典一致，
执行六模型×四任务的完整尺寸合成流程、续跑/坏行/身份拒绝、固定 MMSU scorer、实际 Bash→Python 参数链与模拟 vc。
批量合成流程的模型、音频和 Parquet 明确模拟，fsync 被加速模拟；不作为真实张量/GPU PASS。
可选真实小尺寸 CPU 测试检查六模型的训练后非零参数保存重载/logits、42层/hooks、LoRA和首答案对齐。
缺 Torch/Transformers/Safetensors 时明确 skip；`--require-tensors` 会将任何跳过判为失败，不作为正式评测前置。
报告为 `audit_reports/text71427_ced_eval_validation.json`。远程实际权重加载、显存及四项 GPU 评测待用户提交。

2026-11-20 本地最终验收：38项中32项执行通过、6项真实 CPU 模型测试缺依赖明确跳过，
`PASS_LOCAL_WITH_SKIPS`。3239份原有源码 normalized-LF SHA 保持；Python3.10语法、实际 Bash 参数链、
六模型×四项完整尺寸合成流程、原函数 AST/协议、严格加载顺序、续跑/错误拒绝及固定 MMSU scorer通过。
Windows 沙箱限制临时目录与 Git index.lock 后，获准在沙箱外完成离线测试及 Git dry-run 清单检查。
没有访问远程生产资产、提交真实 GPU 作业或代为 commit/push。

只暂存本次文件，RM 按块选择新增记录，保留已有修改：

```powershell
git add -- code/RSmol/text71427_ced_eval code/RSmol/run_text71427_ced_eval_5090.sh tests/test_text71427_ced_eval.py tests/test_text71427_ced_eval_loading.py tests/test_text71427_ced_eval_tensors.py tests/test_text71427_ced_eval_workflow.py tests/validate_text71427_ced_eval.py audit_reports/text71427_ced_eval_validation.json
git add -p -- RM.md
git diff --cached --stat
git commit -m "Add isolated four-benchmark evaluation for six text71427 CED routes"
git push
```
