# 原五槽 MeSH 文本生成与音频七槽 router 的 K 标定：第一阶段

本工具只使用原文本 checkpoint，对用户填写的十条文本 prompt 做生成并记录六个原 router。
不加载 CED、音频 mapper 或音频 checkpoint，不注册七槽模块，不更新模型权重。
原训练入口、训练代码、共享 MeSH 实现均保持原状。本地代码通过 GitHub 同步到远程；
真实 checkpoint 和推理位于远程 Linux。

## 固定推理设置

- checkpoint 与 `run_mizar_s1_gbs256_recursive_mesh_ced_small_5090.sh`、
  `run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_listen_5090.sh` 的默认文本来源一致：
  `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244`。
- 严格加载原类和全部权重；检查 hidden576、20物理层、30逻辑执行、两循环、五槽、六个独立 router。
- FP32、eval、inference mode、关闭 autocast/TF32，batch1，seed1234。
- 纯 argmax、无 top-p 或 temperature 过滤；`use_cache=False`、每 token 完整重算。
- `max_new_tokens=100` 固定；遇到 checkpoint EOS 提前停止，因此不是强行生成满100个 token。
- 直接使用原 tokenizer，`add_special_tokens=True`，不添加聊天模板、不修改大小写、
  不 strip、不补齐 prompt、不截断。prompt 与100个生成位置必须一起放入1024上下文，
  且不超过 checkpoint 的 `max_position_embeddings`；超长报错。
- 每次实际前向用只读 hooks 核对 `0..14,5..14,15..19` 物理轨迹和六个 router 调用。

## 创建并填写十条 prompt

以下操作在 GitHub 同步后的远程登录节点执行。创建模板和检查输入仅需 Python 标准库，
不会使用 GPU 或加载 torch。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol

PROMPTS=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mesh_text_router_k_5090/inputs/prompts.jsonl
python -B scripts/calibrate_mesh_text_router_k.py --write-prompts-template "$PROMPTS"
nano "$PROMPTS"
```

模板自动创建 p01～p10 共十行，初始 `prompt` 为空。请填写所有十行，保留唯一的字符串 `id`。
文件为 UTF-8 JSONL，每行一个完整 JSON 对象，字段必须是 `id`、`prompt`，例如：

```jsonl
{"id": "p01", "prompt": "Your first prompt goes here."}
{"id": "p02", "prompt": "Your second prompt.\nThis is the same prompt on a new line."}
```

上述只展示两行格式，实际必须有十条非空 prompt。prompt 中的换行写成 `\n`，
双引号写成 `\"`，不要写成跨物理行的 JSON 对象。支持中文、Windows CRLF 和 UTF-8 BOM。
空行会忽略，非空记录必须正好十条；重复 id、空 prompt、拼错字段、重复 JSON key 都报错。
模板创建拒绝覆盖已有文件；第二次运行实验可直接编辑原文件，或创建另一个模板路径。
prompt 是原始续写输入，工具不会自行包装成 instruction/chat 格式。

## 提交单卡推理

填完十条后，在远程 `code/RSmol/` 执行：

```bash
PROMPTS=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mesh_text_router_k_5090/inputs/prompts.jsonl
OUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mesh_text_router_k_5090/text10_cap005_$(date +%Y%m%d_%H%M%S)

bash run_mesh_text_router_k_5090.sh \
  --prompts-jsonl "$PROMPTS" \
  --output-dir "$OUT" \
  --max-new-tokens 100
```

资源为 `pdgpu-5090 / 8 CPU / 32G / 1 GPU / 1 node`，原有容器及 `mellow_c8204d8` 环境。
只提交这一个推理任务，没有 smoke 或 tensor 验证前置任务。提交前有标准库输入/config检查；
tokenizer 实际长度和真实权重加载在计算节点验证。
默认文本 checkpoint 已固化，也可用 `--text-model-dir /absolute/checkpoint-dir` 显式覆盖。

省略 `--output-dir` 时自动使用：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mesh_text_router_k_5090/run_<timestamp>/
```

每次提交打印完整输出与日志路径；日志位于同一根目录的 `logs/`。
输出目录必须是新目录或空目录。失败时保留已完成样本和 FAIL 报告，重跑使用新输出目录，
本次十样本工具不提供 resume。输入 JSONL 保存在外部 `inputs/`，不会进入代码仓库。

## 记录内容

| 文件 | 内容 |
|---|---|
| `prompts.jsonl` | 本次十条输入的规范副本 |
| `generations.jsonl` | 完整 prompt/token IDs、continuation/token IDs、原始含特殊 token 的输出、EOS/长度停止原因 |
| `generations.md` | 便于人工阅读的十条输入和原始输出 |
| `router_records.csv` | 每个唯一 token 位置 × 六个 router 的五个 logits、五槽概率、logsumexp、token/id/区域/生成步 |
| `runtime_audit.json` | 软件/设备、strict loading、参数结构、每次实际层轨迹检查、router 参数前后 hash |
| `run_config.json` | 完整参数、输入/hash、checkpoint/tokenizer 文件 hash、源码 hash、Git commit 和运行状态 |
| `calibration_report.json` | 共用 K、每 router 候选边界与概率统计、最坏 token、prompt/生成段独立统计 |
| `calibration_report.md` | K 和六个 router 的均值/P95/最大概率表 |
| `generation_report.json` | 完整性状态；只有 `status=COMPLETE` 才表示十样本阶段结束 |

六个 router 名称依次为 `write_pre/read_pre/write_0/read_0/write_1/read_1`。
记录原模型实际使用的查询：prelude 输出，以及两轮循环各自的输入；不把循环输出当成后两组查询。

prefill 一次记录全部 prompt 位置，之后每次只保存新加入的那个生成 token 位置，
防止无 KV cache 的重复前向使旧 token 重复计权。最后一个生成 token（包括提前生成的 EOS）
还没有作为输入出现，因此额外运行一次只读完整前向来记录它，**不再选择新 token**。
生成上限仍是100，最多101次前向。token位置是0-based；最后一次的 `forward_kind`
为 `post_generation_observation`。CSV中的 `generation_step` 是0-based前向步号，
该最后观察步等于已经生成的 token 数。CSV中的 token 可能含换行/逗号，读取时使用标准 CSV parser。

## 保守 K 的定义

两个新增槽均假设 `weight=0`、`bias=-K`。对某个原 router 的五个 logits 定义：

```text
L = logsumexp(z1,...,z5)
q(K) = P(new_slot_1) + P(new_slot_2) = sigmoid(log(2) - K - L)
```

默认 `--new-slot-mass-cap 0.005`：**两个新增槽合计低于0.5%，不是每个槽0.5%**。
它是对所有记录位置的最大值约束，不是均值目标，不要求六个 router 的概率一样。

```text
每个位置在 q=cap 时的边界 = log(2*(1-cap)/cap) - L
共同边界 = 六个router、十样本全部记录位置中上述边界的最大值
推荐K = max(0.1, ceil((共同边界 + 1.0) * 10) / 10)
```

`--k-margin` 默认1.0，为额外 logit 余量；随后向上取整至0.1。
因此记录位置的最大 q 通常至多约0.185%，比0.5%阈值更保守。
初始化两槽概率相同，各为q的一半；原五槽相对概率保留，总概率变为1-q。
报告同时提供合计1%与0.5%对应的共用/各router理论边界，便于后续判断。
只保存原五槽概率不足以重算K，必须使用原logits或L，因此三者都保留。

这是**原文本输入分布上的标定参考**。音频输入可能改变绝对logits和后续循环输入，
增加margin并不能证明音频位置的上界。后续音频七槽实现完成后仍需做真实初始化前向核对。

## 查看结果与离线调整

任务结束后，使用提交时打印的实际输出目录：

```bash
cat "$OUT/generation_report.json"
cat "$OUT/calibration_report.md"
```

需要换成更小的阈值时，标准库即可从CSV重算，不重新生成、不用GPU：

```bash
python -B scripts/calibrate_mesh_text_router_k.py \
  --records-csv "$OUT/router_records.csv" \
  --new-slot-mass-cap 0.001 \
  --k-margin 1.0 \
  --output-dir "${OUT}_recalibrate_cap001"
```

## 本地验证与远程范围

仓库根目录的统一验证入口：

```bash
python -B tests/validate_mesh_text_router_k.py
```

包含显式七槽softmax/公式比较、最坏token上界、极端logits数值、输入和CSV拒绝条件、
离线重算、真实Bash的模拟提交/参数转义/环境/退出码检查，以及Python/Bash语法。
另提供5项真实小型CPU MeSH tensor检查：hook无扰动、100token上限、EOS与末token观察、
上下文/EOS错误拒绝、扩展Linear的真实softmax数值。

本机没有torch/transformers，因此真实tensor检查跳过；远程可在已有依赖的环境手动执行：

```bash
python -B tests/validate_mesh_text_router_k.py --require-tensors
```

此命令不下载、不使用CUDA、不提交任务，也不自动作为上述推理任务的前置要求。
本地状态只表示代码检查完成，真实权重加载、十条生成和K数值待远程执行。
