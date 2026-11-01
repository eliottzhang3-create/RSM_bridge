# 五条 CED 多层交互路线的 MMAU test-mini / MMAR 评测

两个既有提交入口 `run_mizar_s1_ced_small_mmau_eval_5090.sh`、
`run_mizar_s1_ced_small_mmar_eval_5090.sh` 新增以下五个
`RSMOL_MIZAR_CED_EVAL_VARIANT`。新增实现位于训练目录外：

```text
scripts/evaluate_mizar_s1_ced_multilayer_benchmark.py
scripts/mizar_s1_ced_multilayer_eval_runtime.py
scripts/evaluate_mizar_s1_ced_multilayer_benchmark.sh
```

训练源码、共享递归实现和原有普通 CED / MeSH / 四条非 MeSH evaluator 保留。
默认不指定 variant 时仍为 `base`。每条新路线直接导入自己的实际训练模型，
完整严格加载音频训练 checkpoint；不重新实现模型交互，也不修改训练 forward。

## 路线与默认产物

| variant | contract 版本 | 新增分支参数 | 模型上下文与行为 |
|---|---|---:|---|
| `recursive_5_10_5_layer36912_crossattn` | v2 | 4,892,741 | 四层原始 CED grids + 固定 prompt mask；逻辑索引5、15执行文本 cross attention，保留固定γ=0.3 |
| `recursive_5_10_5_layer36912_query32_crossattn` | v1 | 6,241,733 | 四层 grids 各压缩为32个摘要后交互；共享 learned queries / compression attention，保留固定γ=0.3 |
| `recursive_mesh_layer4_8_inject` | v1 | 2,444,930 | 原顺序 `(A4,A8)`；在原 memory read₀、read₁之后分别注入音频位置 |
| `recursive_mesh_layer4_8_listen` | v1 | 2,450,700 | 原顺序 `(A4,A8)`；前两次边界执行原 write → 新 listen → 原 read，保留两个新增 routers |
| `recursive_mesh_layer36912_listen` | v2 | 3,552,788 | 四层 grids + prompt mask；四路 sigmoid 聚合、逐音频 token tanh 幅度 gate、两次 memory listen 写入，无固定γ或/4缩放 |

query32 路线 README 的母本遗留描述不作为架构身份：实际 `contract.py` 是
`ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer36912_query32_crossattn_v1`。
其32个摘要只用于交互分支，输入 prefix 保持383。

五条训练目录分别是训练根
`mellow_official_training_c8204d8_adamw_cosine/` 下的：

```text
mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_crossattn
mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_query32_crossattn
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_listen
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer36912_listen
```

所有远程 run 均位于
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/` 下。
默认 checkpoint 为各 run 的 `formal/checkpoints/model--step-17313.ckpt`，
runtime 为同 run 的 `formal/runtime.json`：

| variant | 相对输出根的默认 run |
|---|---|
| `recursive_5_10_5_layer36912_crossattn` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer36912_crossattn_5090/formal_20261030_140459065784533` |
| `recursive_5_10_5_layer36912_query32_crossattn` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer36912_query32_crossattn_5090/formal_20261030_162028139130855` |
| `recursive_mesh_layer4_8_listen` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_listen_5090/formal_20261030_172450546025072` |
| `recursive_mesh_layer4_8_inject` | 用户尚未提供已完成 run，必须显式指定实际产物 |
| `recursive_mesh_layer36912_listen` | 用户尚未提供已完成 run，必须显式指定实际产物 |

最后两条不猜测训练时间戳，也不自动选择 latest。未指定 run 或完整路径时，
提交脚本在调用 `vc submit` 前报错。

## 与原 CED 相同的推理和评分协议

| 项目 | 五条路线的统一设置 |
|---|---|
| 数据 | MMAU test-mini / MMAR 各完整1000条；原 metadata、音频查找逻辑、原数据和选项顺序 |
| 音频 | mono、32 kHz；fixed20 中心裁剪/右补零至20秒，再分成两个10秒半段；重拼20秒后才进入 CED |
| Prompt | 原固定选项模板、整体小写、训练 tokenize/EOS 后缀；截断/右补 `!` 到256 tokens |
| Prefix | `126 audio + 1 SEP + 256 prompt = 383`，第一个答案 logit 在382 |
| 数值 | 单GPU、batch1、FP32、全部 eval、inference mode；seed1234、TF32关闭及原 cuDNN 设置 |
| 解码 | `attention_mask=None`、`labels=None`、`use_cache=False`，逐 token 完整重算，纯 argmax |
| 停止 | 两个 benchmark 都最多300个新 token，首次生成 EOS 后停止 |
| 文本 | `skip_special_tokens=False`、`clean_up_tokenization_spaces=False`；EOS、大小写、空白原样保存，不 strip 或提取选项字母 |
| 评分 | 原官方 scorer 与预测字段；必须1000条 generated、零 skip，且 scorer 成功确认1000条分母 |
| 资源 | `pdgpu-5090 / 8 CPU / 32G / 1 GPU / 1 node`，原容器及 `mellow_c8204d8` |

协议字典直接复制普通 CED 的 `PROTOCOL`。runner 的15个公共函数与原评测保持 AST 一致，
差别集中在参数解析、模型上下文传递及输出身份。
prompt mask 只用于新增交互分支的有效 prompt 池化，不传给 LM 的 causal attention，
也不屏蔽原来可见的 `!` padding。解码期间 mask 固定为256，不包含新增答案 token。

每个样本只进行一次 CED 编码；原始特征作为局部上下文显式传给每一步 LM。
LM 内的投影、query32 压缩和交互仍按训练实现每次完整 forward 执行。
MeSH memory 由原训练 forward 每次重新创建，不跨 token 完整 forward 或样本持久化。
四层 MeSH listen 的 prefix API 返回 `(prefix, output, None)`；评测侧从
`output["layer_grids"]` 和本次 tokenize 的 `attention_mask` 组装上下文。

## 严格加载与执行检查

加载前检查 formal checkpoint 的 schema、route、合同/SHA、同 run 的 runtime、
训练源码 SHA、递归模型源码 SHA、资产、step17313、3 epochs、loss 标识、
训练 geometry `8,8,4` 和分支合同。已完成 checkpoint 可用于评测，
不套用会拒绝完成状态的训练 resume 校验，也不恢复 optimizer/scheduler。

用对应路线构建真实训练模型，转 FP32 后 `strict=True` 加载完整 `state_dict`，
包括 CED、原 mapper、LM 和所有 learned query/投影/MLP/gate/router。
仅统一规范化全体键共有的 DDP `module.` 前缀；混合前缀、缺键和多余键都拒绝。
加载后调用实际训练结构 validator，再冻结所有参数并检查全部 eval/FP32。
tokenizer 沿用原训练检查，不增长词表、不 resize embedding。

每次 LM forward 的只读 hooks 检查：

- 20个独立物理层按 `0..14,5..14,15..19` 执行30次，无 KV cache。
- 两次分支在逻辑索引5、15，4/8路线的特征顺序为4→8。
- MeSH 六个原 routers 和两个额外 listen routers 独立且按原模型顺序调用。
- query32 每次 LM forward 一次压缩，输入 query `[4,32,576]`、音频 bank `[4,126,576]`。

router hooks 观察的是**权重计算调用顺序**：原 write router → 原 read router → 新 listen router。
这与 memory 操作的原 write → 新 listen → 原 read 一致，read 权重在实际 read 之前已经算好。
真实张量测试另外观察实际 `_write` / `_read` 和分支执行顺序。
hooks 不返回替代输出，不修改隐藏状态或 logits；训练数值诊断模式保持关闭。
源码 SHA 不符时应恢复匹配训练时的源码，不能关闭校验加载。
模型构建仍需要原预训练 CED 和文本资产位于合同指定的远程位置。

## 三份现有 checkpoint：六项 full 评测

通过 GitHub 同步代码后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer36912_crossattn bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer36912_crossattn bash run_mizar_s1_ced_small_mmar_eval_5090.sh

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer36912_query32_crossattn bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer36912_query32_crossattn bash run_mizar_s1_ced_small_mmar_eval_5090.sh

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer4_8_listen bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer4_8_listen bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

每条命令提交一个单卡 full 评测作业，无 smoke 或 tensor 测试自动前置任务。
compute wrapper 激活 `mellow_c8204d8` 并启用 HF 离线模式，不安装或升级依赖。
提交时打印 variant、checkpoint、runtime 和输出路径。

## 另外两条训练完成后

将下面两个变量替换为**实际已经产出 checkpoint 的 run 根目录**：

```bash
inject_run=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_5090/formal_REPLACE_WITH_ACTUAL_RUN
listen_run=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer36912_listen_5090/formal_REPLACE_WITH_ACTUAL_RUN

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer4_8_inject RSMOL_MIZAR_CED_RUN_ROOT="$inject_run" bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer4_8_inject RSMOL_MIZAR_CED_RUN_ROOT="$inject_run" bash run_mizar_s1_ced_small_mmar_eval_5090.sh

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer36912_listen RSMOL_MIZAR_CED_RUN_ROOT="$listen_run" bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer36912_listen RSMOL_MIZAR_CED_RUN_ROOT="$listen_run" bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

也可用 `--run-root "$inject_run"` 等 CLI 参数。标准
`<run>/formal/checkpoints/<file>` checkpoint 路径可推导其 run；对五条新路线，
显式 run 优先，其次是显式标准 checkpoint 所属 run，最后才使用已知默认 run。
非标准 checkpoint 路径应同时提供 `--checkpoint`、`--runtime-config`、`--output-dir`。
新路径解析兼容 `--option value` 和 `--option=value`。

保留原有环境覆盖接口：`RSMOL_MIZAR_CED_RUN_ROOT`、`RSMOL_MIZAR_CED_CHECKPOINT`、
`RSMOL_MIZAR_CED_RUNTIME_CONFIG`、`RSMOL_MIZAR_CED_ROUTE_ROOT`、
`RSMOL_MIZAR_CED_EXPECTED_STEP`、`RSMOL_MIZAR_CED_EXPECTED_GEOMETRY`、
`RSMOL_MMAU_DATASET_DIR`、`RSMOL_MMAR_DATASET_DIR`、
`RSMOL_MIZAR_CED_MMAU_OUTPUT_DIR`、`RSMOL_MIZAR_CED_MMAR_OUTPUT_DIR`。
额外 CLI 参数继续传给 runner；默认评测 step17313，训练 geometry8,8,4。

## 结果与续跑

结果默认写入各训练 run 自己的：

```text
eval/mmau_test_mini_step17313/
eval/mmar_step17313/
```

各项保存 `run_config.json`、`progress.jsonl`、`raw_generations.jsonl`、
`predictions_fixed_order.json`（MMAU）或 `predictions_official.json`（MMAR）、
`official_evaluation.txt`、`evaluation_report.json`。
JOB.log 位于远程仓库 `code/RSmol/log/`，按 variant 和 benchmark 区分。

`run_config.json` 绑定 checkpoint 完整 SHA、runtime、variant/route、数据、scorer、
评测实现源码和原协议。相同身份重新提交会从已落盘记录续跑，不重新推理完成的样本。
身份不同、源码变化或非空无身份目录会拒绝写入，应指定新的独立输出目录。
同一输出目录只运行一个作业。

成功需 `evaluation_report.json` 的 `status=PASS`、1000条 generated / 零 skip，
且官方 scorer 成功确认1000条分母。结构和执行检查在
`model_audit.recursive_audit`、`model_audit.recursive_forward_audit`。
完全复用1000条已有结果时没有新 forward，后者可以是 PENDING / 0。
本地代码验证报告的 PASS 不代表远程权重或 benchmark 评分已经通过。

## 验证与边界

在仓库根目录运行：

```bash
python -B tests/validate_mizar_s1_ced_multilayer_eval.py
```

汇总报告是 `audit_reports/mizar_s1_ced_multilayer_eval_validation.json`。
2026-10-31本地结果：93项实际检查通过，18项真实 CPU tensor 检查因缺少
Torch/Transformers跳过（本次新增9项、旧回归9项），Python/Bash语法通过。
涵盖五路线真实合同识别、交叉路线拒绝、strict 加载接口、原协议/公共函数 AST 对照、
显式上下文、NumPy 解码检查、十种1000条模拟流程、续跑、官方 scorer 成败分母、
十种提交组合/路径覆盖及旧 evaluator 回归。

另有9项真实 CPU PyTorch/Transformers 测试，使用实际训练模型及非零 gates，
检查 adapter/direct logits 与 tokens、训练683/评测383首答案对齐、strict restore、
跨样本隔离、query32 压缩和 MeSH 实际操作顺序。多数使用缩小 LM 宽度/词表，
保留20物理层、30执行和383 prefix；结构检查另外使用生产 hidden576。
这些测试不用远程权重、CUDA、下载或作业提交，不能替代真实 checkpoint 的 GPU 验证。

具备原有 tensor 依赖的环境可手动强制所有 tensor 测试实际执行：

```bash
python -B tests/validate_mizar_s1_ced_multilayer_eval.py --require-tensors
```

缺依赖或任一 tensor skip 会使该模式失败。本机缺 Torch/Transformers 时明确记录跳过，
不得把 stand-in/NumPy 检查写成真实 Torch 检查通过。
三份远程 checkpoint 内部合同、配套 runtime 和六项 full 评测尚待服务器实际核验。
