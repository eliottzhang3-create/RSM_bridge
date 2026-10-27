# CED-Small + MeSH checkpoint 的 MMAU / MMAR 评测

这条路线评测音频 S1 训练完成后的完整 MeSH checkpoint。保留普通 CED 的 Python runner、adapter 和运行脚本，使用隔离的 `evaluate_mizar_s1_ced_recursive_mesh_benchmark.py`、`mizar_s1_ced_recursive_mesh_eval_runtime.py` 和同名 Bash wrapper。两个现有 launcher 通过 `RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh` 选择它们；未设置 variant 或设为 `base` 时继续使用原普通 CED 评测。

## 目标产物与加载

默认 MeSH run root：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_5090/formal_20261027_081123150934256
```

配对读取该 run 的 `formal/checkpoints/model--step-17313.ckpt` 与 `formal/runtime.json`，使用训练目录 `mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small` 构建模型。虽然模型构建会读取训练时的本地预训练资产，最终推理的 CED、mapper、文本层、embedding/head 和全部 routers 均由完整音频 checkpoint 的 `state_dict` 严格覆盖；不会恢复训练 optimizer/scheduler/RNG。

启动核验：

- checkpoint schema、route、步数、训练 batch 几何及其与 runtime 的完整合同/SHA 一致。
- 训练源码、共享 MeSH 源码、本地 CED 与文本预训练资产仍与训练时记录一致；训练目录和共享 MeSH 实现未为评测而修改。
- 模型确为 hidden576、20个独立物理层、30次逻辑执行、两次中间循环、5个临时 memory slots，以及3个 write / 3个 read routers。六个 routers 的12个参数 tensor 独立，维度为576→5，不包含持久化 memory。
- tokenizer 沿用训练路线校验：EOS/SEP=0，注册已有 `!` 为PAD，不增加词表或调整 embedding 大小。最终模型全部 FP32、eval，参数不求梯度。
- 每次生成的真实 LM forward 后只读检查 `0..14,5..14,15..19` 调度与无 KV cache；不打开 tensor audit/routing stats 模式，不替换 logits 或输出。

任何身份、结构或权重加载不匹配均停止，不自动回退至普通 SmolLM2。

## 与普通 CED 相同的评测协议

| 项目 | 固定设置 |
|---|---|
| 数据 | MMAU test-mini 和 MMAR 各完整1000条；原官方文件、顺序及选项顺序 |
| Prompt | 原固定选项模板、小写、训练 tokenizer、EOS 后缀、右侧 `!` padding/truncation，固定256 tokens |
| 音频 | 原 fixed20、在线 CED small、逐样本 dB 与离线配置 loader、官方 `freq_merge_126` mapper |
| 布局 | 126 audio + 1 SEP + 256 prompt = prefix383；首个 answer logit 位于382 |
| 数值 | FP32、全部eval、TF32关闭；沿用原 seed 与确定性设置 |
| 生成 | `attention_mask=None`、`labels=None`、`use_cache=False`；argmax；逐token完整序列重算；最多300个新token，首次EOS停止 |
| Memory | 原 MeSH 每次完整前向创建新的五槽临时 memory，不跨生成步复用 |
| 文本 | `skip_special_tokens=False`、`clean_up_tokenization_spaces=False`；保留原始生成和EOS，不strip、不提取选项标签 |
| 评分 | 原官方 scorer、原预测字段与完整1000条覆盖要求 |
| 资源 | pdgpu-5090、8 CPU、32G、1 GPU、1 node；原容器与 `mellow_c8204d8` |

新 runner 的18个函数与普通 CED runner AST 完全一致，adapter 的 `greedy_decode` 与生成协议字典一致。MeSH 和普通 CED 输出目录、日志及评测源码 SHA 身份独立；保留原输出目录认领与 ProgressStore 续跑逻辑。

## 正式提交

通过 GitHub 将本地修改同步到服务器后，在远程登录节点执行两次提交：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh \
  bash run_mizar_s1_ced_small_mmau_eval_5090.sh

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh \
  bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

每条命令提交一个完整评测作业。compute wrapper 自动激活 `mellow_c8204d8` 并设置离线模式；不需要安装或升级依赖，也没有新增 smoke 前置作业。

默认结果位于上述 MeSH run root 的：

```text
eval/mmau_test_mini_step17313/
eval/mmar_step17313/
```

两项评测分别保存 `raw_generations.jsonl`、`predictions_fixed_order.json` / `predictions_official.json`、`run_config.json`、`evaluation_report.json` 和 `official_evaluation.txt`。成功标准为 `evaluation_report.json` 中 `status=PASS`、完整1000条生成且无跳过，并且官方评分器确认1000条结果；真实结构与循环记录位于 `model_audit.recursive_audit`、`model_audit.recursive_forward_audit`。

## 路径覆盖与恢复

保留原参数接口：`RSMOL_MIZAR_CED_RUN_ROOT`、`RSMOL_MIZAR_CED_CHECKPOINT`、`RSMOL_MIZAR_CED_RUNTIME_CONFIG`、`RSMOL_MIZAR_CED_ROUTE_ROOT`、`RSMOL_MIZAR_CED_EXPECTED_STEP`、`RSMOL_MIZAR_CED_EXPECTED_GEOMETRY`，以及各 benchmark 的数据集、输出目录变量和额外 CLI 参数。仅覆盖 run root 时，checkpoint、runtime 和结果目录同步由该 root 推导。

同一 checkpoint、数据、协议、评测代码与输出目录下，重新执行原提交命令即可接续已经落盘的记录；不同 checkpoint 或代码身份不能混用已有结果目录。指定其他 MeSH run 示例：

```bash
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh \
RSMOL_MIZAR_CED_RUN_ROOT=/path/to/another_mesh_run \
  bash run_mizar_s1_ced_small_mmau_eval_5090.sh
```

`recursive_mesh` 必须显式指定；仅更换 checkpoint 路径不会自动切换模型路线。未知 variant 会在提交前拒绝。

## 验证范围

本地新增25项依赖轻量检查，并通过原CED评测12项回归，共37项。覆盖真实 Bash launcher 的默认/分流/覆盖/引号传参/资源/环境/退出码，以及严格 state_dict 调用、DDP key 前缀、router别名、错误身份和结构拒绝、逐token调度检查、EOS/300-token上限、原始文本、输出身份与续跑。Python和Bash语法检查通过。

本地没有Torch/Transformers、远程权重或数据，因此真实 GPU 加载、1000条生成及官方评分尚未验证。另提供两项真实小模型CPU检查，验证 trace hook 保持 logits/参数不变及每次前向重新初始化五槽 memory；本地跳过，可在远程单独运行（不加载大权重，不是正式评测的前置门禁）：

```bash
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate mellow_c8204d8
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
python -B tests/test_mizar_s1_ced_recursive_mesh_eval_tensor.py -v
```
