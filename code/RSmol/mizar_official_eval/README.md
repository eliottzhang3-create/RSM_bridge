# 官方 Mizar 权重：沿用项目 CED MMAU/MMAR 协议

本目录为独立评测适配，加载官方 **Mizar-159M Stage 3 / Q4 / seed20260905 / update200**。
本地只编辑代码，通过 GitHub 同步；模型、数据、GPU 作业均留在远程 Linux。
原训练代码、原 evaluator 和所有旧 variant 均保留；两个旧提交脚本各只新增一个
`official_mizar` 分支，默认仍为 `base`。

## 远程提交

本地改动提交并推送 GitHub 后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol

RSMOL_MIZAR_CED_EVAL_VARIANT=official_mizar bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=official_mizar bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

两次提交分别完整处理 MMAU test-mini 1000 条、MMAR 1000 条。
使用 `vc submit`、`pdgpu-4090`、原镜像、8 CPU / 32G / 1 GPU / 1 node，
计算节点激活 `mellow_c8204d8`。不增加自动 smoke、tensor 测试或单独 GPU 前置任务。
保留原 `_5090.sh` 入口文件名，`official_mizar` 分支实际提交到 `pdgpu-4090`。
加载时校验文件身份及模型结构；正式生成过程中只读 hooks 检查每次执行 30 个独立层。

默认模型目录已经对应上传位置：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M
```

必须保留以下文件（目录中其他文件可继续保留）：

```text
Mizar159M/
  checkpoints/final/stage3_q4_seed20260905.ckpt
  assets/smollm2-135m/
    config.json
    generation_config.json
    merges.txt
    model.safetensors
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
  assets/ced-small/
    README.md
    config.json
    configuration_ced.py
    feature_extraction_ced.py
    model.safetensors
    modeling_ced.py
    preprocessor_config.json
  model_info.json
  provenance/local_bundle.json
```

18 个文件的大小和 SHA256 固定在本目录 `release_manifest.json`。
checkpoint 为 637470191 bytes，SHA256 为
`0c542f63e11ed55d360feb0d66e7be0e8c6a878f44cffd302a92388f915388a2`。
训练 manifest、训练数据和官方仓库完整源码不参与模型加载，不需要额外上传。
CED 配置沿用项目现有离线 CSV 兼容处理，不联网下载标签表。

## 加载方式与协议一致性

`runtime.py` 直接导入现有非递归 CED 路线的标准模型，构造 CED-Small、
frequency-merging mapper 和 **30 个独立层**的 SmolLM2，然后使用
`torch.load(..., map_location="cpu", weights_only=True)` 读取官方完整参数，
`load_state_dict(..., strict=True)` 全量覆盖。
该文件是 plain state_dict，不读取项目训练 `runtime.json`，不假设 step17313，
也不接受其他训练 checkpoint、DDP 包装或只加载 mapper。
校验 454 个 tensor entries、187628509 个元素（包含 tied alias）、
159299840 个独立模型参数和 tied 49152×576 embeddings。
模型从加载到推理均使用 FP32，全模块 eval、关闭梯度，TF32 关闭。

`evaluate.py` 直接复用原 evaluator 的函数对象，包括 `generate`、音频解码、
metadata/scorer audit、预测写入与官方 scorer；生成循环仍为原 `greedy_decode`。
完整遍历、断点续跑、失败处理和覆盖率检查也沿用原控制流程，
仅把已加载模型的身份检查改为官方 release。

| 设置 | 本次与原 CED evaluator 一致 |
| --- | --- |
| 音频 | mono 32kHz，长音频中心裁 20s，短音频右补零至 20s；切成 10s 两半后在 CED 前重拼，只编码一次 |
| prompt | 官方选项原顺序、小写、追加 ` <\|endoftext\|>`；右截断/补 `!` 至 256 tokens |
| prefix | 音频126 + SEP0一个 + prompt256 = 383；首个答案 logit 位于382 |
| padding | PAD=`!`/17，EOS=SEP=0；prompt PAD 保持可见，attention_mask=None |
| 推理 | batch1、纯 argmax、每步完整 forward、use_cache=False；遇首个 EOS 或300新 tokens 停止 |
| 输出 | skip_special_tokens=False、clean_up_tokenization_spaces=False；保留原始 EOS、空格，不 strip、不抽取答案字母 |
| 数据/评分 | 原数据路径及官方版本审计；两个 benchmark 各1000条、零跳过；只运行对应官方 evaluation.py，要求评分分母1000 |
| 随机/数值设置 | seed1234、torch线程1、FP32、TF32关闭，与原 runtime 一致 |

这是“**官方最终 Mizar，在本项目相同评测协议下**”的对照结果。
官方原生推理采用前20秒、不同 question EOS 处理、vLLM 及特殊 token 清理；
本次按用户要求使用项目原协议。官方 final 已完成 Stage 3，项目参照训练为 Stage 1；
MMAU test-mini 也不同于论文9k评测和五seed统计，不能把这里结果标成论文分数复现。
官方来源：[Mizar_159M](https://github.com/kaiyangli-ml/Mizar_159M)，
模型版本和两类 mapper 的官方 AST 摘要记录在 release manifest。

## 输出、恢复与路径覆盖

默认输出根：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_official_stage3_q4_seed20260905_matched_ced_v1/
  mmau_test_mini/
  mmar/
```

每个目录保存 `run_config.json`、`raw_generations.jsonl`、
`evaluation_report.json`、`official_evaluation.txt`，以及
MMAU 的 `predictions_fixed_order.json` 或 MMAR 的 `predictions_official.json`。
`evaluation_report.json` 同时记录 release、严格加载、实际层执行审计、覆盖率和官方分数。
`log/mizar_official_{mmau,mmar}_4090.<timestamp>.JOB.log` 保存作业日志。

中断后重新执行对应提交命令，使用同一输出目录按 JSONL 恢复已完成样本。
输出身份包含权重/资产、模型源码、评测代码、数据、scorer 和完整生成协议；
任何身份变化都拒绝接续。需要重新完整评测时使用新的 `--output-dir`。

例如移动模型目录或指定全新输出：

```bash
RSMOL_MIZAR_CED_EVAL_VARIANT=official_mizar bash run_mizar_s1_ced_small_mmau_eval_5090.sh \
  --model-dir /another/path/Mizar159M \
  --output-dir /another/output/mmau_official
```

支持 `--checkpoint` 指定同一官方 checkpoint 的其他存放位置；仍必须匹配固定 SHA256。
不传该选项时，checkpoint 从最终生效的 `--model-dir` 自动推导。
数据相关选项与原 evaluator 相同：`--dataset-dir`、`--metadata-json`、
`--parquet`、`--audio-root`、`--evaluation-script`、`--parquet-batch-size`。
CLI 参数优先于 wrapper 默认值。

环境变量为 `RSMOL_MIZAR_OFFICIAL_MODEL_DIR`、`RSMOL_MIZAR_OFFICIAL_CHECKPOINT`、
`RSMOL_MIZAR_OFFICIAL_MMAU_OUTPUT_DIR`、`RSMOL_MIZAR_OFFICIAL_MMAR_OUTPUT_DIR`。
数据目录沿用 `RSMOL_MMAU_DATASET_DIR` / `RSMOL_MMAR_DATASET_DIR`。
旧训练版本的 `RSMOL_MIZAR_CED_CHECKPOINT`、runtime、run-root、step/geometry 设置不作用于本分支。

## 本地验证

仓库根目录手动执行：

```bash
python -B tests/validate_mizar_official_eval.py
```

该入口包含官方 release/strict loader/协议复用/两类1000条模拟流程/断点恢复/
Bash假提交及原有全部 CED evaluator 回归；不下载资产、不提交作业、不调用CUDA。
默认缺 tensor 依赖时明确 skip；`--require-tensors` 将缺依赖或 tensor skip 判为失败。
可选 `--model-dir <已下载目录>` 额外只读核验本地18个实际文件的 SHA256，
该操作也不反序列化真实模型。报告为 `audit_reports/mizar_official_eval_validation.json`。

Windows 本地缺少 Torch/Transformers，CPU tensor 检查和远程真实 checkpoint 加载、
GPU 完整评测均需要在具备依赖的环境确认；模拟流程或文件 hash 通过不代表 GPU PASS。
