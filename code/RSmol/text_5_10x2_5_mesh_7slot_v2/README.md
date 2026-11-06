# 七槽 v2 MeSH：转换、BF16完整一轮文本训练与音频交接

从四槽 BF16 完整一轮路线复制隔离。新模型为
`recursive_model_5_10x2_5_mesh_7slot_v2.py`，类名
`RecursiveLlama5_10x2_5Mesh7SlotV2ForCausalLM`。
架构合同为 `logical_30_physical_20_5_10x2_5_mesh_7slot_v2_3router`，
训练合同为 `text_mesh_x2_7slot_v2_bf16_full_epoch_v1`。
历史七槽的3081步路线、音频七槽/文本五槽路线以及原四/六槽实现保持原样。
本地仅维护代码，通过GitHub同步；实际权重、数据、转换和GPU训练均在远程Linux。

## 结构与转换

hidden576、20物理层/30逻辑执行、两轮共享中段，结构为 `5+10×2+5`。
执行轨迹为 `0..14,5..14,15..19`。每forward重建 `[B,7,L,576]` memory，
slot0为原始input embeddings，其余六槽为零；memory不是持久参数。
三组独立write/read共六个 `Linear(576,7)`，12份参数tensor，共24,234个参数。
prefix query/value取prefix output；循环query取循环输入、write value取循环输出。
读写公式保持；router初始化为 `sqrt(2/(7×hidden_size))`，采样后截到±3std，bias零。

CPU转换默认来源：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5
```

接受原始30层SmolLM2或带明确转换metadata的干净20层5-10-5产物。
30→20映射保持 `0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29`；
20层来源直接继承，不再次抽层。逐tensor核对继承权重和tied embedding，保留tokenizer，
只新初始化七槽routers。默认拒绝已训练/已有MeSH来源、覆盖目标及Git checkout内模型产物。
转换metadata记录来源、映射、seed、继承核对、12份新router参数及架构合同。

默认转换产物：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x2-5-mesh-7slot-v2
```

## 正式训练配置

| 项目 | 配置 |
|---|---|
| 调度资源 | pdgpu-3090 / 64 CPU / 256G / 8 GPU / 1 node |
| 环境 | 原容器与rsmol，不升级环境 |
| 精度 | FP32参数和AdamW状态，模型前向BF16 autocast，FP32 CE/router softmax |
| 其他数值策略 | 无GradScaler、TF32关闭；CE、归一化及backward在autocast外 |
| Batch | 8×microbatch8×GA16=GBS1024 |
| Context | 1024，原padding/attention mask |
| 数据 | 原SmolLM2-135M-10Bsubset/data/*.parquet，persistent直接读取 |
| 顺序 | 排序后shard index modulo8，各rank首轮shard/行顺序不变 |
| 时长 | 完整一轮9244 optimizer updates，463 warmup |
| LR | max8e-4，min8e-5，原step-level warmup/cosine |
| AdamW | betas(0.9,0.95)、eps1e-8；矩阵decay0.1、bias/norm/低维不衰减 |
| Loss | 全文本causal NTP，按八rank整个GA窗口的有效目标token总数归一化 |
| 裁剪 | 全局L2=1.0，非有限梯度报错；router collapse为非致命诊断 |
| 保存 | 每500步及最终步，保留最近三份完整checkpoint |
| RNG | seed0，各rank seed+rank；保存各rank Python/CPU/CUDA RNG |

必须支持CUDA BF16；不静默回退。CPU仅允许Gate A诊断。
FP32 embedding/residual/memory语义保持，BF16不意味着全部激活减半。
checkpoint的config.torch_dtype=float32描述保存权重精度；BF16策略另写metadata/state/report。

## 三段运行命令

本地将全部新代码和RM.md提交推送至GitHub后，在远程登录节点同步并转换：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/scripts/convert_stepwise_5_10x2_5_mesh_7slot_v2.sh
```

wrapper激活rsmol、离线、清空CUDA_VISIBLE_DEVICES，仅CPU转换。
可通过 `RSMOL_5_10X2_5_MESH_7SLOT_V2_SOURCE_CHECKPOINT` 指定干净的原始SmolLM2来源，
通过 `RSMOL_5_10X2_5_MESH_7SLOT_V2_OUTPUT_DIR` 指定独立转换目录。

转换成功后提交文本正式训练；以下变量用于保留后续音频初始化路径：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
TEXT_RUN="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh_7slot_v2/formal_1epoch_bf16_$(date +%Y%m%d_%H%M%S%N)"
RSMOL_5_10X2_5_MESH_7SLOT_V2_OUTPUT_DIR="$TEXT_RUN" \
bash run_stage4_5_10x2_5_mesh_7slot_v2_formal_3090.sh
printf 'Audio text initialization: %s/checkpoint-009244\n' "$TEXT_RUN"
```

正式入口直接通过vc提交，不要求先跑诊断smoke。确认文本作业完成，且产生
`checkpoint-009244`、完整metadata及 `stage4_gate_FORMAL_audit.json` 后，再提交音频：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_7slot_v2_ced_small_3090.sh \
  --text-model-dir "$TEXT_RUN/checkpoint-009244" \
  --stage-workers 4
```

新开shell时先将TEXT_RUN恢复为上一步打印的实际文本run目录。音频入口要求显式路径，
无预设时间戳、无自动选择latest。它在登录节点进行CPU metadata检查，计算节点再次校验并将
实际模型/config/tokenizer内容SHA和文本provenance绑定到音频runtime恢复身份。
只继承LM、已训练routers和tokenizer，不读取/恢复文本optimizer、scheduler、RNG或数据游标。
音频阶段沿四槽音频母本使用FP32，详见
[七槽音频配置与恢复](../mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_7slot_v2_ced_small/README.md)。

## 文本保存恢复

默认输出为远程 `outputs/RSmol/stage4_5_10x2_5_mesh_7slot_v2/formal_1epoch_bf16_<timestamp>/`。
最终为其下 `checkpoint-009244/`。每次完整保存含模型/config/tokenizer、AdamW、scheduler、
全局step、八rank游标/RNG、manifest及complete marker；恢复实际跳过已消费Parquet行。
只接受本v2 BF16正式训练的相同batch/精度/数据/seed/调度合同，目标仍为累计9244步。
正式恢复显式使用FORMAL_RESUME_FROM和新的空输出目录：

```bash
RSMOL_5_10X2_5_MESH_7SLOT_V2_FORMAL_RESUME_FROM=/absolute/path/to/original/checkpoint-005000 RSMOL_5_10X2_5_MESH_7SLOT_V2_OUTPUT_DIR="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh_7slot_v2/resume_$(date +%Y%m%d_%H%M%S%N)" \
bash run_stage4_5_10x2_5_mesh_7slot_v2_formal_3090.sh
```

可选诊断为独立Gate D 0→10、Gate E 10→12；它们使用短scheduler，正式训练从转换模型新开始：

```bash
bash run_stage4_5_10x2_5_mesh_7slot_v2_smoke_3090.sh
RSMOL_5_10X2_5_MESH_7SLOT_V2_RESUME_FROM=/actual/smoke/output/checkpoint-000010 bash run_stage4_5_10x2_5_mesh_7slot_v2_resume_3090.sh
```

## 验证与边界

仓库根目录运行 `python -B tests/validate_mesh_7slot_v2.py`。
本地116项实际检查通过，52项真实tensor检查缺依赖跳过（新路线26、四槽回归26）。
包括真实文本checkpoint writer到音频CPU preflight/config衔接、错槽/旧七槽/短训练/精度/不完整产物拒绝、
完整model/converter/text trainer受限AST对照、完整音频更新循环对照、206个原文件SHA、
真实Bash假提交/参数转发/CPU-only转换/共享路径启动、原四槽与历史七槽回归。
所有新Python/Bash语法及Shell LF通过。报告为 `audit_reports/mesh_7slot_v2_validation.json`。
初次Windows沙箱运行遇临时目录权限问题；同一流程沙箱外复验通过。
`--require-tensors`禁止所有tensor skip，包含CUDA BF16检查，需要完整依赖及GPU计算节点。
真实小模型tensor测试覆盖转换继承/保存重载、七槽memory/读写来源、两轮梯度及恢复后下一步一致性。
本机缺torch/transformers等，未读取远程生产权重/数据，也未执行生产转换、GPU训练或八卡恢复。
不登记远程PASS。本轮未执行commit/push或真实作业提交；现有MMAU/MMAR evaluator需要另行适配。
