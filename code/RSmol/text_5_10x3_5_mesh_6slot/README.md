# 纯文本固定三循环 MeSH：5＋10×3＋5，六槽、四组 routers

本路线独立于纯三循环和历史七槽三循环路线。20 个物理层按
`0..4 → (5..14)×3 → 15..19` 执行40次，中间三轮复用同一份十层参数。
KV cache 有40个逻辑槽；文本训练 `use_cache=False`，所有 backbone 和 routers 全量训练。

MeSH 记忆为每次 forward 新建的 `[B,6,L,576]` 激活：槽0放原始输入 embedding，其余五槽置零。
共有四组独立 write/read routers，即8个 `Linear(576,6,bias=True)`、16个参数 tensor、27,696个参数。
第0组 query/write value 均为 prelude 输出；后三组 query 为各轮中间层输入，write value 为该轮输出。
每次 `memory += value * write_weights`，然后加权 read 得到下一阶段输入；不做 embedding scale，
不将记忆存为参数或 checkpoint 状态。router softmax FP32，无额外辅助 loss；坍缩诊断只告警。
router weight 初始化为 std=`sqrt(2/(6*576))` 的正态后 clamp ±3std，bias0，兼容 meta 加载。

本地 Windows 编辑核心代码，通过 GitHub 同步；远程 Linux 保存全部资产和执行训练。
十个母本保持不变，归一化 LF SHA256 见 `source_guard.json`。

| 项目 | 六槽三循环 MeSH 路线 |
| --- | --- |
| 模型 | `code/RSmol/recursive_model_5_10x3_5_mesh_6slot.py` |
| 转换 | `code/RSmol/scripts/convert_stepwise_5_10x3_5_mesh_6slot.py` / `.sh` |
| 训练 | `code/RSmol/scripts/train_stage4_5_10x3_5_mesh_6slot_ddp.py` / `.sh` |
| 提交 | `code/RSmol/run_stage4_5_10x3_5_mesh_6slot_formal_3090.sh` |
| 环境变量前缀 | `RSMOL_5_10X3_5_MESH_6SLOT_` |
| 架构合同 | `logical_40_physical_20_5_10x3_5_mesh_6slot_4router` |
| 训练合同 | `text_mesh_x3_6slot_bf16_full_epoch_v1` |
| 转换产物 | `/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x3-5-mesh-6slot` |
| 训练输出根 | `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x3_5_mesh_6slot/` |
| 提交日志 | `code/RSmol/log/text_x3_mesh6_formal.<timestamp>.JOB.log` |

## 远程转换和正式提交

在远程登录节点同步代码并进行 CPU 离线转换。默认源为干净的
`models/SmolLM2-5-10-5`，已有 20 个物理层直接继承，不二次抽层。
也可显式选择原始 30 层 `models/SmolLM2`，按固定源层
`0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29` 映射。
转换不接受训练 checkpoint 或已有新递归/MeSH 路线；输出目录必须不存在。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/scripts/convert_stepwise_5_10x3_5_mesh_6slot.sh
bash code/RSmol/run_stage4_5_10x3_5_mesh_6slot_formal_3090.sh
```

转换用 CPU、8 线程、离线加载；验证所有继承 tensor、40 次 forward、保存重载的
tensor/logits 和 tokenizer 一致性后发布目录。已有产物不可覆盖；失败 staging 保留诊断。
不读取训练数据。转换成功后不必重复转换，后续可单独执行最后一行提交训练。

需要原始 30 层源时，仅将转换命令替换成：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RSMOL_5_10X3_5_MESH_6SLOT_SOURCE_CHECKPOINT=/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2 \
  bash code/RSmol/scripts/convert_stepwise_5_10x3_5_mesh_6slot.sh
```

提交器直接 `vc submit`：`pdgpu-3090`、1 节点、8 GPU、64 CPU、256G，
镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`。
容器激活 `/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/rsmol` 后调用本路线 `torchrun`。
GPU 训练必须走此提交器；不在登录节点运行训练 Python 或 torchrun。
不自动运行数据审计、Gate B 或 smoke。正式入口清空遗留通用 `RESUME_FROM`；恢复使用下节专用变量。

## 训练配方与数据口径

两条新路线均复制较新的七槽 v2 全 epoch 训练母本：
8 rank × micro4 × GA32 = GBS1024，**9,244 updates / 9,244 scheduler steps / 463 warmup**，
epochs=1，context1024，seed0。项目一轮的名义样本额度为 `9244×1024=9,465,856`；
不保证严格遍历全部有效文本恰好一次。流保留 shard 尾部不足 micro4 的 batch，实际样本/token 数由文本长度和分片边界决定。

两条路线均使用现有 85 个 parquet，按文件名排序后 `index % 8` 分 rank，第一轮保持 shard/行顺序，
直接持久盘读取、动态 padding、不 packing、不引入旧纯文本的 bounded shuffle。
不扫描 footer 或生成审计报告；只发现并核对 85 个文件。按 rank 保留准确的 shard/row cursor。

FP32 参数/AdamW 状态、仅模型 forward 用 BF16 autocast、FP32 CE、backward 在 autocast 外，
无 GradScaler、TF32 关闭。跨所有 rank 和 32 次累积的有效 token 归一化梯度，clip1.0。
AdamW `(0.9,0.95)`、eps1e-8、amsgrad=false；矩阵 decay0.1，bias/norm/低维参数 decay0。
这些读取/分组规则与历史 x2 纯文本不同，两个新 x3 路线保持一致。
max/min LR `8e-4/8e-5`，沿用母本 warmup/cosine 公式和 step 索引；最后一次更新用索引9243，
索引9244才正好落到 min LR。日志 loss 为 rank0 本地有效 token 加权均值，梯度归一化使用全局有效 token。

每 500 步及最后一步保存，保留最近三个完整 checkpoint，正常结束为
`checkpoint-008500`、`checkpoint-009000`、**`checkpoint-009244`**。
每次提交自动生成独立时间戳输出目录，可用 `RSMOL_5_10X3_5_MESH_6SLOT_OUTPUT_DIR` 指定新的空目录。
模型覆盖路径用 `MODEL_DIR`；转换覆盖路径用 **`CONVERT_OUTPUT_DIR`**，两者与训练 `OUTPUT_DIR` 分开。
自定义转换输出后必须设置同前缀 `MODEL_DIR` 指向该转换产物。

## 同路线完整恢复

将下方示例中的 `REPLACE_WITH_PREVIOUS_RUN` 换成实际训练输出目录。恢复必须来自本路线同一 FORMAL 配方，
保留完整模型/config/tokenizer、optimizer、scheduler、所有 rank 的数据游标和 Python/CPU/CUDA RNG。
从累计 step 继续至9244，输出使用新的空目录；不能拿纯三循环、x2、旧七槽或诊断 checkpoint 恢复。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RSMOL_5_10X3_5_MESH_6SLOT_FORMAL_RESUME_FROM=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x3_5_mesh_6slot/REPLACE_WITH_PREVIOUS_RUN/checkpoint-009000 \
  bash code/RSmol/run_stage4_5_10x3_5_mesh_6slot_formal_3090.sh
```

训练目录包含 `stage4_gate_FORMAL_audit.json`、`ddp_diagnostics/rank*.heartbeat.json` 和独立 checkpoint。
checkpoint 内有 `mesh_checkpoint_metadata.json`、`training_state.pt`、`data_manifest.json`、
`checkpoint_manifest.json`、`checkpoint_complete.json`。只有正式报告 PASS 且 optimizer_steps=9244、
完整 `checkpoint-009244` 存在时，才可登记远程训练完成。

## 验证与代码提交

统一验证命令为 `python -B tests/validate_text_x3_routes.py`，远程 CPU 张量验证加 `--require-tensors`。
两条路线共用的完整验证命令和精确 Git add/commit/push 命令见
[纯三循环文档的验证与代码提交](../text_5_10x3_5/README.md#验证与代码提交)。
本地缺少 Torch/Transformers 时只报告本地合同、游标和 Bash 测试通过，真实 tensor 测试跳过；
未访问远程权重/数据、未代为提交作业，不代表八卡 GPU 训练或恢复已验证。
