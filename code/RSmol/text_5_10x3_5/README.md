# 纯文本固定三循环：5＋10×3＋5

本路线为独立纯文本全量训练：20 个物理层、40 次逻辑执行，前后各五层执行一次，
中间十个物理层用同一份参数执行三次。没有 MeSH、router、记忆槽、LoRA 或音频模块。
KV cache 按 40 个逻辑位置分槽；训练每次 forward 显式 `use_cache=False`。

本地 Windows 仅编辑核心代码，通过 GitHub 同步；权重、parquet、日志和 checkpoint 均留在远程 Linux。
未改原 x2 纯文本及 x3 七槽 MeSH 文件。十个母本的归一化 LF SHA256 见 `source_guard.json`。

| 项目 | 纯三循环路线 |
| --- | --- |
| 模型 | `code/RSmol/recursive_model_5_10x3_5.py` |
| 转换 | `code/RSmol/scripts/convert_stepwise_5_10x3_5.py` / `.sh` |
| 训练 | `code/RSmol/scripts/train_stage4_5_10x3_5_ddp.py` / `.sh` |
| 提交 | `code/RSmol/run_stage4_5_10x3_5_formal_3090.sh` |
| 环境变量前缀 | `RSMOL_5_10X3_5_` |
| 架构合同 | `logical_40_physical_20_5_10x3_5_pure` |
| 训练合同 | `text_pure_x3_bf16_full_epoch_v1` |
| 转换产物 | `/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x3-5` |
| 训练输出根 | `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x3_5/` |
| 提交日志 | `code/RSmol/log/text_x3_pure_formal.<timestamp>.JOB.log` |

## 远程转换和正式提交

在远程登录节点同步代码并进行 CPU 离线转换。默认源为干净的
`models/SmolLM2-5-10-5`，已有 20 个物理层直接继承，不二次抽层。
也可显式选择原始 30 层 `models/SmolLM2`，按固定源层
`0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29` 映射。
转换不接受训练 checkpoint 或已有新递归/MeSH 路线；输出目录必须不存在。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/scripts/convert_stepwise_5_10x3_5.sh
bash code/RSmol/run_stage4_5_10x3_5_formal_3090.sh
```

转换用 CPU、8 线程、离线加载；验证所有继承 tensor、40 次 forward、保存重载的
tensor/logits 和 tokenizer 一致性后发布目录。已有产物不可覆盖；失败 staging 保留诊断。
不读取训练数据。转换成功后不必重复转换，后续可单独执行最后一行提交训练。

需要原始 30 层源时，仅将转换命令替换成：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RSMOL_5_10X3_5_SOURCE_CHECKPOINT=/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2 \
  bash code/RSmol/scripts/convert_stepwise_5_10x3_5.sh
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
每次提交自动生成独立时间戳输出目录，可用 `RSMOL_5_10X3_5_OUTPUT_DIR` 指定新的空目录。
模型覆盖路径用 `MODEL_DIR`；转换覆盖路径用 **`CONVERT_OUTPUT_DIR`**，两者与训练 `OUTPUT_DIR` 分开。
自定义转换输出后必须设置同前缀 `MODEL_DIR` 指向该转换产物。

## 同路线完整恢复

将下方示例中的 `REPLACE_WITH_PREVIOUS_RUN` 换成实际训练输出目录。恢复必须来自本路线同一 FORMAL 配方，
保留完整模型/config/tokenizer、optimizer、scheduler、所有 rank 的数据游标和 Python/CPU/CUDA RNG。
从累计 step 继续至9244，输出使用新的空目录；不能拿 MeSH、x2、旧七槽或诊断 checkpoint 恢复。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RSMOL_5_10X3_5_FORMAL_RESUME_FROM=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x3_5/REPLACE_WITH_PREVIOUS_RUN/checkpoint-009000 \
  bash code/RSmol/run_stage4_5_10x3_5_formal_3090.sh
```

训练目录包含 `stage4_gate_FORMAL_audit.json`、`ddp_diagnostics/rank*.heartbeat.json` 和独立 checkpoint。
checkpoint 内有 `recursive_checkpoint_metadata.json`、`training_state.pt`、`data_manifest.json`、
`checkpoint_manifest.json`、`checkpoint_complete.json`。只有正式报告 PASS 且 optimizer_steps=9244、
完整 `checkpoint-009244` 存在时，才可登记远程训练完成。

## 验证与代码提交

本地统一验证：`python -B tests/validate_text_x3_routes.py`。本地缺 Torch/Transformers 时真实 tensor 测试明确跳过。
远程 rsmol 内可用以下纯 CPU 合成测试，无生产资产，不作为训练前置 gate：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/rsmol
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -B tests/validate_text_x3_routes.py --require-tensors --output /tmp/text_x3_cpu_validation.json
```

本地 Git 提交两个新路线及共同验证的 PowerShell 命令如下。不会加入工作区其他历史修改；
`RM.md` 已有用户修改，须另行审阅其 diff 后决定一并提交。验证报告只登记本地测试，不能代表远程 GPU PASS。

```powershell
cd C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
git add -- code/RSmol/recursive_model_5_10x3_5.py code/RSmol/recursive_model_5_10x3_5_mesh_6slot.py code/RSmol/text_5_10x3_5 code/RSmol/text_5_10x3_5_mesh_6slot
git add -- code/RSmol/scripts/convert_stepwise_5_10x3_5.py code/RSmol/scripts/convert_stepwise_5_10x3_5.sh code/RSmol/scripts/convert_stepwise_5_10x3_5_mesh_6slot.py code/RSmol/scripts/convert_stepwise_5_10x3_5_mesh_6slot.sh
git add -- code/RSmol/scripts/train_stage4_5_10x3_5_ddp.py code/RSmol/scripts/train_stage4_5_10x3_5_ddp.sh code/RSmol/scripts/train_stage4_5_10x3_5_mesh_6slot_ddp.py code/RSmol/scripts/train_stage4_5_10x3_5_mesh_6slot_ddp.sh
git add -- code/RSmol/run_stage4_5_10x3_5_3090.sh code/RSmol/run_stage4_5_10x3_5_formal_3090.sh code/RSmol/run_stage4_5_10x3_5_mesh_6slot_3090.sh code/RSmol/run_stage4_5_10x3_5_mesh_6slot_formal_3090.sh
git add -- tests/text_x3_test_support.py tests/test_text_x3_routes.py tests/test_text_x3_pure_shell.py tests/test_text_x3_mesh_shell.py tests/test_text_x3_tensor.py tests/validate_text_x3_routes.py audit_reports/text_x3_routes_validation.json
git diff --cached --stat
git diff --cached --check
git commit -m "Add isolated fixed x3 pure and six-slot MeSH text training"
git push origin main
```

如暂存区事先存在其他改动，先审阅 `git diff --cached`，再执行 commit。当前交付没有代为 commit、push 或提交远程训练。
