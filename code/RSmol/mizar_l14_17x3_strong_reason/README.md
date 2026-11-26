# 原始 Mizar：仅训练 L14–17，共享三轮，无额外注入

本路线从 `mizar_l9_12x3_strong_reason` 复制隔离。本地 Windows 修改代码，经 GitHub 同步到远程 Linux；正式数据、权重和训练输出在远程。原 L9–12 路线、数据准备器及已有结果保持。顶层入口只提交一次正式作业，没有额外 GPU smoke 前置。

## 1. 模型与训练边界

默认完整加载：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M/
checkpoints/final/stage3_q4_seed20260905.ckpt
```

继续严格核对18个官方 release 文件的大小/SHA，以及全部454项 FP32、finite 状态。构造模块所用 base assets 最终全部由官方 final checkpoint 覆盖，包括已经训练的 CED/projector 和原30层 LM；不是从旧循环训练 checkpoint 初始化。

原始层号从1开始：

```text
1..13 -> (14..17) x 3 -> 18..30 -> 原 final norm -> 原 lm_head
```

保留30个物理层，执行38次 block；零基层索引为 `0..12,(13..16)x3,17..29`。三轮共用原第14–17层的同一份参数，执行完整683位置，不复制成12个独立层。循环边界直接传递上一轮完整 hidden states，不捕获 A8、不注册 audio_gate、不添加投影、LoRA、router、残差注入或新参数。原 Transformer block 内部正常残差和原126音频 token 输入保持。

只训练第14–17层全部 attention、MLP 和层内 norm 参数：每层3,540,096，合计 **14,160,384**。其余 **145,139,456** 个原参数全部冻结，保持 eval，包括 CED、projector、embedding、其他26层、final norm 和 tied lm_head。state_dict 恰好保留原454项。

前13层可在 no_grad 中执行；后缀18–30层及 final norm/head 保留对输入的 autograd，参数本身无梯度。三轮之间无 detach，完整 BPTT；无 KV cache，无 decoder gradient checkpointing，原 final norm 只执行一次。

## 2. 保持参考训练设置

| 项目 | 设置 |
|---|---|
| 数据 | 相同 sealed strongAC + ReasonAQA，保留原直接答案和 CoT 视图 |
| 配额 | 旧训练实际使用的 strongAC 视图 N，ReasonAQA 无放回取 N，共2N |
| Batch/epoch | 8GPU × micro4 × GA4，最大GBS128，完整1epoch |
| 步数 | `ceil(2N/128)`；尾批按实际 GA，支持不满4条及空rank，无drop/补齐重复 |
| AdamW | maxLR1e-4/minLR1e-5，betas=(0.9,0.999)，eps1e-8，WD0.01，bias/一维不衰减 |
| LR | 原round(总步数×5%) warmup和cosine，最后一次真实更新到minLR |
| 精度 | FP32，无autocast，TF32关闭，原GradScaler/GradNormTracker |
| 随机性 | seed1234，模型/文本随机数按1234+rank，原数据打乱及分片 |
| 音频 | 原SoundFile/mono32k/fixed20中心截取或右补零，再分10+10，在线冻结CED/projector |
| 布局 | 126audio+SEP1+prompt256+answer300=683，prefix383 |
| 文本/loss | 原S1模板、空格EOT后截断/PAD17/EOS0，PAD可见因果注意力；仅有效answer shifted CE |
| 归约 | 每卡micro有效token mean → 实际GA mean → 活跃rank mean；空rank零任务损失参加相同通信 |
| 保存 | 每500步及实际最后一步完整保存，保留最近3份 |

## 3. 数据准备与复用

默认直接只读复用原数据目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/l9_12x3_strong_reason_v1
```

目录名保留 L9–12 是原数据来源记录，不代表新模型的循环位置。`prepare_manifest.py`、`previous_training.py` 和所有数据处理源码逐字节继承；不重新抽样或改写已发布数据。CPU admission 验证原 manifest seal、各文件 SHA、历史成员证据和动态来源/视图配额。新 index 绑定新路线源码 SHA 和同一 dataset seal，在新 cache 根创建；不能直接把旧路线 index 当作新源码 index。

若当前数据已成功准备，无需再次生成。可只校验：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/prepare_mizar_l9_12x3_strong_reason.sh --verify-only
```

若数据尚未准备成功，沿用原 CPU prepare：

```bash
bash code/RSmol/prepare_mizar_l9_12x3_strong_reason.sh
```

原始S1/旧hybrid/audit/映射/tokenizer身份、旧训练未使用才允许过滤、旧使用音频复用原audit路径和失效拒绝的规则全部保持。218是不同引用数，不能直接从QA行数相减。真实N由远程prepare确定，不硬编码651296行或5089步。已有自定义数据目录通过训练的 `--data-dir` 传入。

## 4. 单次正式提交

GitHub完整同步后，在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# 纯参数预览，不激活环境、不建立index、不提交。
bash code/RSmol/run_mizar_l14_17x3_strong_reason_5090.sh --check-args

# CPU校验/index成功后，只提交一次正式作业。
bash code/RSmol/run_mizar_l14_17x3_strong_reason_5090.sh
```

资源保持 `pdgpu-5090 / 8GPU / 32CPU / 256G / 1node`，镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`，conda `mellow_c8204d8`，Transformers4.54.1。原Miniconda路径及 `RSMOL_CONDA_BASE` / `MELLOW_CONDA_BASE` 覆盖保持，不安装或升级共享环境。

自定义路径：

```bash
bash code/RSmol/run_mizar_l14_17x3_strong_reason_5090.sh \
  --data-dir /hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/l9_12x3_strong_reason_v1 \
  --model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/my_l14_17x3_run \
  --stage-workers 4
```

`--output-dir` 是外层 run root，内部创建 `formal/`，fresh要求空目录。不要同时提交默认和自定义命令。默认位置：

```text
output: outputs/RSmol/mizar_final_l14_17x3_strong_reason_gbs128_5090/formal_<tag>/formal/
index:  outputs/RSmol/mizar_final_l14_17x3_strong_reason_gbs128_5090/index_cache/<signature>/
log:    code/RSmol/log/mizar-l14-core.<tag>.1.log
tmpfs:  /dev/shm/mizar_l14_17x3_sr_<token>
```

提交打印确切log/output/manifest/admission路径。CPU检查失败不调用vc。计算节点先读共享盘开始训练，单后台producer默认4workers复制本manifest的去重原音频；完整文件/字节验证后，八rank在optimizer边界MIN集合判定并共同切本地。退出先stop/wait回收producer，再按root/token所有权清理。路径守卫与新tmpfs前缀一起更新。每次恢复重新staging。

## 5. 完整恢复与完成判定

路线身份为 `mizar_final_l14_17x3_strong_reason_gbs128_v2_previous_training`。只接受本路线相同源码、模型、数据、环境和配方的完整checkpoint。恢复模型、具名AdamW库存/moments、scheduler、Scaler、GradNorm、8rank RNG及游标，沿用原外层run root。旧L9–12及其他路线checkpoint不能作为本路线完整resume。

```bash
bash code/RSmol/run_mizar_l14_17x3_strong_reason_5090.sh \
  --resume-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/my_l14_17x3_run/formal/checkpoints/model--step-500.ckpt
```

不传output时从checkpoint自动推导原root；显式output也必须是同root。必须选最新完整checkpoint；整epoch已完成的checkpoint不能再续训。较新metrics/partial尾行原样备份后回退，损坏完整行拒绝。保存文件名由真实更新数决定：`formal/checkpoints/model--step-<total_steps>.ckpt`。

`formal/training_report.json` 的 `status=PASS`、实际总步数、完整无重复/无drop覆盖、冻结状态保持及最终checkpoint重读通过，且shell确认producer完成并清理，才算正式完成。`READY_FOR_GPU` 或提交成功不是训练PASS。

## 6. 本地验证与同步

测试：`tests/test_mizar_l14_17x3_strong_reason.py`、`tests/test_mizar_l14_17x3_previous_training.py`。验证器和报告：

```text
tests/validate_mizar_l14_17x3_strong_reason.py
audit_reports/mizar_l14_17x3_strong_reason_validation.json
```

从tests目录执行，避免已有code包遮蔽stdlib code：

```bash
cd tests
MIZAR_LOCAL_BUNDLE=/path/to/Mizar159M python -B validate_mizar_l14_17x3_strong_reason.py \
  --official-bundle /path/to/Mizar159M
```

验证覆盖：真实小30层Llama eager/SDPA单遍native对照与独立38次纯循环对照；全部位置直接衔接、四层共享、12次核心调用梯度、冻结后缀输入梯度、因果性及冻结状态；原683/CE/数据配额/历史过滤/任意尾批；真实2进程CPU Gloo的GA4/尾GA1/空rank与独立梯度对照；完整恢复下一步一致性；真实Bash入口/模拟vc/torchrun、staging清理与路径守卫。

官方bundle选项使用已有本地官方文件及合成静音，真实CED/projector+683位置前向/反传/AdamW检查；不是远程音频或GPU验证。本地音频适配仅用于测试进程中禁用未用torchaudio native扩展，生产源码不含适配。实际运行结果和版本见验证报告。远程生产音频齐备、CUDA/NCCL、5090显存/吞吐和收敛仍由服务器确认。

`source_guard.json` 固定本任务开始时已有tracked Python/Shell/JSON及RM的normalized-LF SHA。RM只前置记录，原字节保留。`sync_files.txt` 给出新交付与保持的依赖。原路线文件、测试和报告不改。

本地PowerShell同步命令（由用户执行）：

```powershell
Set-Location C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
git add RM.md code/RSmol/mizar_l14_17x3_strong_reason code/RSmol/run_mizar_l14_17x3_strong_reason_5090.sh tests/test_mizar_l14_17x3_strong_reason.py tests/test_mizar_l14_17x3_previous_training.py tests/validate_mizar_l14_17x3_strong_reason.py audit_reports/mizar_l14_17x3_strong_reason_validation.json
git diff --cached --check
git diff --cached --stat
git commit -m "Add isolated Mizar L14-17 three-round core training without injection"
git push origin HEAD
```
