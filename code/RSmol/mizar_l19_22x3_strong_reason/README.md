# 原始 Mizar：L19–22 共享三轮，仅训练四层，无额外注入

从 `mizar_l9_12x3_strong_reason` 完整复制隔离。原路线的代码、入口、数据和输出保持。正式权重、音频和训练输出在远程 Linux，本地代码通过 GitHub 同步。

## 模型与冻结合同

严格加载完整官方 `/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt`；先检查 18 个 release 文件的大小/SHA，再严格继承原始 454 个状态，包括已训练 CED、projector 和全部 30 层 LM。assets 中的基础权重仅构造模块，最终由 final checkpoint 完整覆盖。无需转换模型或加载旧递归训练 checkpoint。

按原始一基层号执行 `1..18 → (19..22) × 3 → 23..30`，30 个物理层、38 次 block 调用。三轮使用同一组四层 Parameter；上一轮输出直接进入下一轮，所有 683 个位置参与循环。无 A8/anchor、额外音频残差、audio gate、LoRA、MeSH 或新增模型参数；正常 Transformer 层内残差保留。

只解冻第 19–22 层的全部 attention、MLP 和层内 norm，共 14,160,384 个参数，每层 3,540,096。其余 145,139,456 个参数冻结且相关模块保持 eval，包括 CED/projector、embedding/head、其余 26 层和最终 norm。完整模型仍有 159,299,840 个独立参数、454 个 state 条目，embedding/head 绑定不变。

前 18 层在 no_grad 下执行；三轮完整 BPTT，不 detach。第 23–30 层、最终 norm/head 虽冻结，保留对输入的反向传播。最终 norm 仅在第 30 层之后执行一次。无 KV cache、跨 forward 隐状态或 decoder gradient checkpointing。

## 与参考路线一致的训练设置

- 同一份封存 strongAC/ReasonAQA 数据：旧训练实际使用过的完整 strongAC 视图 N 条，ReasonAQA 无放回抽 N 条，总 2N；不因换层重抽、过滤或改变顺序。保留旧训练成员证据、原路径、直接答案和 CoT。
- 8 GPU × microbatch4 × GA4，最大 GBS128，完整一个 epoch，更新数 ceil(2N/128)。动态尾批、每卡 1..4 条及空 rank、无丢弃和 sampler padding；空 rank 仅零任务损失参加 DDP。
- AdamW max/min LR 1e-4/1e-5、betas=(0.9,0.999)、eps=1e-8、矩阵 WD0.01、bias/一维 WD0。实际总步数 round(5%) warmup，之后 cosine，最后一次真实更新到 minLR。原 GradScaler/GradNormTracker 保持。
- FP32、无 autocast、TF32 关闭；seed1234，模型 RNG 为 1234+rank。每卡 microbatch 有效答案 token mean → 实际 GA mean → 有效 rank mean。
- SoundFile 解码、mono32k、fixed20 中心裁剪/右补零，再分 10+10；在线 CED/projector。126audio+SEP1+prompt256+answer300=683，prefix383，PAD17、EOS/SEP0；原 S1 模板与 EOT 后截断，PAD 可见因果注意力，answer-only shifted CE。
- 每 500 步及实际最后一步完整保存，保留最近三份；完整状态恢复、数据游标、8rank RNG、具名 optimizer moments、scheduler、Scaler、GradNormTracker、日志回退及最终 checkpoint 重读均保持。

## 数据复用与正式提交

默认数据目录故意沿用原名称：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/l9_12x3_strong_reason_v1
```

目录内使用最新历史成员过滤修复生成的 sealed manifest（schema v2）。新路线校验相同 seal 和五个数据文件 SHA，按自己的源码身份生成独立 CPU index，不重建数据成员。若原数据已经成功准备，直接提交；若此前只有失败的 `.building-*`，先用原准备入口生成，旧诊断目录无需清理。

代码经 GitHub 同步后，在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# 仅当原 sealed 数据尚未成功生成时执行；已准备完成可跳过
bash code/RSmol/prepare_mizar_l9_12x3_strong_reason.sh

# 参数预览，不创建 index、不读取权重、不提交
bash code/RSmol/run_mizar_l19_22x3_strong_reason_5090.sh --check-args

# CPU seal/index/release 检查成功后，只提交一次正式作业
bash code/RSmol/run_mizar_l19_22x3_strong_reason_5090.sh
```

原 preparer 按相同输入幂等复用；若使用自定义数据目录，准备与训练须显式指向同一目录。

```bash
bash code/RSmol/run_mizar_l19_22x3_strong_reason_5090.sh \
  --data-dir /hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/l9_12x3_strong_reason_v1 \
  --model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/my_l19_22x3_run \
  --stage-workers 4
```

资源保持 pdgpu-5090、8GPU/32CPU/256G/1node；镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`，环境 `mellow_c8204d8`，Transformers4.54.1。沿原 Miniconda 路径，支持 RSMOL_CONDA_BASE/MELLOW_CONDA_BASE；不修改共享环境。直接 formal，无额外 GPU smoke 或人工放行前置。

默认输出与日志：

```text
outer root: /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
            mizar_final_l19_22x3_strong_reason_gbs128_5090/formal_<tag>/
training:   <outer root>/formal/
index:      .../mizar_final_l19_22x3_strong_reason_gbs128_5090/index_cache/<signature>/
log:        code/RSmol/log/mizar-core19.<tag>.1.log
tmpfs:      /dev/shm/mizar_l19_22x3_sr_<unique-token>
```

`--output-dir` 指外层 run root，内部自动追加 formal。顶层打印确切 log_path、admission、manifest 和 output。CPU READY_FOR_GPU、vc 提交成功均不是训练 PASS。

单后台 producer 默认 4workers 复制本数据集去重原音频；共享盘立即训练。全部文件/字节验证后，八 rank 在 optimizer 边界 MIN 集合判定共同切本地。保留 producer 心跳/死亡/错误/容量检查，退出先 stop/wait 回收再按独立 root/token 清理；恢复重新 staging。

正式完成看 `formal/training_report.json` 的 status=PASS、optimizer_steps=runtime 实际 total_steps、full_epoch_no_drop_no_repeat=true，以及 shell 成功退出。最终 checkpoint 为 `formal/checkpoints/model--step-<total_steps>.ckpt`。不预设旧版 5089 步或远程实际 N。

## 完整恢复

只接受本 L19–22 无注入路线、相同源码/数据/模型/环境/配方的完整 checkpoint。旧 L9–12 checkpoint 不可跨路线恢复。新训练从官方 final 开始，optimizer 等从零建立；本路线中断后用以下命令恢复，继续原 epoch 的剩余部分：

```bash
bash code/RSmol/run_mizar_l19_22x3_strong_reason_5090.sh \
  --resume-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/my_l19_22x3_run/formal/checkpoints/model--step-500.ckpt
```

从 checkpoint 自动推导原外层 run root；显式输出也须同 root。选择最新完整 checkpoint，已完成整个 epoch 不再续训。较新日志/partial tail 原样备份后回退，完整坏行拒绝。

## 本地验证与同步

新测试与报告：`tests/test_mizar_l19_22x3_strong_reason.py`、`tests/test_mizar_l19_22x3_previous_training.py`、`tests/validate_mizar_l19_22x3_strong_reason.py`、`audit_reports/mizar_l19_22x3_strong_reason_validation.json`。

在已有兼容测试环境、tests 工作目录运行：

```bash
cd tests
MIZAR_LOCAL_BUNDLE=/path/to/Mizar159M python -B validate_mizar_l19_22x3_strong_reason.py \
  --official-bundle /path/to/Mizar159M
```

测试覆盖真实小型30层 Llama eager/SDPA、native 单遍对照、38次轨迹、三轮直接衔接和任务梯度、冻结后缀传梯度、无额外参数、683布局/CE、生产尺寸参数库存、真实两进程 Gloo GA4/尾批/空rank、完整 checkpoint 下一更新一致、旧路线拒绝、sealed数据/索引、Bash 单次提交及 stop-reap-cleanup。音频和批量数据夹具明确是合成输入。测试不作为正式训练额外前置。

完整官方验证仅在显式提供已有本地 release 时检查18文件、454继承、生产 CED/projector 与683token CPU前反向及冻结不变；不代表远程音频、CUDA/NCCL、5090显存/吞吐或收敛已通过。最终本地28/28 CPU回归通过、无跳过，完整官方release检查通过；32份Python3.10语法、2份Shell和4块README Bash命令、Git可见性和新增文件whitespace检查通过。原26份继承文件逐字节一致、完整训练更新与LR/分片函数AST保持。实际运行结果见新报告；Windows本地 torchaudio ABI 测试适配仅在测试进程启用，生产代码无此适配。

`source_guard.json` 固定任务开始时的既有源码/测试/文档等摘要；RM只前置记录保留原字节。`sync_files.txt` 给出精确新增交付与原数据准备依赖。代码不会自动 git commit/push 或提交 GPU；下面 Git 命令供操作者执行：

```powershell
Set-Location C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
git add RM.md code/RSmol/mizar_l19_22x3_strong_reason code/RSmol/run_mizar_l19_22x3_strong_reason_5090.sh tests/test_mizar_l19_22x3_strong_reason.py tests/test_mizar_l19_22x3_previous_training.py tests/validate_mizar_l19_22x3_strong_reason.py audit_reports/mizar_l19_22x3_strong_reason_validation.json
git diff --cached --check
git diff --cached --stat
git commit -m "Add isolated Mizar L19-22 three-pass core training without injection"
git push origin HEAD
```

保留母本 LICENSE/NOTICE、Mizar ClearBSD 和原依赖许可。包内数据准备/解析代码按母本保留以复用 seal 校验和数据回归，正式生产数据仍由原准备入口统一生成。
