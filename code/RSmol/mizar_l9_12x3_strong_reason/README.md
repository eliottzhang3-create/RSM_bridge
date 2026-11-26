# 原始 Mizar：L9–12 × 3，strongAC + ReasonAQA，GBS128

本路线独立于已有训练包。本地只改代码，经 GitHub 同步；正式权重、音频、manifest 和训练输出都在远程 Linux。实现不会自动 commit、push 或提交 GPU。顶层训练命令只提交一次正式作业，没有额外 GPU smoke 作业。

## 1. 固定模型与训练口径

完整加载 `/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt`，严格继承原有 454 个状态，包括已训练的 CED、projector 和全部 30 层文本模型。18 个 release 文件逐项检查大小和 SHA256。目录中的基座权重仅用于构造模块，最终全部由官方 final checkpoint 覆盖。

按原始层编号执行：`1..8 → (9..12) × 3 → 13..30`，共 38 次 block 调用，四层共用一份参数。只训练原始第 9–12 层全部参数和一个共享 `Linear(576,1)`，合计 **14,160,961** 个参数；其余原始参数 **145,139,456** 个冻结，并保持 eval。冻结后缀保留输入梯度，三轮完整 BPTT；不使用 KV cache 或 decoder gradient checkpointing。

`A8` 是第 8 层 raw post-block 输出的前 126 个 audio token，位于 final norm 前。第二、第三轮进入第 9 层前，使用前一轮输出的音频潜状态 `H_audio`：

```text
H_audio ← H_audio + tanh(Linear(H_audio)) × A8
```

同一个门控在两处共享，weight、bias 都从 0 初始化；每个 token 得到一个标量，不向 SEP/text 位置直接加残差。每次 forward 重新计算 A8，不保留跨 batch 状态。原始 final norm 只在第 30 层后执行一次。

训练一个完整 epoch：8 GPU × microbatch4 × GA4 = **GBS128**。实际 strongAC 视图数记为 N，ReasonAQA 抽 N 条，总计 2N 行，总更新数 `ceil(2N/128)`。N 在 CPU 阶段根据旧训练实际用过的行确定，不再把 325,648/651,296/5,089 硬编码为当前配额与步数。尾批可不足4条/卡或出现空 rank；空 rank 仅用零损失上下文参加同一 DDP 通信，不重复计入样本或任务梯度。真实非空 rank 的 microbatch loss 求均值，再按实际 GA 平均，无丢弃或 sampler padding。

AdamW：全部可训练参数 maxLR=1e-4、minLR=1e-5，betas=(0.9,0.999)、eps=1e-8、weight decay=0.01，bias/一维参数不 decay。warmup 为实际总步数的 round(5%)，随后 cosine，最后一次真实更新达到 minLR。保留参考实现的 FP32、无 autocast、TF32 关闭、GradScaler 和 GradNormTracker。

音频保持参考 fixed20：SoundFile 解码、mono、32k、长音频居中截取 20 秒、短音频右补零，再分两个 10 秒片段。在线计算 CED/projector，不缓存特征。文本保持 S1 原语义与模板：音频126 + SEP1 + prompt256 + answer300 = 683；PAD17、SEP/EOS0；追加空格 EOT 后截断，不补回被截断的 EOS。因果注意力中 PAD 可见，CE 只统计有效 answer token，logit382..681 对齐 answer300。先求每卡 microbatch 有效 token 均值，再求实际 GA 均值，最后由 DDP 平均。

## 2. 先在 CPU 生成 manifest

从原始 **1,477,976** 行 S1 出发，用旧 `s1_stage1_hybrid_filtered.json` 与配套 audit TSV 核对历史训练成员身份。原始元数据上限如下；这些数量不代表远程音频都可用：

| 视图 | 行数 |
|---|---:|
| strongAC 直接答案 `audiomcq_mcq` | 257,892 |
| strongAC 额外 CoT `audiomcq_cot` | 67,756 |
| ReasonAQA 单原始音频 QA，随机抽样 | 325,648 |
| 总计 | **651,296** |

当前策略按用户确认的规则：旧训练没用过的 strongAC 行过滤；用过的完整原行和直接答案/CoT 视图保留，使用旧 audit 的原路径。如果旧训练用过、但原路径现在失效或旧 manifest/TSV 相互矛盾，则报错要求恢复，不把它伪装成“未使用”而过滤，也不换成猜测文件。原始行与旧行用全部原字段的 SHA 核对；旧 hybrid 各来源总数也必须与既有训练配方的 1,477,492 行一致。

ReasonAQA 候选仍为原始 S1 的 756,968 行，抽样数量自动等于实际保留的 strongAC 视图总数 N。历史 `filepath2` 是处理后的第二音频块，不作为第二个原始录音。按解析后的真实原文件路径分组，seed1234 打乱每组 QA、每轮打乱仍有剩余的录音组，逐组抽一个未用 QA，直至 N 条，以最大化录音覆盖。路径别名归为一组；不声称做了声学近重复检测。

默认旧训练 manifest 为 `/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json`，audit TSV 同参考训练入口。原有 resolver/mapping 作为非历史引用的严格 fallback 保留；已有历史引用只复用原路径。CPU 准备只解析、stat、tokenize，不复制、解码音频或运行模型。日志打印 `PREVIOUS_TRAINING_COUNTS`、`RETAINED_DATASET_COUNTS`，给出保留/过滤的直接答案、CoT、来源、唯一引用与最终配额。

远程仓库根目录执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/prepare_mizar_l9_12x3_strong_reason.sh
```

默认 source 为 `/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1.json`。若保留的是原始压缩文件，可显式指定：

```bash
bash code/RSmol/prepare_mizar_l9_12x3_strong_reason.sh \
  --source /hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1.json.gz
```

要求原始 source SHA256：plain `de92b7efd1dd6b5fd30b3f62c31639a3bcff6766a088849658fdd6bc8520641d`；gzip `111abde09c75abfd8dc6ee66c4762d7557d0b55e62e5cbde0bcc7b485176ac91`。不自动替换已有 S1 文件。

默认生成目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/l9_12x3_strong_reason_v1/
  train.jsonl          原字段/文本/答案保持，按原始 S1 行序输出
  selection.jsonl      每行源索引、视图、原行 SHA、raw/resolved 路径
  resolved_paths.tsv   仅本次选中数据的路径映射
  dataset_report.json  数量、录音覆盖/交集、字节数、CoT/直接答案截断与 EOS 统计
  excluded_strong_rows.jsonl  每个过滤视图的源索引、原行SHA、旧训练未使用证据及旧audit记录
  manifest.json        发布状态、原始/旧训练输入身份、动态配额、五文件大小/SHA 与 seal
```

可覆盖 `--source`、`--output-dir`、`--resolved-paths`、`--mapping-json`、`--model-dir`、`--previous-training-manifest`。默认旧 TSV：`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv`。

同命令可幂等重跑：先检查 seal/文件 SHA，只有相同 source、旧训练 manifest、mapping、preparer 和 tokenizer 才复用，拒绝覆盖不同结果。采用独占 `.prepare.lock` 和同级临时目录，成功后整体发布；失败保留 `.building-*` 内 `failure.json`、`unresolved_audio.json` 等诊断。旧失败目录无需删除；新命令重新生成最终目录。此前若已经成功发布 v1 数据，应给 `--output-dir` 指定新目录，再将其传给训练 `--data-dir`。若进程被强杀，先确认旧 preparer 已停止，再人工处理其遗留锁。

仅校验已生成结果：

```bash
bash code/RSmol/prepare_mizar_l9_12x3_strong_reason.sh --verify-only
```

## 3. 提交一次正式训练

```bash
# 仅打印参数，不激活环境、不创建 index、不提交
bash code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh --check-args

# CPU 校验与 index 成功后，单次 vc submit
bash code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh
```

队列 `pdgpu-5090`，8 GPU、32 CPU、256G、单节点；镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`，conda 环境 `mellow_c8204d8`，Transformers4.54.1。使用原 Miniconda 路径，可通过 `RSMOL_CONDA_BASE`（或 `MELLOW_CONDA_BASE`）覆盖，不安装或升级共享环境。

顶层提交前在 CPU 验证 manifest、构建或复用 SHA 固定的 mmap index、校验完整官方 release 和 resume 身份；失败不会调用 vc。提交时打印 `log_path`、admission、manifest 和输出路径。默认：

```text
log:    code/RSmol/log/mizar-core.<tag>.1.log     # JOB 占位符由 vc 展开
output: /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
        mizar_final_l9_12x3_strong_reason_gbs128_5090/formal_<tag>/formal/
index:  .../mizar_final_l9_12x3_strong_reason_gbs128_5090/index_cache/<signature>/
```

计算节点上只有一个后台 producer，把本次选中数据的去重原音频复制到本作业的 `/dev/shm/mizar_l9_12x3_sr_<token>`。训练先读共享盘；每个文件原子复制并检查大小/源 stat，所有目标文件和总文件数/字节数验证完成后，八个 rank 在 optimizer 边界做 MIN 集合判定，再一起切换本地路径。copy 失败、producer 死亡、心跳异常或容量不足会终止作业。退出先停止并 wait 回收 producer，再按 root/token 所有权清理。恢复时重新 staging。

自定义路径示例：

```bash
bash code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh \
  --data-dir /hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/l9_12x3_strong_reason_v1 \
  --model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/my_l9_12x3_run \
  --stage-workers 4
```

`--output-dir` 始终指**外层 run root**，内部会创建 `formal/`；另支持 `--index-cache-root`。新训练要求空输出目录。完成标志为 `formal/training_report.json` 的 `status=PASS`、`optimizer_steps` 等于 runtime 的实际 `geometry.total_steps`，并通过完整 epoch 校验；成功退出还要求 producer 完成并执行清理。

输出包括 `runtime.json`、`config_audit.json`、`trainable_params.txt`、`metrics.jsonl`、staging 报告、每卡首次任务梯度/完成记录，以及完整状态 checkpoint。每 500 步和实际最后一步保存，保留最近三份。最终 checkpoint 为 `formal/checkpoints/model--step-<total_steps>.ckpt`。训练内校验角色任务梯度非零、冻结参数/缓冲不漂移、完整分片消费，并重读最终 checkpoint。错误写入 `failure_rankN.json` 和 FAILED training_report；也保留 shell 阶段与 staging 诊断。

## 4. 完整断点恢复

只接受本路线相同模型、数据、源码、环境版本与训练配方的**完整状态**。恢复 model、AdamW moments、scheduler、GradScaler、GradNormTracker 和每卡 RNG；不接受权重式续训。使用最新完整 checkpoint；完成整个 epoch 的 checkpoint 不能再续训。此次筛选策略与源码身份更新，旧 v1 数据/运行不允许跨配方恢复；应使用重新生成的 manifest 和新训练目录。

```bash
bash code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh \
  --resume-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/my_l9_12x3_run/formal/checkpoints/model--step-500.ckpt
```

未传 `--output-dir` 时，从 checkpoint 自动推导原外层 run root。显式传入时也必须是原 root。晚于恢复点的完整日志先原样备份为 `metrics.jsonl.before-resume-*`，再截回 checkpoint；不完整尾行一并保留在备份中。缺失或损坏的完整历史行会拒绝恢复。只重新消费尚未训练的确定性分片，最后一次更新仍按实际 GA 计算。

## 5. 本地验证与同步

当前修复报告：仓库根 `audit_reports/mizar_l9_12x3_previous_training_fix_validation.json`；初版完整模型验证保留在 `audit_reports/mizar_l9_12x3_strong_reason_validation.json`，其中 651,296 行是原始元数据上界。测试入口在 `tests/validate_mizar_l9_12x3_strong_reason.py`，含历史成员与任意尾批回归。测试应从 `tests/` 执行，避免仓库已有 `code/` 包遮蔽 Python stdlib。

已验证真实小型 30 层 Llama eager/SDPA、原始30层单遍一致性、38次共享调用、逐 token 公式/位置、因果性、完整反传和冻结保持；真实2进程 CPU Gloo 的 GA4/GA1、AdamW 下一步恢复一致性；manifest 发布/幂等/index/篡改拒绝、staging 时序/失败清理，以及带空格路径的真实 Bash 模拟提交。正式入口生命周期夹具模拟了 vc/torchrun 和 Git Bash 缺少的 flock。

初版对本地完整 S1 元数据实际抽样：651,296 行原始上界，ReasonAQA 覆盖全部 53,673 个原路径组；当前训练的 N 仍需远程旧 hybrid 核对。初版对已有官方完整18文件执行 SHA 校验、严格454状态继承、真实 CED/projector + 683 token 前向/反传/AdamW 更新，冻结状态保持。此次模型结构和权重加载不改。音频为合成静音，不代表真实训练指标。

用户实际报错为 202 个 AudioCaps 文件候选缺失、15 个 MusicCaps 旧 S2 记录不可用、1 个 Clotho basename 歧义。修复不会猜测这些引用对应的音频：逐行检查旧训练是否使用，未使用才过滤；已使用则复用旧 audit 路径，失效即报错。新增测试证明此边界、ReasonAQA 等量、任意尾批/恢复覆盖及真实 Gloo 空 rank 零任务梯度。不能把 218 个引用直接当作 218 条 QA 相减，直接答案和 CoT 的具体影响数以新报告为准。

本地环境为 Torch2.15 CPU、Transformers4.54.1；已有 torchaudio2.9 原生扩展 ABI 不匹配，测试进程显式禁用了未用的原生扩展，实际 SoundFile 与 PyTorch 音频变换正常运行；生产代码没有此适配。未验证远程真实音频齐备、CUDA/NCCL、5090 显存/吞吐或正式收敛，这些由远程 CPU manifest 和正式作业检查给出。

通用验证命令（在已有兼容测试环境中）：

```bash
cd tests
MIZAR_LOCAL_BUNDLE=/path/to/Mizar159M python validate_mizar_l9_12x3_strong_reason.py \
  --source-s1 /path/to/original/s1.json.gz --official-bundle /path/to/Mizar159M
```

`sync_files.txt` 列出交付文件与两项原有映射依赖；旧文件 normalized-LF SHA 由 `source_guard.json` 固定，RM 只前置新记录并保留原全部字节。

本地 PowerShell 提交命令（尚未替用户执行）：

```powershell
Set-Location C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
git add RM.md code/RSmol/mizar_l9_12x3_strong_reason code/RSmol/prepare_mizar_l9_12x3_strong_reason.sh code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh tests/test_mizar_l9_12x3_strong_reason.py tests/test_mizar_l9_12x3_previous_training.py tests/validate_mizar_l9_12x3_strong_reason.py audit_reports/mizar_l9_12x3_previous_training_fix_validation.json
git diff --cached --check
git diff --cached --stat
git commit -m "Align strongAC selection with previous training and balance ReasonAQA"
git push origin HEAD
```

复用来源：参考 GBS256/recursive5-10-5 CED-small 路线的数据、fixed20、staging、mmap index 和参考 loss/GradNormTracker；官方输入模型路线的 CED wrapper、离线 config、Llama 调用兼容层；原始 raw-audio audit 的 resolver 与流式读取。新路线内修改，已有路线文件不改；保留 LICENSE、NOTICE 和依赖许可。
