# S3：L11～14 三轮共享循环、两处残差门控、独立训练 L15

此目录是隔离的新训练实现。入口为 `code/RSmol/run_mizar_s3_l11_14x3_residual_gate_5090.sh`；原 `run_mizar_l9_12x3_strong_reason_5090.sh` 转发到这个入口并原样传参。原 strongAC 训练包、S3 全参数训练入口、数据准备和评测实现保留。

## 模型与初始化

```text
L1..10 → L11..14 第1遍 → L11..14 第2遍 → L11..14 第3遍 → L15 → L16..30 → 原 final norm
```

四层权重在三遍中共享，共 38 次 block 调用。L15 解冻但只在三轮后执行一次。

第一遍正常执行 `H1=Fθ(H0)`，不加偏置、不使用门控。第二、第三遍分别执行：

```text
Xr = Hprev + pr
Gr = sigmoid(Linear_r(RMSNorm(Xr)))
Hr = Xr + Gr * (Fθ(Xr) - Xr)           r=2,3
```

`RMSNorm` 无 affine 参数，epsilon 使用原 LM 的 `rms_norm_eps`。两个 `Linear(576,1)` 独立，weight 全零、bias=-4.595，初始 gate≈0.01000119；每个 token 一个标量，广播到 576 通道，对音频、SEP、文本、PAD 均生效。偏置只有两个 576 维向量，全零初始化，分别在第二、三轮之前加到所有 token，进入 gate、F 和残差直通分支。

初始化时额外两轮影响较小；仍计算三遍循环，并非跳过计算，也不与原模型严格相等。没有音频锚点、音频重注入、共享 gate、tanh gate 或第一遍 gate。保留完整 BPTT；冻结后缀仍对输入传播梯度，关闭 KV cache 和 gradient checkpointing。

从 `/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt` 完整 strict 加载原始 454 个 FP32 状态，并逐值核验，再创建新参数。使用原模型目录内的文本、CED、projector 资产。仅 L11～15、两个 gate 和两个偏置可训练；CED、projector、embedding/tied head、外层和 final norm 冻结，冻结模块保持 eval。

可训练参数 **17,702,786**：L11～14 为 14,160,384，L15 为 3,540,096，新增 gate+偏置为 2,306；冻结原参数 141,599,360。状态字典为 459 项。

## 数据与音频

复用已发布、已过滤的完整 S3 目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s3_full_filtered_v1
```

使用 `train.json` 与同目录 `resolved_paths.tsv`；272,796 条全部参与每个 epoch，不重新筛 strongAC、不加入 ReasonAQA、不沿用参考入口的半轮预算。

| 来源 | 行数 |
|---|---:|
| AVQA | 26,481 |
| AudioCaps | 49,486 |
| Clotho | 1,991 |
| CompA-R | 48,045 |
| LP-MusicCaps-MTT | 7,871 |
| MusicCaps | 631 |
| SpeechCraft | 113,975 |
| TACOS | 24,316 |

`prepare_manifest.py` 只读调用原 S3 封存校验；`prepare_index.py` 在登录 CPU 上核验 seal、来源计数、映射及全池文本长度，生成本实验独立的不可变索引。默认提交自动执行这些检查，已有 S3 数据无需重建。

音频为 **SoundFile 解码 → 重采样 32kHz → 多声道均值转单声道 → 超过 20s 取中间 20s → 不足 20s 右补零**，最终 640,000 个采样点。中心起点为 `floor((重采样后长度-640000)/2)`。随后拆为 10+10 秒，再拼回整段供原 CED 在线计算；不缓存 encoder 特征。

原 S3 数据记录中的历史 `audio_input_contract`/seal 仍描述 first20；本实验在 `contract.py` 和运行配置中明确采用 **center20 运行时覆盖规则**。不修改或重新封存原 S3 数据，原全参数实验继续使用原规则。

保留 S3 的 question+原序 a/b/c/d 选项及小写：prompt256 不加 EOS，answer96 加空格 EOS；任何截断在预检或 loader 中报错。音频126+SEP1+prompt256=prefix383，总长479；PAD=`!`/17，SEP/EOS=0，attention_mask=None，答案监督使用 logits382..477 与 96 个答案位置，忽略答案 PAD。

staging 沿用 shared-first：计算节点内单个后台进程复制去重原音频字节到独占 `/dev/shm/mizar_s3_l11_14x3_gate_<token>`。复制全部完成且文件数、字节数和源文件 stat 核验通过后，八 rank 在 optimizer 边界以 collective MIN 一致切换；第二个 epoch 复用同一份缓存，恢复任务会重新 staging。退出时先停止并回收 producer，再按 owner/token 清理。

## 训练参数

| 参数组 | max LR | min LR |
|---|---:|---:|
| 两个 gate + 两个循环偏置 | 6e-5 | 6e-6 |
| 共享 L11～14 | 1.5e-5 | 1.5e-6 |
| L15 | 5e-6 | 1e-6 |

单机八张 5090，microbatch4×GA4×8=GBS128。FP32，关闭 autocast 和 TF32；AdamW β=(.9,.999)、eps=1e-8，weight decay=.01，bias/一维参数及循环偏置不衰减。沿用 GradNormTracker 自适应裁剪、非有限梯度检查和 1e6 应急阈值。

前 **100 个真实 optimizer 更新** warmup，沿用启动下限 0.1×各组 max LR，第100步达到峰值；随后各组独立 cosine 到各自 min LR，最后一次实际更新达到 min。两个完整 epoch 每轮重新确定性打乱，无重复补齐、无丢尾；每轮2,132步，合计 **4,264步**，最后一个更新共28条、GA1。

约每0.5 epoch 在 optimizer 边界保存一次：**1066、2132、3198、4264**，最多保留4个完整 checkpoint。checkpoint 包含全模型、AdamW、scheduler、scaler、GradNormTracker、八 rank RNG 和 epoch/update 游标。

## 远程执行

在 GitHub 同步完成后，于远程仓库根执行。登录节点环境默认沿用 `mellow_c8204d8`，Transformers 固定4.54.1；镜像、8GPU/32CPU/256G/1node 沿用原5090任务。不自动安装或修改环境。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# 只打印配方，不读取数据、不申请 GPU
bash code/RSmol/run_mizar_s3_l11_14x3_residual_gate_5090.sh --check-args

# 登录 CPU 检查数据、全池文本、官方资产与输出目录；不提交 GPU
bash code/RSmol/run_mizar_s3_l11_14x3_residual_gate_5090.sh --preflight-only

# 正式提交一次；会再次核验并复用相同源码/数据的 CPU index
bash code/RSmol/run_mizar_s3_l11_14x3_residual_gate_5090.sh
```

也可直接运行旧入口 `bash code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh`，行为与新入口一致。只运行一个正式提交命令即可。原 `--source`/`--resolved-paths`/`--prepare-only` 不再用于此入口；数据目录内必须已有完整封存的 S3 数据。

每次默认生成新的输出目录，失败的非空输出不可用于 fresh run。支持 `--data-dir`、`--model-dir`、`--output-dir`、`--index-cache-root`、`--stage-workers` 和 `--resume-checkpoint`。`--output-dir` 是外层目录，实际训练写到其 `formal/` 子目录。

仅接受本实验相同源码、官方资产、S3 内容、环境和配方的完整 checkpoint；旧 strongAC/AudioMCQ/全参数 S3 checkpoint 不兼容。原 run 内恢复示例：

```bash
bash code/RSmol/run_mizar_s3_l11_14x3_residual_gate_5090.sh \
  --resume-checkpoint /完整原输出目录/formal/checkpoints/model--step-2132.ckpt
```

入口从 checkpoint 推导原外层输出目录；拒绝已经完成的 checkpoint、存在更新 checkpoint 的回退、源码/参数组/数据身份变更。若代码发生修改，应另起新实验。

## 日志与结果

提交时打印确切的 `log_path`、`output`、`admission`。主日志位于：

```text
code/RSmol/log/mizar-s3-l11-14x3-gate.<时间>.JOB.log
```

```bash
LOG=$(ls -t code/RSmol/log/mizar-s3-l11-14x3-gate.*.JOB.log | head -n 1)
tail -f "$LOG"
```

默认结果根在 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s3_l11_14x3_residual_gate_2epochs_gbs128_5090/formal_<时间>/formal/`。每步写 `metrics.jsonl`，含三组实际/下一步 LR、loss、梯度、有效全局批量和 staging 读取模式；stdout 每25步及最后一步打印。每个来源的 `SOURCE_PROBE_PASS` 后才进入训练。`staging_progress.json` 是音频复制进度，不能据此判断已经训练。

`config_audit.json`、`trainable_params.txt`、`first_gradient_rank*.json`、`epoch*_rank*.json` 和 `rank*_completion.json` 保存配方、可训练参数、梯度及覆盖审计。最终 checkpoint 是 `checkpoints/model--step-4264.ckpt`，成功以 `training_report.json` 的 PASS 和 checkpoint reload 核验为准；异常写 `failure_rank*.json`/外层失败记录。

## 本地验证与同步

测试入口为 `tests/validate_mizar_s3_l11_14x3_residual_gate.py`，必须从 `tests/` 执行，避免仓库 `code/` 遮蔽 Python 标准库。使用已有的兼容 PyTorch/Transformers4.54.1 音频环境，不要求下载资产。

```bash
cd tests
python validate_mizar_s3_l11_14x3_residual_gate.py --official-bundle /本地可选的/Mizar159M
```

包含真实小30层 Llama eager/SDPA、独立公式/因果性/BPTT、实际 SoundFile/重采样/中心裁剪、S3封存索引夹具、两轮 trainer/中途恢复逐值对照、两进程 Gloo 和真实 Bash 控制流（vc 为模拟）。可选官方资产检查用真实454权重/CED/projector执行479-token前向反向和 AdamW，并验证冻结权重逐值不变。其结果不代表已验证远程生产音频、八卡 NCCL、显存和吞吐。

本地 PowerShell 的 Git 命令（本次实现没有自动 commit/push）：

```powershell
git add -- RM.md code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh code/RSmol/run_mizar_s3_l11_14x3_residual_gate_5090.sh code/RSmol/mizar_s3_l11_14x3_residual_gate tests/test_mizar_s3_l11_14x3_residual_gate.py tests/validate_mizar_s3_l11_14x3_residual_gate.py audit_reports/mizar_s3_l11_14x3_residual_gate_validation.json
git diff --cached --check
git diff --cached --stat
git commit -m "Add S3 gated L11-14 recurrent training with center20 audio"
git push
```

`sync_files.txt` 列出交付文件及运行依赖。`parent_sources.json` 记录未改动参考文件 SHA；本实验源码 SHA 同时绑定新包、新旧入口及封存 S3 prepare 依赖。
