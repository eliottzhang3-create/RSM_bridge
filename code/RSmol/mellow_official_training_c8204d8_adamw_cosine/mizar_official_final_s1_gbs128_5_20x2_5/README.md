# 官方 Mizar final：5 + 20×2 + 5，S1 GBS128 训练

代码在 Windows 本地维护，经 GitHub 同步到远程 Linux。权重、原始音频、索引和训练产物留在远程。这里提供独立训练器和正式提交入口，复用已经转换完成的完整模型。

## 正式提交

本地代码提交并推送 GitHub 后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_official_final_s1_gbs128_5_20x2_5_5090.sh --stage-workers 4
```

默认初始化目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-5-20x2-5-serial84-looplora-r8
```

入口在登录 CPU 上做完整转换产物的只读文件审计，并构建/复用本路线独立的原始路径索引；随后直接提交一个正式训练任务。资源沿用参考：`pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`，原容器及 `mellow_c8204d8`。没有自动 smoke、tensor 前置作业，不自动安装依赖。

可选参数：`--model-dir DIR`、`--output-dir DIR`、`--resume-checkpoint PATH`、`--resolved-paths TSV`、`--index-cache-root DIR`、`--stage-workers N`。初始化使用单个完整模型目录；tokenizer 和 CED 构造资产分别取该目录内的 `tokenizer/` 和 `ced-small/`。

## 初始化与训练配方

- 完整继承官方 final，经既有转换产物严格加载 **1038** 份 state tensor，保留已保存的 LoRA A/B、三个 gate、input projection、CED4/8 MLP。不会重新转换或用基础 assets 权重覆盖 final。
- 保留30个物理层。prelude 0–4，中间5–24共享基础权重循环两次，coda 25–29；执行 `0..24,5..24,25..29`，共50次。
- 中间20层七种投影使用两套独立 rank8/alpha16/scale2/dropout0 LoRA，分别对应两轮，覆盖完整序列；基础权重也训练。
- 逻辑5/25的音频位置0–125依次注入共享 input residual，再注入 CED8/CED4 residual；CED gate读取 post-input hidden。input projection每个LM forward一次，CED每个音频批次一次。
- CED冻结且始终eval。完整模型165,334,019独立参数；新增LoRA3,256,320、音频分支2,777,859。无memory/router/KV cache/gradient checkpointing。

| 参数 | maxLR | AdamW decay |
| --- | --- | --- |
| 循环层5–24基础参数，包括两种norm | 6e-4 | 矩阵0.01；bias/一维0 |
| 两套LoRA和全部新增音频参数 | 6e-4 | 同上 |
| 原mapper、prelude/coda、embedding、最终norm/lm_head | 3e-4 | 同上 |
| CED | 冻结 | 不进入optimizer |

tied embedding/lm_head只计入一份优化器参数。按角色细分组，保留参数名、独立数量、maxLR和decay，写入`config_audit.json`与完整checkpoint。

数据固定为S1 hybrid **1,477,492**条、原八来源混合与原manifest/TSV。3epochs；**8 × microbatch4 × GA4 = GBS128**。每epoch11,542次更新，舍弃116条；总计**34,626**次更新，warmup**1,731**。原AdamW、5%warmup+cosine、10%LR下限、GradNormTracker及optimizer→scheduler顺序保留。第一步实际LR为maxLR的10%。

FP32参数/前向，无autocast、无TF32，保留参考GradScaler。音频在线解码为mono、resample32k、先整段center crop/right pad到20秒，再分成两个10秒；完整20秒进入CED。序列为126audio+1SEP+256prompt+300answer=683。PAD17、EOS/SEP0、LM `attention_mask=None`；prompt/answer右padding与右truncation，原EOS追加策略保持。

answer-only CE保持参考定义：**每个rank的每个microbatch先按自己的有效answer token求均值，再平均四个microbatch，最后由DDP平均八个rank**。有效目标同时满足answer attention mask和非PAD17；保持原next-token shift。没有改成按全局有效token加权。

## 边训练边 staging

1. 登录节点的CPU索引保存原始录音路径、临时本地相对路径及原始行顺序。计算节点加载索引，不重新扫描整个音频树，不预解码或缓存CED特征。
2. 计算节点只启动一个后台raw-byte producer，默认4个copy worker。检查tmpfs/cgroup容量，保留32GiB余量；发布本次分配的metadata和`TRAINING_READY`，`initial_steps=0`，随后复制全部原始文件。
3. `torchrun`只等待本次metadata可用；**训练直接读取原共享绝对路径**，不等待音频prefill/watermark。构造/审计模型同时，后台复制持续进行。
4. 每个文件先写`.partial`，核对来源size/mtime、复制字节数，再原子改成最终文件名。全部文件和字节计数符合计划后，producer发布`COMPLETE`。
5. 每个optimizer step开始、取四个microbatch之前，所有rank通过`all_reduce(MIN)`共同判断完整复制状态；全部确认后一起切到本节点`/dev/shm/.../audio/`。同一optimizer step内部不会切换。
6. producer异常、心跳过期、计数/所有权异常均中止运行。退出时先停止并reap producer，再只清理本次token所拥有的tmpfs根。共享原始数据不参与清理。

这沿用参考路线的size/mtime和字节数复制校验；不对每个音频额外计算SHA256。日志`AUDIO_READ_SWITCH`记录切换边界，metrics记录`audio_read_mode`、共享/本地读取数量、`staging_complete`和`local_switch_step`。staging进度在`staging_progress.json`及作业日志中。

## 输出与完整恢复

默认输出root：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_official_final_s1_gbs128_5_20x2_5_5090/formal_<timestamp>/
```

其下`formal/`保存`runtime.json`、`config_audit.json`、`trainable_params.txt`、`metrics.jsonl`、各rank首步/门控打开后梯度审计、`training_report.json`以及`checkpoints/`。作业日志位于仓库`code/RSmol/log/mizar_official_final_s1_gbs128_5_20x2_5/`。index-cache、tmpfs、route身份和resume均与旧路线隔离。

六个半epoch保存点：**5771、11542、17313、23084、28855、34626**，另保存`model--epo-3.ckpt`。checkpoint包含全模型、AdamW moments、scheduler、GradScaler、GradNormTracker、八rank RNG、epoch/下个optimizer step游标及代码/完整初始化资产/数据/配方合同。

恢复只接受本路线、同代码/依赖/资产/数据/配方的完整正式checkpoint，继续使用原output root。旧模型、转换产物和其他路线的checkpoint不能当full resume。恢复时重新建立当前节点的staging producer，先读共享数据再共同切换；不沿用旧节点的本地路径或旧report。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_official_final_s1_gbs128_5_20x2_5_5090.sh \
  --resume-checkpoint /path/to/original_output_root/formal/checkpoints/model--step-5771.ckpt \
  --stage-workers 4
```

`--output-dir`省略时从checkpoint自动推导原root；指定时必须与之相同。若fresh曾使用自定义`--model-dir`，恢复也传同一目录。正式训练已经完成的34626-step/epoch3 checkpoint拒绝继续训练。

恢复前检查参数名/顺序、AdamW groups/LR/decay、moments形状/精度/有限值、scheduler和GradNorm游标、八rank RNG及tied权重一致性；每rank恢复自己的RNG。DataLoader用独立CPU generator、0worker与明确sampler游标，避免重放已消费行和额外消耗模型RNG。日志可包含中断后重算区间，checkpoint游标是恢复依据。

## 手动验证与范围

仓库根目录运行：

```bash
python -B tests/validate_mizar_official_training.py
# 远程已有依赖环境中，要求真实小尺寸CPU autograd/optimizer/resume测试，禁止任何tensor skip：
python -B tests/validate_mizar_official_training.py --require-tensors
```

验证包含真实文件/后台线程staging协议、旧节点report拒绝、半epoch游标、所有学习率位置、原母本/转换包50文件SHA保护、数据/loss/clipping AST对照、真实Bash假提交和被阻塞copier夹具。小尺寸tensor测试覆盖实际50层调用、双LoRA梯度解锁、优化器唯一分组与norm LR、answer shift/mask/均值、AdamW/scheduler/GradNorm/CPU RNG精确恢复。假scheduler/trainer只用于Bash控制流测试，不登记为真实训练。

本机缺少Torch/Transformers/audio依赖时，真实tensor项明确SKIP。报告为仓库`audit_reports/mizar_official_training_validation.json`；`--require-tensors`遇缺依赖或skip即失败。上述本地检查不等于远程完整权重、真实八卡GPU、显存、收敛或完整恢复已通过。

既有转换包及产物的`training_intent.implemented=False`保留原样，表示转换时状态。实际trainer和训练合同由本目录负责；不向转换包添加文件，不使已生成产物的实现身份失效。既有MMAU/MMAR evaluator接入本训练checkpoint需另行适配。
