# 官方 Mizar final：5 + 20×2 + 5，仅 input 重注入，S1 GBS128 正式训练

本目录从已有 `mizar_official_final_s1_gbs128_5_20x2_5` 训练器复制隔离。使用独立 input-only 模型、配置合同、output、log、job、index-cache、tmpfs 和恢复身份。Windows 本地维护代码，GitHub 中转；权重、数据和正式执行留在远程 Linux。

## 转换与正式提交

先使用 [CPU 转换实现](../../mizar_official_recursive_5_20x2_5_input_only/README.md) 从官方 Mizar final 创建独立初始化目录。本地代码提交推送后，远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5_input_only.sh convert --seed 0 --threads 8
```

确认转换命令成功后提交正式训练：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_official_final_s1_gbs128_5_20x2_5_input_only_5090.sh --stage-workers 4
```

默认初始化目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-5-20x2-5-input-reinject
```

提交入口先在登录 CPU 做只读产物文件审计，并构建/复用独立路径索引；随后直接 `vc submit` 一个正式训练任务。资源沿参考 `pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`、原容器和 `mellow_c8204d8`。没有自动 smoke/tensor 前置作业，也不自动安装依赖。CPU 转换和训练是两个独立命令；转换不申请 GPU，训练在计算节点用八卡运行。

可选参数：`--model-dir DIR`、`--output-dir DIR`、`--resume-checkpoint PATH`、`--resolved-paths TSV`、`--index-cache-root DIR`、`--stage-workers N`。自定义转换输出时，提交训练须传同一 `--model-dir DIR`。

## 模型与优化器

严格加载完整 **458** 份 state tensor，继承官方 final 的完整 LM、mapper 和 CED，以及转换保存的 input projection/gate；构造时的临时初始化被完整 state 覆盖。保留 30 物理层、50 次调用、`0..24,5..24,25..29`，中段两轮共享权重。

每个 LM forward 只将原 mapper 音频 embeddings 投影一次；逻辑 5/25 的两次循环前，使用共享 gate 对当前音频 hidden 加权注入。没有 encoder 中间层注入、LoRA、memory 或 router。CED 最终特征每个音频批次提取一次，冻结且 eval；mapper 和完整 LM 继续训练。总独立参数 **159,632,769**，其中新增 **332,929**。GPU loader 不调用会清空 CUDA 环境的 CPU 转换入口。

| 参数 | maxLR | AdamW decay |
| --- | --- | --- |
| 循环层 5–24 基础参数，包括 norm | 6e-4 | 矩阵 0.01；bias/一维 0 |
| 共享 input projection 和 gate | 6e-4 | 同上 |
| 原 mapper、prelude/coda、embedding、最终 norm/lm_head | 3e-4 | 同上 |
| CED | 冻结 | 不进入 optimizer |

tied embedding/lm_head 只计入一份优化器参数。参数按六种角色细分，再按 decay 分组；组内保存参数名和顺序，完整检查覆盖及唯一性。

gate 从零开始，首步投影的 task gradient 为零，gate、mapper 和 LM 应获得 task gradient；第一次更新打开 gate 后，第二次更新检查投影也获得 task gradient。该首步行为和原 gate 设计一致。

## 保持原训练配方

- S1 hybrid **1,477,492** 条，原八来源混合、manifest 和 resolved-paths TSV；3 epochs。
- **8 × microbatch 4 × GA 4 = GBS128**；每 epoch **11,542** updates，舍弃 116 条；总计 **34,626** updates，warmup **1,731**。
- 原 AdamW、5% warmup + cosine、10% LR 下限、GradNormTracker 和 optimizer → scheduler 顺序。第一步实际 LR 是 maxLR 的 10%。训练 seed 1234，各 rank RNG 为 1234 + rank。
- FP32 参数与前向，无 autocast、无 TF32，保留参考 GradScaler；无 KV cache、无 gradient checkpointing。
- 原始录音在线 mono/resample32k，整段 center crop/right pad 到 20 秒，再拆两个 10 秒输入；完整 20 秒进入 CED。保持 `freq_merge_126`、mapper dropout 0.05、原 prompt/answer 模板及 EOS 追加策略。
- 序列为 **126 audio + 1 SEP + 256 prompt + 300 answer = 683**。PAD17、EOS/SEP0，右 padding/truncation，LM `attention_mask=None`。
- answer-only next-token CE：每 rank、每 microbatch 按自己的有效 answer token 求均值，再平均四个 microbatch，最后由 DDP 平均八个 rank；保持 attention mask 与非 PAD17 的联合有效目标条件。

默认 manifest 为 `/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json`，默认 TSV 为 `outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv`。复制的数据模板、解码工具和完整 microbatch 更新循环保留母本语义。

## 共享读取启动，后台 staging 后集体切换

1. 登录 CPU 准备路径索引，保持原行顺序、共享原始录音绝对路径与本地相对路径；不读取/解码整库音频。
2. 计算节点启动一个后台 producer，默认四个 worker 复制 raw bytes 到本节点独立 `/dev/shm`。检查容量并保留 32 GiB 余量。`initial_steps=0`，只等待本次分配的 owner/path metadata 和 `TRAINING_READY`。
3. **训练立即从共享原始音频路径读取，同时后台 staging 继续。** 不等音频 prefill/watermark，不预计算 CED features。
4. 单文件先写 `.partial`，核对来源 size/mtime 和字节数，再原子发布。整库文件数和字节数通过后才发布 `COMPLETE`。
5. 每次 optimizer update 开始、读取四个 microbatch 之前，八 rank 用 `all_reduce(MIN)` 共同确认完整复制状态，一起切到节点本地。一个 update 内不切换；不逐文件提前读本地副本。
6. producer 失败、死亡、心跳过期或所有权/计数不符时中止。退出先停止并 reap producer，再清理本 token 拥有的 tmpfs 根。

新根前缀为 `/dev/shm/mizar_official_final_s1_gbs128_input_only_5_20x2_5_`。日志 `AUDIO_READ_SWITCH` 记录切换时完成的更新数；metrics 保存 `audio_read_mode`、共享/本地读取数、`staging_complete` 和 `local_switch_step`。复制保持母本 size/mtime/字节数校验，不新增每个音频的 SHA256。

## 输出与完整恢复

默认输出 root：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_official_final_s1_gbs128_5_20x2_5_input_only_5090/formal_<timestamp>/
```

其下 `formal/` 保存 `runtime.json`、`config_audit.json`、`trainable_params.txt`、`metrics.jsonl`、各 rank 首步/门控打开后梯度审计、`training_report.json` 和 `checkpoints/`。作业日志在仓库 `code/RSmol/log/mizar_official_final_s1_gbs128_5_20x2_5_input_only/`。输出目录可由 `--output-dir DIR` 显式指定，fresh 拒绝非空目录。

六个半 epoch 保存点：**5771、11542、17313、23084、28855、34626**，另保存 `model--epo-3.ckpt`。均包含全模型、AdamW moments、scheduler、GradScaler、GradNormTracker、八 rank RNG、epoch/下次 update 游标及代码/初始化资产/数据/配方合同。最终为 `formal/checkpoints/model--step-34626.ckpt` 和 `model--epo-3.ckpt`。

恢复示例，替换为实际 checkpoint 路径：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_official_final_s1_gbs128_5_20x2_5_input_only_5090.sh \
  --resume-checkpoint /path/to/original_output_root/formal/checkpoints/model--step-5771.ckpt \
  --stage-workers 4
```

省略 `--output-dir` 时自动从 checkpoint 推导原 root；显式指定则必须一致。若首次使用自定义 `--model-dir`，恢复也传同一目录。只接受本 input-only 路线同代码、依赖、资产、数据、配方的完整正式状态；转换目录、原 serial84/LoRA 路线或其他实验 checkpoint 不能作 full resume。已完成全部 34,626 updates 的 checkpoint 拒绝继续训练。

恢复前检查参数名/顺序、LR/decay、moments 形状/精度/有限值、scheduler/GradNorm 游标、tied 值及八 rank RNG。DataLoader 采用独立 CPU generator、0 worker 和明确 sampler 游标；每 rank 恢复自己的 RNG。每次恢复都建立当前节点的新 staging，重新从 shared 开始，完成后再集体切换，不复用旧节点 report 或本地路径。

## 本地检查与验证范围

```bash
# 仓库根目录，手动诊断；不作为正式提交的自动前置任务。
python -B tests/validate_mizar_input_only.py
python -B tests/validate_mizar_input_only.py --require-tensors
```

当前本地 **63 项实际检查通过，12 项真实 tensor 检查缺依赖跳过**。包含父模型/训练入口等 45 文件 SHA 保护、复制模板/工具一致性、完整更新循环/保存 RNG/GradNorm AST 对照、全部 34,627 个 LR 位置、resume 拒绝条件、真实被阻塞 copier 线程的共享启动/完成切换，以及真实 Bash 假提交/资源/路径传参/失败清理。可选 CPU tensor 测试使用小 hidden 但完整 30 物理/50 逻辑层，检查 input 公式、梯度解锁、唯一优化器分组、严格重载、loss 和 AdamW/CPU RNG 恢复。

报告为 `audit_reports/mizar_input_only_validation.json`；`--require-tensors` 遇缺依赖或任何 skip 会失败。Bash 夹具只证明控制流，不能算真实训练。当前未执行远程生产转换、真实 GPU/DDP、显存/收敛或八卡恢复验证。旧 MMAU/MMAR evaluator 接入本路线需另外适配。
