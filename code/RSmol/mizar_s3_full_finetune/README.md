# Mizar final → 完整 S3 池，projector + 全 LM，0.5 epoch

独立入口：`code/RSmol/run_mizar_s3_full_finetune_3090.sh`。
训练/恢复/日志/原音频后台复制由已有 `mizar_l14_17x3_strong_reason` 复制后修改；
模型文件来自已有标准 Mizar CED 路线，执行原生 30 层各一次。
没有修改这些父路线或此前的 `mizar_s3_data_prepare`。来源文件及 LF SHA 见 `parent_sources.json`。
本地 Windows 编辑，通过 GitHub 同步；生产权重、数据、输出均留在远程 Linux。

## 固定实验配置

- 初始化完整 `/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt`。
  先用本地离线资产建立结构，再严格覆盖全部 454 个 state entries；核对名字、形状、dtype、有限值、tied values 和逐值继承。
  `release_manifest.json` 固定官方 final、CED、文本模型和 tokenizer 文件的 SHA。
- 冻结 `audio_encoder.base.ced.*` 并始终 eval；完整 `audio_encoder.base.mapper.*` 和
  `caption_decoder.lm.*` 可训练，包含 30 层、embedding、final norm 和 tied head。
  不增加层、循环、LoRA 或门控。训练参数表输出 `trainable_params.txt`。
  实测 mapper3,365,952 + LM134,515,008 = **137,880,960** 可训练参数；CED21,418,880 冻结。
- FP32，关闭 autocast/TF32/KV cache/gradient checkpointing；保留既有 GradScaler 与 GradNormTracker。
- AdamW：所有可训练参数统一 max LR `5e-6`、min `1e-6`，betas=(.9,.999)，eps=1e-8，
  weight decay=.01，bias/ndim≤1 不衰减；53 步 warmup（5% round，启动下限 .1×max，即5e-7）后 cosine。
  最后一个实际 update 使用 min LR。
- `pdgpu-3090` / 8 GPU / 64 CPU / 256G / 1 node；micro4 × GA4 × 8 = GBS128。
  原镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`，`mellow_c8204d8`，Transformers4.54.1。
- 完整数据池 272,796 条，seed1234 对全池做一次全局 permutation，0.5 epoch 消费前 **136,398 个唯一行**。
  不额外生成采样 manifest，不按来源重采样，不做 Q4 选择。0.5 epoch 的预算意味着本次运行只遍历半池。
  1,065 个完整128批 + 尾批78 = **1,066 updates**；尾批各 rank 为 `[10,10,10,10,10,10,9,9]`、GA3。
  不用 DistributedSampler 补齐重复行，不丢弃尾批。
- 保存 step **267 / 533 / 800 / 1066**，保留最新4份；完整保存模型、AdamW moments、scheduler、
  scaler、tracker、8 rank RNG、数据/映射/源码/环境身份和样本游标。
- loss 为 answer-only shifted CE（含答案 EOS），每个 rank 的 microbatch 按有效答案 token 求均值，
  再按实际 GA 求均值，最后由 DDP 按活跃 rank 求均值；保留已有 trainer 的归一化语义。

## 输入与正式作业内检查

默认已发布目录：`/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s3_full_filtered_v1`。
用户提供的远程 `S3_DATASET_PASS` 为保留272796、排除17156：
AVQA26481、AudioCaps49486、Clotho1991、CompA-R48045、LP-MusicCaps-MTT7871、
MusicCaps631、SpeechCraft113975、TACOS24316。本路线严格核对这些数量。

以 `train.json` 的原 S3 metadata 为唯一题面/选项/答案来源，使用同目录 `resolved_paths.tsv`
按原 `filepath1` 解析实际音频。CPU 索引只增加 `s1_raw_paths/s1_original_paths/s1_source`
这三个沿用的作业内字段，不修改发布的 manifest 或答案。S3 中相对 S1 修订的 gold 必须保留。

提交前在登录节点核验已有 dataset seal、所有输出 SHA、覆盖关系、保留音频 stat；不重新解码或过滤。
随后检查**全池** prompt/answer token 长度，禁止截断，生成带 seal、tokenizer identity 和原音频映射的不可变 index。
可复用同一身份 index；数据、源码或 tokenizer 变化时使用新身份。CPU 预检失败不会调用 vc。

音频在线 SoundFile 解码 → resample32k → mono → first20s → 右补零640000；
两段10s在进入 CED 前重新拼接，保持完整音频 frontend，不缓存 CED 特征。
沿用后台原字节复制：共享原文件立即可读，全部复制完成后8 rank 在 optimizer 边界统一切换 `/dev/shm`。
复制全池唯一音频，容量需容纳原始字节并预留32GiB；不足直接报告错误。任务退出先停止并回收 producer，再清理本作业 tmpfs。

文本按官方 S3 小写：prompt256 无 EOS；answer 加 ` <|endoftext|>` 后固定96。
序列 audio126 + SEP0 + prompt256 + answer96 =479；prefix383，CE 使用 logits382..477。
PAD为既有 `!`/17，EOS/SEP为0；不新增词表项，causal PAD 可见，attention_mask=None。
正式作业开始时对8来源各一条真实音频前向，检查有限 CE、479训练与383/384生成的前两个预测 logits/argmax，
以及改变未来答案 token 不影响较早 logits；探针保持模型 RNG。每次更新检查 mapper/全30层梯度；
检查点核验 frozen state，最终重载 checkpoint 并核对精确样本覆盖。无单独 GPU smoke 作业。

跨序列长度的 FP32 logits 比较使用 `rtol=1e-4, atol=1e-4`，允许 CUDA 不同矩阵/attention
计算形状带来的小量舍入差异；仍严格拒绝 NaN/Inf、超差和任意 argmax 改变。
同长度的未来答案因果隔离继续逐值相等检查。每来源即时打印 `SOURCE_PROBE_PASS`，
记录前两次比较的最大绝对误差、RMS、容差比值和 argmax；同样写入 `config_audit.json`。

2026-11-28 的首个远程作业在 `source_probes`、首次 optimizer 更新前退出：
12/49152 个近零 logits 超过旧 `atol=1e-5`，报告最大超差约1.475e-5。
修复探针后重新提交到默认新输出目录，不传 `--resume-checkpoint`；无需重新准备过滤数据。
训练源码身份变化会自动生成新索引，CPU 会重新核验 seal/映射/文本，不重复音频解码审计。

随后按用户要求将 max LR 调为5e-6、warmup调为5%（53步）；min LR保留1e-6。
使用默认新输出从官方 Mizar final 重新初始化；旧配方 checkpoint 不作为本配方的 resume 来源。

## 本地 Git 同步

在 Windows 仓库根执行，文件清单只包含本路线及其已有 S3 数据准备依赖：

```powershell
git add --pathspec-from-file=code/RSmol/mizar_s3_full_finetune/sync_files.txt
git diff --cached --stat
git commit -m "Add isolated Mizar full S3 half-epoch finetuning"
git push
```

## 远程提交

登录 Linux，在仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s3_full_finetune_3090.sh
```

入口会打印实际 log、outer output root、admission report 与 final checkpoint。
默认输出 `outputs/RSmol/mizar_final_s3_full_finetune_halfepoch_gbs128_3090/formal_<timestamp>/formal/`。
最终 checkpoint 为 `formal/checkpoints/model--step-1066.ckpt`；
`formal/training_report.json` 的 PASS 才表示该次训练完成，CPU admission 不表示训练成功。

仅预览参数（不读生产资产、不提交）：

```bash
bash code/RSmol/run_mizar_s3_full_finetune_3090.sh --check-args
```

仅完整 CPU 预检（不提交 GPU）：

```bash
bash code/RSmol/run_mizar_s3_full_finetune_3090.sh --preflight-only
```

可用 `--output-dir /absolute/new/outer_run_root` 固定独立输出；
`--data-dir`、`--model-dir`、`--index-cache-root`、`--stage-workers` 支持显式覆盖。
Miniconda 可通过已有 `RSMOL_CONDA_BASE` 或 `MELLOW_CONDA_BASE` 覆盖。

中断后用最新完整 checkpoint 续训，入口自动推断**原 outer root**，数据/源码/环境保持一致：

```bash
bash code/RSmol/run_mizar_s3_full_finetune_3090.sh \
  --resume-checkpoint /absolute/original_outer_root/formal/checkpoints/model--step-533.ckpt
```

拒绝其他路线/已完成 checkpoint，拒绝有更晚完整 checkpoint 的旧恢复点，拒绝迁移到另一 output root。
恢复会保留并截断超出恢复游标的 metrics，备份旧日志；完整 RNG 与 optimizer cursor 一起恢复。

## 本地验证与边界

`tests/test_mizar_s3_full_finetune.py`：真实30层小 Llama eager/SDPA、479 CE/梯度、音频解码重采样、
完整生产尺寸 sampler、真实 trainer 中断恢复逐值比较、完整 checkpoint 拒绝项、两个真实 CPU Gloo 进程的 GA4/GA3、
S3 seal/index 夹具、Bash 单次提交/含空格参数/失败不提交/producer 回收。vc/torchrun 使用明确接口夹具。

`tests/validate_mizar_s3_full_finetune.py --official-bundle ... --reference-s3 ...` 另外检查真实官方454项继承、
真实 CED/projector/LM 的479前向反向及 AdamW，及本地全部289952参考候选的 token/EOS。
报告：`audit_reports/mizar_s3_full_finetune_validation.json`，保留失败历史。
从 `tests/` 运行；使用兼容 Torch、Transformers4.54.1、torchaudio、SoundFile 的 CPU 环境。

本地验证不表示远程生产音频、CUDA/NCCL、3090显存/吞吐或训练收敛通过。
训练期间无独立验证集评分；是否提升需对完成 checkpoint 另行做一致协议评测。
