# 原始 Mizar：L9–12 纯循环四次，完整 AudioMCQ，三轮训练

按用户要求保留入口 `run_mizar_l9_12x3_strong_reason_5090.sh` 与现有包目录名；当前配方已经是 **L9–12×4 / AudioMCQ only / 3 epochs**。本地只改代码，经 GitHub 同步；正式 manifest、音频、权重及输出在远程 Linux。不会自动 commit/push 或提交 GPU。

## 模型与训练配方

严格完整加载 `/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt`，包括原有 CED、已训练 projector、全部30层文本基座与最终 norm。18个 release 文件逐项核验 SHA/大小，原454状态全部继承。没有新增状态或参数。

```text
第1～8层 → [第9～12层] × 4 → 第13～30层 → final norm
```

四轮总共执行四次，每轮完整输出直接送入下一轮；共42次 block 调用。删除原A8注入、额外音频线性门控和tanh。原始Llama MLP的 gate_proj 是基座结构的一部分，正常保留。

只训练原第9～12层全部参数含norm，四轮共享同一份权重，共 **14,160,384** 个训练参数。其余 **145,139,456** 个原参数冻结且eval；冻结后缀保留输入梯度，四轮完整BPTT，无KV缓存、无decoder gradient checkpointing。原final norm只在第30层后执行一次。

| 设置 | 当前值 |
|---|---|
| 数据 | 完整历史AudioMCQ，720,524行/epoch；ReasonAQA=0 |
| epochs | 3 |
| GBS | 8 GPU × micro4 × GA4 = 128 |
| 每轮更新 | 5,630；5,629个完整128更新 + 12条尾批 |
| 总更新 | 16,890 |
| max_lr / min_lr | 2e-5 / 2e-6 |
| warmup | 三轮总步数round(3%)，507步 |
| scheduler | 全局单条warmup + cosine，最后真实更新到min，不逐epoch重启 |
| 保存 | 每轮0.25/0.5/0.75/1.0，共12个保存点，最多保留最新4份 |

每轮保存位置为第1,408、2,815、4,223、5,630个完整optimizer step，对齐四分之一轮后首个更新。全局保存步数：

```text
1408, 2815, 4223, 5630,
7038, 8445, 9853, 11260,
12668, 14075, 15483, 16890
```

最终保留12668/14075/15483/16890四份。checkpoint保存model、AdamW、scheduler、GradScaler、GradNormTracker及所有rank RNG，先校验冻结状态再原子发布，发布成功后才清理较早完整checkpoint。

每轮每个样本恰好训练一次；每轮按seed1234+epoch打乱分片，不drop、不padding重复。生产尾批前4卡各2条、后4卡各1条，GA=1。保留原每卡micro有效answer token均值→实际GA均值→DDP均值口径；若测试/未来尾批出现空rank，仅零损失上下文参加集合通信，不赋予重复任务梯度或消费记录。

AdamW其余参数保持betas=(.9,.999)、eps=1e-8、WD=.01，bias/一维参数免decay。FP32、无autocast、TF32off，参考GradScaler/GradNormTracker保持。fixed20保持mono32k、长音频中心裁剪20s、短音频右补零，再分两个10s块。在线真实CED/projector，不缓存特征。

文本保持126audio+SEP1+prompt256+answer300=683，PAD17/EOS0，追加空格EOT后截断、不补回被截断EOS，因果attention中PAD可见。answer-only shifted CE，logits382..681对齐300个answer位置；固定布局和源文本语义不变。

## CPU生成新的manifest

从旧完整训练实际使用的 `/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json` 中，保留所有 `source=audiomcq` 原行，去掉756,968条ReasonAQA。旧hybrid共1,477,492行；不从未过滤原始S1重新引入历史缺失音频。

strong/weak、直接答案/CoT等原有视图全部纳入，原字段、答案、重复训练视图、相对顺序均保持，不重新采样、不修改标签、不补音频。

| 来源 | 行数 |
|---|---:|
| AudioCaps | 113,844 |
| Clotho | 4,798 |
| MusicCaps | 3,221 |
| CompA-R | 249,161 |
| LP-MusicCaps-MTT | 19,632 |
| SpeechCraft | 287,727 |
| TACOS | 42,141 |
| 总计 | **720,524** |

复用旧 `outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv` 成功路径。旧用过的文件现在缺失/空或TSV冲突会报错，不重新猜映射、不静默过滤。ReasonAQA音频不会检查或staging。CPU准备只读JSON/TSV/stat，不复制音频、不解码、不加载模型。

默认新目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/l9_12x4_audiomcq_only_v1/
  train.jsonl          全部原始AudioMCQ行
  selection.jsonl      原hybrid行索引、原行SHA、视图/strength和真实音频路径
  resolved_paths.tsv   只含AudioMCQ引用的旧成功映射
  dataset_report.json 七来源/视图/strength计数、去掉的ReasonAQA数、文件数/字节数
  manifest.json        输入与四文件SHA/大小、固定新配方seal
```

流式扫描每100,000行打印已扫描/保留/排除数量，末尾验证映射每20,000个引用打印进度；结束后再次校验源输入SHA。独占准备锁、同级building目录、整体no-replace发布；同输入幂等复用，拒绝覆盖其它配方。失败保留failure/unresolved_audio诊断。旧strong+Reason manifest和旧失败目录保留。

远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# 同一个正式入口，先只生成/核验新manifest，不申请GPU
bash code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh --prepare-only
```

也可用保留的CPU包装器：

```bash
bash code/RSmol/prepare_mizar_l9_12x3_strong_reason.sh
bash code/RSmol/prepare_mizar_l9_12x3_strong_reason.sh --verify-only
```

CPU包装器支持 `--source HISTORICAL_HYBRID`、`--resolved-paths TSV`、`--output-dir NEW_DATA_DIR`。当前source指历史hybrid，不是原始未过滤S1。生产数量固定720524，数量变化会拒绝。

## 正式提交与日志

```bash
# 仅打印参数与新配方，不激活环境、不读数据、不提交
bash code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh --check-args

# 自动核验/复用新manifest、准备CPU index、校验完整release，再单次vc submit
bash code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh
```

队列pdgpu-5090、8GPU、32CPU、256G、单节点；镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`，原conda `mellow_c8204d8`，Transformers4.54.1。保留原Miniconda路径和 `RSMOL_CONDA_BASE` / `MELLOW_CONDA_BASE` 覆盖，不修改共享环境，不额外提交GPU smoke。

同一作业一个后台producer，仅将此manifest去重原音频staging到 `/dev/shm/mizar_l9_12x4_mcq_<token>`。先读共享盘；全部文件/字节/源stat校验完成后八rank在optimizer边界MIN集合判定，一起切换本地。三轮复用一次staging；恢复作业重新staging。退出先stop/wait回收producer，再按root/token清理。容量/producer/心跳/复制失败终止并保留诊断。

新默认训练输出与index根：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
  mizar_final_l9_12x4_audiomcq_gbs128_3epochs_5090/
    formal_<tag>/formal/
    index_cache/<signature>/
```

日志：`code/RSmol/log/mizar-audiomcq-x4.<tag>.<JOB>.log`（JOB由vc展开）。提交打印log_path、manifest、admission和输出目录。查看最新正式日志：

```bash
log_file=$(ls -t code/RSmol/log/mizar-audiomcq-x4.*.log | head -n 1)
tail -n 100 -f "$log_file"
```

`formal/metrics.jsonl`逐步记录epoch/epoch_update/epoch_fraction、loss、lr_used/lr_next、尾批实际大小和读盘状态。各epoch/rank覆盖检查、冻结/梯度审计、staging进度均落盘。最终 `training_report.json` 要求PASS、epochs3、16890steps、完整三轮覆盖和最终checkpoint重读一致。最终文件 `formal/checkpoints/model--step-16890.ckpt`。

自定义输出示例：

```bash
bash code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audiomcq_x4_3ep_v1
```

`--output-dir`始终是外层run root，内建formal/。另支持 `--data-dir`、`--source`、`--resolved-paths`、`--model-dir`、`--index-cache-root`、`--stage-workers`。manifest CPU失败不会提交vc。

## 完整恢复

新实验从官方Mizar初始化。旧三轮门控/混合Reason实验的checkpoint不兼容新模型、数据、LR或训练身份；拒绝跨配方恢复。

当前配方恢复必须使用原run root及最新完整checkpoint，恢复所有优化器/调度/GradScaler/GradNormTracker和每卡RNG。epoch边界正确从下一epoch第0步开始，epoch内从本轮已消费位置继续；LR不重置。支持原目录日志较新行/partial tail备份回退，完整坏行拒绝。源码/数据/环境身份必须相同。

```bash
bash code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh \
  --resume-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audiomcq_x4_3ep_v1/formal/checkpoints/model--step-5630.ckpt
```

省略output-dir时从checkpoint推导原外层root。已完成16890步checkpoint拒绝续训。

## 本地验证和Git同步

当前报告 `audit_reports/mizar_l9_12x4_audiomcq_validation.json`。历史两份报告保留作为旧配方记录。测试仍使用现有tests文件名。

验证内容包括：真实小型30层Llama eager/SDPA native单遍一致、42次调用/四轮输入直接衔接、四轮BPTT/因果性/冻结保持；生产meta训练参数14160384；manifest CLI/原路径/强弱及CoT全部保留/零Reason/幂等seal和index；720524行三轮八分片覆盖及尾批；507warmup/12保存点/保留4份；真实完整小模型三轮训练及epoch边界恢复最终bitwise一致；真实2进程CPU Gloo GA4/GA1更新对照；Bash prepare-only/失败不提交/带空格参数链/producer先stop-reap后cleanup。

另使用已有官方完整18文件验证454状态严格加载、实际CED/projector、原576维30层native对照、683token四轮前向/反传/AdamW后冻结状态保持。音频为明确的合成静音，不代表性能或收敛。

本地Torch2.15 CPU/Transformers4.54.1；已有torchaudio2.9未用native扩展ABI不匹配，仅测试进程显式禁用此扩展，Python音频变换/模型仍实际执行，生产代码无适配。Git Bash无flock，生命周期夹具模拟flock/vc/torchrun。正式远程hybrid/音频未在本地生成；精确720524生产manifest由远程CPU命令生成。CUDA/NCCL、5090显存吞吐和正式收敛待远程确认。

测试从tests/执行，避免仓库code包遮蔽stdlib：

```bash
cd tests
python validate_mizar_l9_12x3_strong_reason.py --official-bundle /path/to/Mizar159M
```

本地PowerShell提交（未自动执行）：

```powershell
git add RM.md code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh code/RSmol/mizar_l9_12x3_strong_reason tests/test_mizar_l9_12x3_strong_reason.py tests/test_mizar_l9_12x3_previous_training.py tests/validate_mizar_l9_12x3_strong_reason.py audit_reports/mizar_l9_12x4_audiomcq_validation.json
git diff --cached --check
git diff --cached --stat
git commit -m "Train full AudioMCQ with four shared Mizar loops for three epochs"
git push origin HEAD
```

`sync_files.txt`列全部交付文件。修改局限于现有指定入口/包及其测试与文档；其它训练路线、原音频映射实现/规则保持。保留原LICENSE/NOTICE及依赖许可。RM只前置新记录，原全部字节保持。
