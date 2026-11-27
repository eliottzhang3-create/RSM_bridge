# Mizar L10–14×3：A9音频重构、三个循环偏置、训练L15，strongAC单轮

按用户要求在原入口 `run_mizar_l9_12x3_strong_reason_5090.sh` 与既有包内修改，保留历史文件名。当前配方已经是 **L10–14×3 / shared audio gate / bias3 / train L15 / strongAC only**。本地Windows只改代码，经GitHub同步；正式音频、历史hybrid、生产manifest及训练输出在远程Linux。不会自动commit/push或提交GPU。

## 模型与精确公式

完整继承 `/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt` 的原454状态，包括CED、已训练projector、全部30层文本基座、embedding/tied head及最终norm。18个release文件逐项核验大小/SHA；严格加载完成之后才注册新增模块，避免重置任何已训练权重。

按原始层号、从1开始执行：

```text
L1..9 → [L10..14] × 3 → L15 → L16..30 → final norm
```

共40次block调用，L10..14五层共享三轮；L15不循环、只执行一次并训练。其余参数冻结并eval，冻结后缀保留输入梯度，三轮完整BPTT。无KV缓存/decoder checkpointing/跨forward状态；final norm仅在L30后执行一次。

令H9为第9层raw post-block输出，A9=H9的前126个audio tokens，Yr为第r轮五层循环输出。三个独立576维向量b1/b2/b3从0初始化，存储为一个`loop_bias[3,576]`参数的三行。共享`Linear(576,1)`的weight/bias全0初始化，每个音频token生成一个tanh标量权重。

```text
第1轮：X1 = H9 + b1
       Y1 = Loop(X1)

第r轮（r=2,3）：
       Xr_audio = A9 + tanh(Linear(Y(r-1)_audio)) × Y(r-1)_audio
       Xr_other = Y(r-1)_other
       Xr = concat(Xr_audio, Xr_other) + br
       Yr = Loop(Xr)

之后：L15(Y3) → L16..30 → final norm
```

先以未加本轮偏置的上一轮音频输出计算门控，再重构音频，最后给全部token加当轮偏置。A9取自加b1之前的原L9输出；门控两处共享。偏置覆盖音频、SEP、prompt、answer及PAD位置。初始零门控使第二、三轮音频从`A9+br`开始；文本沿用前一轮输出再加br。

| 可训练参数组 | 参数量 | max_lr | min_lr |
|---|---:|---:|---:|
| 三个偏置＋共享gate | 2,305 | 5e-5 | 5e-6 |
| 共享L10..14 | 17,700,480 | 1.5e-5 | 1.5e-6 |
| 非循环L15 | 3,540,096 | 5e-6 | 1e-6 |
| 总计 | **21,242,881** | | |

原始冻结参数为138,059,264。新增参数只有1728偏置+577gate；状态项从454变为457（gate2、bias矩阵1）。原Llama MLP `gate_proj` 正常保留，与新增audio_gate不同。

## 数据：历史实际使用的strongAC全部视图

从旧完整训练实际使用的 `/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json` 流式筛选 `source=audiomcq` 且 `audio_contribution=strong`，保留直接答案`audiomcq_mcq`与额外CoT`audiomcq_cot`原行；删除全部weak和ReasonAQA，不抽样、不改字段/答案/原顺序，不去掉同音频的不同训练视图。

旧hybrid来源总数仍严格核验为1,477,492行；选中的strong数量在远程实际测得。原始未过滤S1 strong上界257,892直接答案+67,756 CoT=325,648，**不是此次强制配额**。历史缺失引用不重新引回。

沿用 `outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv` 中原成功路径；选中音频现在缺失/空或映射冲突就失败，不猜测替换、不静默丢行。weak/ReasonAQA音频不stat/staging。CPU准备只解析JSON/TSV/stat，不解码、不复制音频、不运行模型。

默认新目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/l10_14x3_strong_a9gate_bias3_v1/
  train.jsonl          仅strongAC直接答案和CoT原行
  selection.jsonl      原hybrid行索引/原行SHA/视图/strength/旧成功路径
  resolved_paths.tsv   仅本次选中音频的历史映射
  dataset_report.json 实际各源/视图/strong计数、排除weak/Reason数、音频数/字节数
  manifest.json        输入身份、实测配额、四文件SHA/大小与seal
```

输入hybrid和TSV前后SHA复核；独占准备锁、同级building目录、整体no-replace发布。相同输入可幂等复用；其它数据配方拒绝覆盖。扫描每10万行打印扫描/保留/排除数量，末尾映射每2万引用打印，失败保存failure/unresolved_audio。旧manifest/旧失败目录/旧训练输出保留。

## 训练与保存

- 1个完整epoch，8GPU × micro4 × GA4 = GBS128。
- N为manifest实测行数，总更新S=`ceil(N/128)`；最后不足128条全部训练，损失除以实际GA，不drop或padding重复。空rank仅零损失上下文参加DDP通信，不获得重复任务梯度。
- 前100个实际optimizer updates warmup，第100步达到各组max；随后各组分别cosine，最后真实更新达到各自min。保留参考warmup的0.1×max下限；生产总步数大于100。小测试配方才将warmup缩到total-1。
- L15的min/max比为0.2，其余为0.1，scheduler/恢复校验分别处理，不能套同一个固定比例。
- 每0.25epoch后的首个完整optimizer step保存：`ceil(S/4), ceil(S/2), ceil(3S/4), S`，最多保留最新4个完整checkpoint。
- 保存model/AdamW/scheduler/GradScaler/GradNormTracker/每卡RNG/游标，冻结检查通过后原子发布；发布成功才清理较早checkpoint。

AdamW betas(.9,.999)、eps1e-8、WD.01、bias/一维免decay保持；loop_bias矩阵虽然二维，也是偏置，不decay。FP32/无autocast/TF32off、原GradNormTracker/GradScaler保持。

fixed20保持mono32k、长音频中心裁剪20s、短音频右补零，然后分两个10s块，在线CED/projector。文本126audio+SEP1+prompt256+answer300=683；PAD17/EOS0、空格EOT后截断、不补回EOS、因果attention中PAD可见保持。answer-only shifted CE，logits382..681对应answer300，每卡micro有效token均值→实际GA均值→DDP/非空rank均值。

## 远程执行：同一个原入口

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# 仅CPU生成/验证新的strongAC manifest，不申请GPU
bash code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh --prepare-only

# 纯打印配置，不激活环境、不读数据、不提交
bash code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh --check-args

# 复用已验证manifest，CPU index/release/admission通过后单次提交正式训练
bash code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh
```

也保留独立CPU包装器：

```bash
bash code/RSmol/prepare_mizar_l9_12x3_strong_reason.sh
bash code/RSmol/prepare_mizar_l9_12x3_strong_reason.sh --verify-only
```

CPU包装器支持`--source HISTORICAL_HYBRID --resolved-paths TSV --output-dir DATA_DIR`；正式入口支持`--source`、`--resolved-paths`、`--data-dir`、`--model-dir`、`--output-dir`、`--index-cache-root`、`--stage-workers`及resume。CPU失败不会调用vc。正式manifest因原hybrid/音频位于远程，需由上述CPU命令在远程生成，本地交付的是已验证的生成实现。

资源维持pdgpu-5090/8GPU/32CPU/256G/单节点，镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`，conda `mellow_c8204d8`、Transformers4.54.1。原Miniconda路径及`RSMOL_CONDA_BASE`/`MELLOW_CONDA_BASE`覆盖保持，不改共享环境，无额外GPU smoke。

作业内一个后台producer，仅stage本次strong去重原音频到`/dev/shm/mizar_l10_14x3_strong_<token>`。先读共享盘；全部目标文件、字节数与源stat验证完成，八rank在optimizer边界MIN集合确认后一起切本地。producer/心跳/容量/复制异常会失败；退出先stop/wait回收producer再owner cleanup。resume重新staging。

默认输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
  mizar_final_l10_14x3_strong_a9gate_bias3_l15_gbs128_5090/
    formal_<tag>/formal/
    index_cache/<signature>/
```

`--output-dir`是外层run root，内部创建formal/；新训练要求空目录。最终`training_report.json`为PASS、epochs1、optimizer_steps等于sealed总步数、完整覆盖及checkpoint重读才表示训练完成。

查看最新正式日志：

```bash
log_file=$(ls -t code/RSmol/log/mizar-strong-l10-14x3.*.log | head -n 1)
echo "$log_file"
tail -n 100 -f "$log_file"
```

提交同时打印确切log_path、manifest、admission和output。`formal/metrics.jsonl`逐步记录loss/实际batch/epoch_fraction，以及`lr_used_by_group`、`lr_next_by_group`三组学习率。另有原参数冻结、各层/gate/三bias梯度检查、音频切换、每卡覆盖与错误traceback报告。final checkpoint为`model--step-<S>.ckpt`，S见runtime或提交前admission geometry。

## 完整恢复与旧配方

本版从官方Mizar初始化；旧x4完整AudioMCQ/旧三轮gate实验checkpoint不兼容。新数据、源码、模型、分组LR与输出身份均独立，拒绝跨配方resume。

本配方完整恢复使用原外层run root与最新完整checkpoint，恢复全部优化器状态/组身份/LR、scheduler、scaler、tracker、每卡RNG、精确样本游标。日志较新行/partial tail先原样备份再回退；损坏完整行拒绝。已完成epoch的checkpoint不续训。

```bash
bash code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh \
  --resume-checkpoint /绝对路径/原runroot/formal/checkpoints/model--step-保存步数.ckpt
```

省略output-dir时自动推导原root；源码/输入/环境版本变化会拒绝resume。旧模型评测路线不改，本版不适用于旧四循环checkpoint解释。

## 验证及本地Git提交

当前报告`audit_reports/mizar_l10_14x3_strong_gate_bias_validation.json`；旧报告保留其历史配方含义。测试入口仍为`tests/validate_mizar_l9_12x3_strong_reason.py`，从tests/运行避免既有code包遮蔽stdlib。

测试覆盖真实30层小Llama eager/SDPA原生30层对照、40调用/共享gate两次/三个偏置顺序、零门控音频重启和文本carry、非零gate独立公式、完整BPTT/L15/冻结后缀梯度/冻结state不变；生产meta参数21,242,881；strong-only CLI/原路径/排除Reason和weak/幂等seal/index；任意尾批分片/实际trainer单epoch恢复最终权重一致；三组100warmup/各自终点/optimizer实际LR；checkpoint组/矩/冻结篡改拒绝及最多4份；真实两进程Gloo正常GA与空rank零任务权重；Bash参数/CPU失败不提交/producer清理时序。

另以已有官方18文件实际SHA和454严格继承，原576维native单遍383prefix逐位一致、真实CED/projector+683token三轮前向/反传及AdamW冻结保持。音频为显式合成静音，不是训练性能指标。本地Torch2.15 CPU/Transformers4.54.1，既有torchaudio2.9未用native扩展ABI不匹配，仅测试进程显式禁用此扩展，实际Python音频变换/模型执行，生产无此适配。Git Bash生命周期夹具显式模拟flock/vc/torchrun。

未验证远程strong最终行数/音频齐备、CUDA/NCCL/5090显存吞吐或正式收敛。历史原始325648是上界，不能当远程实际数据数。

本地PowerShell（尚未自动执行）：

```powershell
git add RM.md code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh code/RSmol/mizar_l9_12x3_strong_reason tests/test_mizar_l9_12x3_strong_reason.py tests/test_mizar_l9_12x3_previous_training.py tests/validate_mizar_l9_12x3_strong_reason.py audit_reports/mizar_l10_14x3_strong_gate_bias_validation.json
git diff --cached --check
git diff --cached --stat
git commit -m "Train strongAC with gated L10-14 recurrence biases and L15"
git push origin HEAD
```

`sync_files.txt`列全部交付文件。仅修改用户指定入口/其包、对应测试、RM/文档；其他训练/评测与共享音频映射实现不改。保留原LICENSE/NOTICE及许可，RM只前置记录并保留全部旧字节。
