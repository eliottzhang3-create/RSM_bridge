# CED-small / 非 MeSH 5＋10×2＋5：init0904 音频训练

从 `mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small` 完整复制隔离。
训练语义只更换文本初始化来源，训练更新、音频、mapper、数据和优化器实现保持母本不变。
本地 Windows 维护代码，通过 GitHub 同步；权重、数据及真实训练在远程 Linux。

## 初始化及隔离范围

默认文本 checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-maxlr8e-4-resume4500-20260904_154711/checkpoint-step-009244
```

直接使用 `recursive_model_5_10_5.RecursiveLlamaForCausalLM`，严格加载完整 LM。
沿用 hidden576、20 个独立物理层、30 次执行和 `0..14,5..14,15..19` 轨迹；
不需要重新转换文本模型。config、tokenizer、原抽层映射和权重加载检查全部保留。
tokenizer 从同一文本 checkpoint 加载，执行原 Mizar 的 `!` PAD 注册并检查词表不增长。

首次训练只继承文本权重和 tokenizer，不读取文本 `training_state.pt`，音频训练从第 0 步开始。
CED-small 从原资产严格加载、冻结且保持 eval；官方 `freq_merge_126` mapper 按原顺序随机初始化，
mapper 和完整 LM 均训练。optimizer、scheduler、GradScaler、GradNormTracker、RNG、游标重新建立。

独立身份：

| 项目 | 本路线 |
|---|---|
| route | `ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_init0904_v1` |
| 提交入口 | `run_mizar_s1_gbs256_recursive_5_10_5_ced_small_init0904_5090.sh` |
| 输出根目录 | `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_init0904_5090/` |
| 索引缓存 | 上述输出根目录的 `index_cache/` |
| 临时音频目录 | `/dev/shm/mizar_s1_gbs256_recursive_5_10_5_ced_small_init0904_<本次token>` |
| 日志 | `code/RSmol/log/mizar_s1_gbs256_recursive_5_10_5_ced_small_init0904.<timestamp>.JOB.log` |

母本、其他训练路线、共享递归模型和现有评测器不改。
`source_guard.json` 记录隔离时 38 份母本/参考文件的归一化 LF SHA256。
`parent_reference_audit.json` 是原路线的历史审计记录，其 PASS 与源码身份仅指原路线；
本路线的本地验证报告另见 `audit_reports/mizar_s1_ced_init0904_validation.json`。

## 保持母本训练设置

| 项目 | 设置 |
|---|---|
| 数据 | hybrid 1,477,492 行，含 MusicCaps 和全部 SpeechCraft splits；原 manifest / resolved_paths |
| 音频 | 原 fixed20：32kHz mono、中心裁剪/右补零；整段处理后进入冻结 CED-small |
| Mapper | 官方 CEDFrequencyMerger + CEDTemporalMapper，freq_merge_126，dropout0.05，resample=False |
| 序列 | audio126 + SEP1 + prompt256 + answer300，prefix383 / total683 |
| 文本与 attention | 原模板、EOS 后缀、固定 padding；PAD `!`（原词表ID17），EOS/SEP0；attention_mask=None / use_cache=False |
| Loss | answer-only；rank-local microbatch 有效 token 均值 → GA 均值 → DDP 均值 |
| Batch | 8 GPU × microbatch8 × GA4 = GBS256 |
| 时长 | 3 epochs，5771 updates/epoch，共17313，warmup866；每epoch丢弃116行 |
| 学习率 | LM max3e-4 / mapper max6e-4，原5% warmup + cosine，floor0.1 |
| AdamW | betas(0.9,0.999)、eps1e-8、WD0.01；bias和ndim≤1不衰减 |
| 梯度 | 原GradNormTracker，initial L2=1 / max=10 / overdrive2.5 / momentum0.995 |
| 精度 | FP32前向，无autocast，TF32关闭；保留原GradScaler |
| 随机性 | seed1234，模型rank seed1234+rank，原数据shuffle/sampler |
| 运行 | mellow_c8204d8，原镜像，pdgpu-5090 / 32CPU / 256G / 8GPU / 1node |

数据路径保持：

```text
manifest=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
resolved_paths=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
CED_MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small
```

登录节点沿用原CPU索引准备；计算节点 initial_steps=0，仅等本次 metadata/owner 就绪，
共享原始音频立即用于训练，同时单producer、默认4个worker后台复制raw bytes。
全部文件/字节完成后，八rank在optimizer边界、读取四个microbatch之前all_reduce(MIN)统一切换本地。
复制失败/producer死亡/心跳异常中止；退出先停止并reap producer，再清理本token拥有的tmpfs。
每次resume重新staging并从shared开始。直接提交一个formal，无自动smoke/tensor前置任务。

## 正式提交

本地代码提交推送到 GitHub 后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_init0904_5090.sh --stage-workers 4
```

默认已经指向指定 init0904 checkpoint，自动创建独立的 `formal_<timestamp>` 输出。
若需显式保存本次输出路径，可用：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_init0904_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_init0904_5090.sh \
  --output-dir "$RUN_ROOT" --stage-workers 4
printf 'Training output: %s/formal\n' "$RUN_ROOT"
```

首次训练使用新空输出目录，不传 `--resume-checkpoint`。原入口的 `--text-model-dir`、
`--ced-model-dir`、`--resolved-paths`、`--index-cache-root` 等参数完整保留。

## 产物和完整恢复

本次 run root 下：

```text
formal/runtime.json
formal/config_audit.json
formal/trainable_params.txt
formal/first_gradient_rank<N>.json
formal/staging_report.json
formal/metrics.jsonl
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/training_report.json
```

沿用六个半epoch保存点2886/5771/8656/11542/14428/17313及epoch3 final。
完整checkpoint保存全部模型、optimizer、scheduler、GradScaler、GradNormTracker、八rank RNG及明确训练游标。
恢复只接受本init0904路线、相同源码/资产/数据/依赖/配方，沿用原run root与原文本来源路径。
母本旧音频checkpoint和文本checkpoint均不能当本路线音频full resume。

将下面 `RUN_ROOT` 替换成实际被中断的本路线run root，checkpoint替换成实际保存点：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_init0904_5090/formal_ACTUAL_TIMESTAMP"
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_init0904_5090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

## 本地验证与可选张量检查

从本地仓库根目录执行：

```powershell
python -B tests/validate_mizar_s1_ced_init0904.py
```

本次本地验证：42项通过，3项真实CPU tensor检查因缺少Torch/Transformers等依赖跳过。
Windows沙箱的临时目录和Bash signal pipe受限后，经批准在沙箱外运行同一验证脚本通过。

检查38份母本SHA、复制源码受限差异、整个训练器/loader/数据/优化器一致性、
全部17314个LR位置、实际fresh/resume控制流、旧route/错身份拒绝、runtime写入，
以及Bash模拟vc提交、共享启动/后台复制/集体切换/失败清理。
五套隔离测试在独立Python进程运行，避免同名contract/models模块交叉导入。
本地缺少Torch/Transformers等依赖时真实CPU tensor检查明确skip，不能记为GPU训练PASS。

远程依赖完整时可手动运行小模型CPU张量检查，要求无skip：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
python -B tests/validate_mizar_s1_ced_init0904.py --require-tensors
```

无需生产权重/数据，不运行CUDA，不提交作业，不是formal前置门槛。
真实远程权重加载、八卡训练/恢复和收敛仍以实际作业报告为准。
现有MMAU/MMAR evaluator尚未登记本init0904训练合同，评测接入需另行处理。

## 本地提交代码

从本地仓库根目录执行，只暂存本路线文件；RM.md按块选择本次记录：

```powershell
git add -- code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_init0904_5090.sh code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_init0904 tests/validate_mizar_s1_ced_init0904.py tests/test_mizar_s1_recursive_5_10_5_ced_small_init0904_route.py tests/test_mizar_s1_recursive_5_10_5_ced_small_init0904_parity.py tests/test_mizar_s1_recursive_5_10_5_ced_small_init0904_offline_config.py tests/test_mizar_s1_recursive_5_10_5_ced_small_init0904_shared_start.py tests/test_mizar_s1_recursive_5_10_5_ced_small_init0904_tensor.py audit_reports/mizar_s1_ced_init0904_validation.json
git add -p -- RM.md
git diff --cached --check
git diff --cached --stat
git commit -m "Add isolated CED audio training from init0904 text checkpoint"
git push
```
