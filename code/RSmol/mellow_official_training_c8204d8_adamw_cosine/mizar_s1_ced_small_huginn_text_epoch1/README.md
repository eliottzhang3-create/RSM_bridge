# Huginn 文本 epoch1 → CED-Small 音频 S1，GBS256

本地 Windows 编辑代码、GitHub 中转，全部权重、数据及训练输出保留在远程 Linux。
本目录复制自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small`；正式入口为
`code/RSmol/run_mizar_s1_ced_small_huginn_text_epoch1_5090.sh`。
默认直接使用用户指定的已训练 Huginn checkpoint，无需转换或先训练文本 epoch2：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_huginn_gbs1024/epoch1/checkpoint-step-009244
```

## 模型与初始化

`models/huginn.py` 复制已有 Huginn 文本模型，仅调整其依赖导入路径，避免独立音频目录运行时
与 Python 标准库 `code` 冲突。计算、参数名称及 checkpoint 格式保持一致。
显式 `MizarHuginnForCausalLM.from_pretrained` 严格加载全部 LM 权重；
缺失、额外、形状不匹配或加载错误均拒绝，不使用全局 AutoModel 注册。
JSON 检查要求本 Huginn 路线完整训练 step9244、20 物理层/30 次执行、hidden576/vocab49152、
既定选层及两轮参数共享，检查绑定 embedding/head 和唯一共享投影 A。

从同目录加载 tokenizer，按原音频代码将已有 `!` 注册为 PAD（通常 ID17），不扩词表；EOS/SEP=0。
只读取 HF 权重、配置和 tokenizer，不反序列化文本 `training_state.pt`。
音频 optimizer/scheduler/GradScaler/GradNormTracker/RNG/采样进度从 step0 重新建立。
**已训练 A 的参数完整继承，不重新初始化；所有 LM 参数包括 A 继续训练。**

```text
fixed20音频 → 冻结且eval的CED-Small → 原可训练freq_merge_126 mapper
                                          ↓
               X = [audio126, SEP1, prompt256, answer300]
               E = Prelude(X)                      # [B,683,576]
              S0 ~ ±3σ截断高斯，σ=√(2/5)            # 每forward一次
              S1 = Core(A(cat([S0,E], dim=-1)))
              S2 = Core(A(cat([S1,E], dim=-1)))
          logits = LMHead(FinalNorm(Coda(S2)))
```

A 为同一个 `Linear(1152,576,bias=False)`，共663,552参数，两次输入顺序均为 `[state,prelude]`；
第二轮不重新采样，S0只为临时激活。E和S1不detach，固定两轮完整反传。
保留原多模态前缀方式，无额外音频注入、MeSH、LoRA、辅助loss或其他新训练分支。
CED从 `models/CED_small` 加载，冻结全部参数及BN/eval状态；mapper沿母本构造顺序随机初始化。

## 严格保留的音频配方

| 项目 | 设置 |
|---|---|
| 数据 | 原hybrid 1,477,492条，同manifest、resolved_paths及模板 |
| 音频 | 原单声道/重采样/中心裁剪/右补零fixed20；完整20秒进入CED预处理 |
| CED/mapper | 原在线FP32 CED、freq_merge_126、dropout0.05、resample=False |
| 序列 | audio126+SEP1+prompt256+answer300；prefix383/total683 |
| LM调用 | `attention_mask=None`、`labels=None`、`use_cache=False` |
| Loss | answer-only；rank-local有效token均值→4个microbatch均值→DDP均值 |
| Batch | 8GPU×micro8×GA4=GBS256 |
| 训练长度 | 3epochs；5771updates/epoch；17313total；每epoch丢弃116条尾部 |
| LR | LM含A max3e-4；mapper max6e-4 |
| Scheduler | 原LambdaLR；866warmup；cosine；floor0.1；optimizer后step |
| AdamW | betas(.9,.999)、eps1e-8、矩阵WD.01；bias/ndim≤1不衰减 |
| 裁剪 | 原GradNormTracker：initial L2=1/max=10、overdrive2.5、momentum.995 |
| 精度 | 参数/前向FP32、无autocast、保留GradScaler、关闭TF32 |
| RNG/数据 | 原seed1234、rank seed1234+rank；原shuffle及连续rank采样；workers0 |
| 运行环境 | 原mellow_c8204d8、原镜像、pdgpu-5090/32CPU/256G/8GPU/1node |

A属于完整LM优化器组，使用3e-4和WD.01。音频阶段不沿用文本训练的LR、GBS或loss归一化方法。
保留原CED资产检查及CPU index准备后直接提交formal，无新增smoke/tensor测试前置。
共享音频立即开始训练，单producer默认4workers后台复制；全部完成后在optimizer边界经全rank MIN确认切换。
退出先stop/reap producer，再清理本token的tmpfs；每次恢复重新staging并从shared读取开始。
临时目录为 `/dev/shm/mizar_s1_gbs256_ced_small_huginn_text_epoch1_<token>`，
兼容原 `stage_data.py` 启动/清理共同要求的 `mizar_s1_gbs256_` 前缀，后缀及token保持路线隔离。

`train.py`、`optim.py`、原decoder、CED、mapper和全部data/index/staging实现逐字节继承。
改变集中在专用loader/合同、隔离命名和配置说明。35份母本/模型参考文件的LF SHA256见 `source_guard.json`。
本目录 `.gitignore` 放行 `source_guard.json` 与 `parent_reference_audit.json`，随代码一起同步。
`parent_reference_audit.json` 是母本历史审计，不能当作本路线生产验证。
`audit_reference.py` 可选比较此次直接母本，正式训练不调用它。

## 远程正式训练

本地提交并push后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_huginn_text_epoch1_5090.sh --stage-workers 4
```

默认输出根：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_huginn_text_epoch1_gbs256_5090/formal_<timestamp>/
```

也可以显式指定初始化和全新输出根，便于后续恢复；母本会在此根下追加 `formal/`：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
HUGINN_AUDIO_RUN=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_huginn_text_epoch1_gbs256_5090/audio_$(date +%Y%m%d_%H%M%S)
bash code/RSmol/run_mizar_s1_ced_small_huginn_text_epoch1_5090.sh \
  --text-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_huginn_gbs1024/epoch1/checkpoint-step-009244 \
  --output-dir "$HUGINN_AUDIO_RUN" --stage-workers 4
```

原始输入资产仍为：

```text
manifest=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
resolved_paths=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
CED=/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small
```

日志：`code/RSmol/log/mizar_s1_ced_small_huginn_text_epoch1_5090.<timestamp>.JOB.log`。
最终完整产物：

```text
<OUTPUT_ROOT>/formal/checkpoints/model--step-17313.ckpt
<OUTPUT_ROOT>/formal/checkpoints/model--epo-3.ckpt
<OUTPUT_ROOT>/formal/runtime.json
<OUTPUT_ROOT>/formal/config_audit.json
<OUTPUT_ROOT>/formal/training_report.json
```

## 同一路线完整恢复

音频checkpoint保存完整CED/mapper/LM（包括A）、optimizer、scheduler、scaler、tracker、8rank RNG及采样游标。
源码身份覆盖隔离Huginn模型、共享文本合同及官方源manifest；拒绝旧音频路线或不同源码/资产/数据/配方。
保留原音频 `1234+rank` 随机流，Huginn噪声和mapper dropout均由它驱动；source probes保护RNG。

恢复沿用**原音频run输出根**，不是新建文本epoch；先将变量设为实际已存在的run目录：

```bash
bash code/RSmol/run_mizar_s1_ced_small_huginn_text_epoch1_5090.sh \
  --output-dir "$HUGINN_AUDIO_RUN" \
  --resume-checkpoint "$HUGINN_AUDIO_RUN/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

## 本地验证与Git提交

```powershell
Set-Location C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
python -B tests/validate_mizar_s1_ced_huginn_text_epoch1.py
```

报告为 `audit_reports/mizar_s1_ced_huginn_text_epoch1_validation.json`。
2026-11-12 修复tmpfs前缀后，本地结果：46项通过，7项真实CPU张量用例因缺少Torch等依赖跳过。
覆盖35份参考源码SHA、完整音频配方和17314个LR位置、严格加载/step0初始化、跨路线恢复拒绝、
共享读取/后台复制/集体切换/退出清理、Bash模拟vc资源和参数，以及Git审计清单可提交性。
新增前缀回归从真实 `run.sh` 执行Bash路径赋值，交给真实 `validate_stage_root`/`cleanup`函数：
修复前重现启动/清理均拒绝，修复后通过；保留软链接/越界/错误token和root拒绝。
Windows上仅POSIX文件查询及删除使用fixture，不创建或删除真实 `/dev/shm` 目录。
旧模拟进程测试未验证目录前缀，现由此回归补齐；模型、checkpoint及训练配方均不变。
缺依赖时真实CPU张量用例明确跳过；它们检查完整多模态梯度、A继承与优化器更新、随机状态恢复等。
远程mellow_c8204d8中可手动运行 `python -B tests/validate_mizar_s1_ced_huginn_text_epoch1.py --require-tensors`，
仅用合成CPU模型，不读生产资产，不作为formal前置条件。源码/模拟检查不代表生产训练已验证。

工作区已有其他路线改动，按以下范围提交；`RM.md` 使用按块暂存仅选择本条Huginn音频记录，
如果与其他记录合在一块，用 `e` 编辑。先检查已有暂存区内容：

```text
git diff --cached --stat
git add -- code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_huginn_text_epoch1 code/RSmol/run_mizar_s1_ced_small_huginn_text_epoch1_5090.sh tests/test_mizar_s1_ced_huginn_text_epoch1_identity.py tests/test_mizar_s1_ced_huginn_text_epoch1_route.py tests/test_mizar_s1_ced_huginn_text_epoch1_offline_config.py tests/test_mizar_s1_ced_huginn_text_epoch1_shared_start.py tests/test_mizar_s1_ced_huginn_text_epoch1_staging_roots.py tests/test_mizar_s1_ced_huginn_text_epoch1_tensor.py tests/validate_mizar_s1_ced_huginn_text_epoch1.py audit_reports/mizar_s1_ced_huginn_text_epoch1_validation.json
git add -p -- RM.md
git diff --cached --stat
git commit -m "Add isolated Huginn text epoch1 CED audio S1 training"
git push
```

交付过程不代为commit/push或提交真实GPU作业；远程权重读取、CUDA训练和8rank恢复结果以实际运行报告为准。
