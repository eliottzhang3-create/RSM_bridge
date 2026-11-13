# init0904：CED8→4 串行注入、input 重注入和两轮独立 LoRA r8 音频训练

完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8`。
训练语义只更换文本初始化来源，模型、音频、数据、优化器和完整训练循环保持母本实现。
本地 Windows 维护核心代码，经 GitHub 同步；生产权重、数据、输出和 GPU 执行位于远程 Linux。

## 初始化与模型

默认文本 checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-maxlr8e-4-resume4500-20260904_154711/checkpoint-step-009244
```

直接使用本目录 `models/recursive_layer_inject.py` 的显式非 MeSH `RecursiveLlamaForCausalLM`。
严格加载完整 LM，拒绝 missing/unexpected/mismatched/error；保持原 embedding/head 绑定、
hidden576、20 个独立物理层、30 次执行及 `0..14,5..14,15..19` 轨迹和原抽层映射。
沿用已有转换后的 checkpoint，不需要重新转换。tokenizer 从同目录加载，执行母本的
`!` PAD 注册并检查词表不增长、EOS/SEP0。

首次只继承文本 LM 和 tokenizer，不读取或反序列化文本 `training_state.pt`。
音频从 step0 开始，重新建立 optimizer、scheduler、GradScaler、GradNormTracker、各 rank RNG 和数据游标。
冻结 eval 的 CED-small 从原 `models/CED_small` 资产严格加载；原 `freq_merge_126` mapper 新初始化并训练。
完整文本 LM 继续全量训练；不是冻结基础权重的 LoRA 微调。

每批一次 CED 编码，取得 final 及完整 block4/block8 的 final norm 前输出。
原 mapper 产生 E `[B,126,576]` 作为 LM 音频前缀，独立 A4/A8 MLP 保持
LN1536 → Linear1536/576 → GELU → Dropout0 → Linear576/576 → LN576。
从 E 每次完整 LM forward 只计算一次共享 `audio_linear(E)`，不 detach。

| 零基逻辑入口 | 串行残差顺序 | 中间十层 LoRA |
|---|---|---|
| 5 | 共享 input 残差 → 独立 CED8 残差 | 第一套 A/B |
| 15 | 同一 input 残差 → 独立 CED4 残差 | 第二套 A/B |

CED gate 读取 post-input hidden，残差只直接更新前126个位置；无第三次注入。
两套 CED MLP/gate 注册顺序仍为4、8，输入接口始终 `(A4,A8)`；共享 input gate 和两个 CED gate 均零初始化。

仅物理层5..14的 q/k/v/o 和 gate/up/down 七类投影新增两套独立 LoRA。
基础 W/bias 共享且全量训练，前五层和后五层不加 LoRA。rank8、alpha16、scale2、dropout0，
A 为 Kaiming uniform、B 为零；独立 CPU RNG 初始化，不扰动母本 mapper、音频分支或后续 RNG。
LoRA 作用于该轮完整序列，每次层调用和异常退出后清除轮次选择；不合并 adapter，保持无 KV，禁用 decoder gradient checkpointing。

保持 1,628,160 个新 LoRA 参数及 2,777,859 个串行音频分支参数，合计相对普通 CED 循环母本新增 4,406,019。
保留首步两轮 B 有任务梯度、A 任务梯度为零，以及第一次更新后 A/B 和打开 gate 的音频上游梯度审计。

## 原样保留的训练设置

| 项目 | 设置 |
|---|---|
| 数据 | hybrid 1,477,492 行，原所有来源、manifest、resolved_paths 和采样顺序 |
| 音频 | mono、32kHz、中心裁剪/右补零 fixed20；整段 CED 预处理，原 freq_merge_126 |
| 序列 | audio126 + SEP1 + prompt256 + answer300 = 683；prefix383 |
| 文本协议 | 原模板与 EOS 后缀、PAD `!`/17、EOS/SEP0；attention_mask=None，use_cache=False |
| Loss | answer-only；rank-local microbatch 有效 token 均值 → GA4 均值 → DDP 均值 |
| Batch | 8GPU × microbatch8 × GA4 = GBS256 |
| 长度 | 3 epochs；5771 updates/epoch；17313 total；866 warmup；每 epoch 丢116行 |
| 学习率 | 基础 LM max3e-4；mapper、音频分支及两套 LoRA max6e-4 |
| 优化 | 原 AdamW betas(.9,.999)/eps1e-8，矩阵WD.01/bias及低维WD0；LambdaLR warmup5%+cosine floor.1 |
| 梯度 | 原 GradNormTracker，initial L2=1/max=10/overdrive2.5/momentum.995 |
| 精度 | FP32，无 autocast，关闭 TF32，保留 GradScaler |
| 随机性 | seed1234、rank RNG=1234+rank、原 shuffle/sampler |
| 环境 | mellow_c8204d8，原镜像，pdgpu-5090 / 32CPU / 256G / 8GPU / 1node |

原数据和 CED 路径：

```text
manifest=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
resolved_paths=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
CED_MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small
```

共享原始音频立即开训，单 producer 默认4个 worker 后台复制原始字节。
全部复制完成后在 optimizer 边界、读取四个 microbatch 前，经全 rank all_reduce(MIN)共同切换本地。
退出先停止并 reap producer，再清理本 token 拥有的 tmpfs；每次恢复重建 staging，从 shared 开始。
登录节点沿用原 CED 资产检查和 CPU metadata 索引；计算节点运行 torchrun。
直接提交一个 formal，无新增自动 smoke/tensor 前置作业。

## 隔离与源码对照

| 项目 | 本路线 |
|---|---|
| route | `ced_small_mizar_s1_fixed20_gbs256_init0904_layer8_4_audio_reinject_loop_lora_r8_v1` |
| launcher | `run_mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_5090.sh` |
| 输出根 | `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_gbs256_5090/` |
| index cache | 上述输出根的 `index_cache/` |
| tmpfs | `/dev/shm/mizar_s1_gbs256_ced_init0904_in84_loop_lora_r8_<token>` |
| 日志 | `code/RSmol/log/mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_5090.<timestamp>.JOB.log` |
| job 前缀 | `mizar-ced-i0904-in84-lr8-g256-` |

运行源码只调整 `contract.py` 的 route/默认文本路径和 `run.sh` 的入口提示/tmpfs 名称，
launcher 只调整初始化来源和隔离命名。`train.py`、`optim.py`、`write_config.py`、
全部模型/LoRA/CED/数据/索引/staging 源码与母本逐字节一致。
另有独立的 `audit_reference.py` 校验完整源码差异白名单和完整 semantic recipe。
69份母本、共享文本模型及此前普通 init0904 路线文件的 LF SHA256 记录于 `source_guard.json`。
`parent_reference_audit.json` 是原路线历史审计，`reference_audit.json` 是本路线本地源码审计；两者均不表示真实 GPU 验证。
目录内 `.gitignore` 明确放行三份审计 JSON，普通 Git 提交可完整同步。

tmpfs 名称保留原路径守卫要求的 `mizar_s1_gbs256_` 前缀；
本地回归执行实际 shell 赋值和原 `validate_stage_root`/`cleanup` 函数，检查前缀、直属路径、软链接和 owner/token。
母本、此前普通 init0904、其他训练路线及现有评测均保持原样。

## 远程正式提交

本地代码提交并推送 GitHub 后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_5090.sh \
  --stage-workers 4
```

默认已经指定 init0904 checkpoint，自动创建独立的 `formal_<timestamp>` run root。
首次使用新的空输出目录，不传 `--resume-checkpoint`。如需显式记录本次输出位置：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_gbs256_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
bash code/RSmol/run_mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_5090.sh \
  --output-dir "$RUN_ROOT" --stage-workers 4
printf 'Training output: %s/formal\n' "$RUN_ROOT"
```

母本 `--text-model-dir`、`--ced-model-dir`、`--resolved-paths`、`--index-cache-root` 等参数保留。

## 产物与完整恢复

run root 下的主要产物：

```text
formal/runtime.json
formal/config_audit.json
formal/trainable_params.txt
formal/first_gradient_rank<N>.json
formal/opened_gate_gradient_rank<N>.json
formal/staging_report.json
formal/metrics.jsonl
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/training_report.json
```

沿用 step2886/5771/8656/11542/14428/17313 及 epoch3 最终保存。
完整 checkpoint 包括基础 LM、两套 LoRA、全部音频分支、optimizer、scheduler、GradScaler、
GradNormTracker、八 rank RNG 和训练游标。恢复不重置已训练的 LoRA 或音频模块。
只接受本路线相同源码、资产、数据、依赖和配方的完整 formal checkpoint，并沿用原 run root；
原母本、普通 init0904 或其他注入顺序不能作为本路线 full resume。

将下面的 run root 和保存点替换成实际被中断的本路线产物：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_gbs256_5090/formal_ACTUAL_TIMESTAMP"
bash code/RSmol/run_mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_5090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

## 本地验证

在仓库根目录执行：

```powershell
python -B tests/validate_mizar_s1_ced_init0904_loop_lora.py
```

报告位于 `audit_reports/mizar_s1_ced_init0904_loop_lora_validation.json`。
检查69份原文件摘要、全部复制源码和完整配方、17314个LR位置、实际 step0/完整恢复控制流、
严格加载及错误身份拒绝、非零 LoRA 两轮数值/串行 post-input gate/梯度规则、
真实并发复制和共享切换、真实 Bash 入口配合模拟 vc/torchrun、实际 tmpfs 路径守卫及 Git 审计文件可提交性。
六套测试分别在独立 Python 进程运行，避免通用模块名交叉导入；语法按远程 Python3.10 检查。

本地缺少 Torch/Transformers 时，真实 CPU 张量检查明确跳过。
已有依赖的服务器环境可手动执行合成 CPU 小模型检查，要求无 skip：

```bash
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate mellow_c8204d8
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
python -B tests/validate_mizar_s1_ced_init0904_loop_lora.py --require-tensors
```

该检查不读取生产资产、不运行 CUDA、不提交作业，也不是 formal 前置。
真实 CPU checkpoint 测试中的 collective/scaler 使用明确 fixture，不代表真实八卡恢复。
真实权重加载、八卡训练与恢复以远程实际报告为准；本次仅准备代码，不代为提交作业。
现有评测合同尚未登记本 init0904 LoRA 新身份，后续评测需单独接入。

## 本地 Git 提交

从本地仓库根目录执行，只暂存本路线文件；`RM.md` 按块选择本次记录：

```powershell
git add -- code/RSmol/run_mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_5090.sh code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8 "tests/*mizar_s1_ced_init0904_loop_lora*.py" audit_reports/mizar_s1_ced_init0904_loop_lora_validation.json
git add -p -- RM.md
git diff --cached --check
git diff --cached --stat
git commit -m "Add isolated init0904 CED8-to-4 audio training with loop LoRA r8"
git push
```
