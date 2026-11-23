# 三循环 7–14×3–7 MeSH：CED4/8 write-value fusion + 三轮独立 LoRA r8

2026-11-24，按确认计划完整复制隔离自 mizar_s1_gbs128_text71437_mesh_ced_small，
迁入 mizar_s1_gbs128_text71427_mesh_ced48_wfuse_lora_r8 的两处融合与完整恢复逻辑。
本地 Windows 维护代码，经 GitHub 同步；生产模型、音频、数据和 GPU 执行均在远程 Linux。
既有训练路线、共享文本模型和评测代码保持；并行 audio8/read3 路线的修改不属于本次提交。

## 正式提交

本地提交并推送本路线代码后，在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced48_wfuse_lora_r8_3090.sh
```

默认从以下已训练三循环文本 checkpoint 继承完整模型、已训练 routers 和 tokenizer，无需重新转换：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x3_7_mesh/formal_reinit1200_seed1_20261119_084545_3432609/checkpoint-009244
```

CED 仍为 /hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small。
登录 CPU 检查文本完成标记/config、CED 资产及准备/复用 index；成功后通过 vc submit 申请
pdgpu-3090 / 8GPU / 64CPU / 256G / 1node，原镜像、mellow_c8204d8 环境。
直接 formal，无新增 GPU smoke 或人工放行前置。job 含完整后缀少于60个字符。

## 精确前向与参数

28 个独立物理层，56 次执行：0..6、(7..20)×3、21..27。继承全部四组 write/read：
八个独立 Linear(576,6)，16 个 router 张量、27,696 参数，不重初始化。
每完整 forward 在 prefix 前重建六槽 [embedding,0,0,0,0,0]，覆盖全部683个位置。

| 边界（已执行层数） | write/read query | write value | 后续 |
|---|---|---|---|
| 7，前七层后 | prefix output | 前126个音频位置与 A4 融合；其余原样 | 第一轮 / LoRA 0 |
| 21，第一轮后 | 第一轮 input | 第一轮 output 的前126位置与 A8 融合；其余原样 | 第二轮 / LoRA 1 |
| 35，第二轮后 | 第二轮 input | 第二轮 output 原样 | 第三轮 / LoRA 2 |
| 49，第三轮后 | 第三轮 input | 第三轮 output 原样 | 后七层，从逻辑49开始 |

第三轮前不增加音频融合；第三轮仍能通过既有 memory 利用前两次引入的信息。
不增加 seed 槽、listen router、音频 gate、额外 memory write 或 hidden 残差注入。

CED 冻结且所有子模块保持 eval，每批一次编码取得 final 和完整 block4/8 norm前输出。
保持整段 fixed20 的 mel/BN 后切分、逐样本 dB，以及 [split,frequency,time] 到
[126,4,384] 的正确排列。final 仍走官方 freq_merge_126 主 mapper；
两个独立 LN1536→Linear1536/576→GELU→Dropout0→Linear576/576→LN576 产生 A4/A8，
主 mapper/MLP 均在 CED no_grad 外训练。

两套独立无 bias Linear(1152,576) 只替换上述两次 write value：

```text
V_audio = W_f @ concat(H_audio, A_CED)
W_f 初始 = [I576, 0]
```

hidden 在前、CED MLP 输出在后；整个矩阵训练，非原地构造 V。
SEP、prompt、answer/PAD 的 value 和 router query 来源保持。
[I,0] 与零 LoRA B 时，同文本权重/主 mapper/输入下数学上等价普通三循环六槽母本；
CPU 小模型已核对 logits/继承参数梯度容差内一致，不宣称生产 CUDA 逐位一致。

LoRA 仅安装在物理7..20的 q/k/v/o、gate/up/down 七种投影。
每层/投影/轮次三套独立 A/B，r8/alpha16/scale2/dropout0，A Kaiming/B零；
基础 W/bias 共享且全量训练，前后七层无 LoRA。共294对/588矩阵，3,419,136参数。
每轮98次投影调用，共294次；中段层轨迹42次、全模型56次。
逐调用 finally 复位轮次；拒绝 KV cache 和 decoder gradient checkpointing。
每个 forward 显式接收本批次 (A4,A8)，不持久缓存跨样本状态。

| 相对普通三循环 CED-small 新增 | 参数 |
|---|---:|
| 两套中间层 MLP | 2,443,776 |
| 两套融合矩阵 | 1,327,104 |
| 三轮 LoRA | 3,419,136 |
| 合计 | 7,190,016 |

先严格加载并逐张量核对文本 FP32 权重、全部16个 router 张量和 tied embedding/head，
再注册新融合与 LoRA；复核原值、原 Parameter 对象与绑定不变。新模块初始化隔离 CPU RNG。
只继承文本权重/config/tokenizer，不读取 training_state.pt；音频 optimizer/scheduler/
GradScaler/GradNorm/RNG/数据游标均从本轮 step0 新建。文本来源 seed1 不改变音频 seed1234。

## 训练合同

| 项目 | 配置 |
|---|---|
| 数据 | 原 hybrid 1,477,492 行，原 mapping/顺序，含 MusicCaps/全部 SpeechCraft splits |
| 音频/文本 | mono32k、中心裁20秒/右补零；原 S1 EOS/PAD/截断 |
| 序列 | audio126 + SEP1 + prompt256 + answer300 = 683，prefix383 |
| 损失 | 外部 shifted answer-only CE，首答案logit382；rank-local有效token mean→GA4均值→DDP均值 |
| Batch | 8GPU × micro4 × GA4 = GBS128 |
| 时长 | 3epochs，11542更新/epoch，总34626，warmup1731，每epoch丢弃116行 |
| 最大LR | 基础LM/原routers 3e-4；主mapper/MLP/fusion/三轮LoRA 6e-4 |
| 优化 | 原AdamW(.9,.999)/eps1e-8，矩阵WD.01、bias/低维WD0 |
| 数值/调度 | FP32，无autocast/TF32，保留GradScaler、LambdaLR floor.1、原GradNormTracker |
| RNG | seed1234、rank1234+rank；原数据顺序 |

训练立即读取共享原始音频，默认4worker后台复制 raw bytes；全量文件/字节通过后，
八rank在 optimizer 边界、取microbatches前 all_reduce(MIN) 共同切换本地副本。
tmpfs 为 /dev/shm/mizar_s1_gbs256_text71437_ced48_wfuse_lora_r8_<token>：
历史 gbs256 前缀满足原安全守卫，实际训练GBS为128。
退出先停止/reap producer，再核对owner/token清理本任务目录。

保留首次三轮输入/输出反传和所有16个router张量的真实梯度审计，随后立即释放保留边界。
首步融合Encoder半矩阵可有任务梯度；其初值为零使MLP首步任务梯度为零，
零LoRA B使A首步任务梯度为零，均为预期。新增10个分支角色需在32次真实更新内全rank累计解锁；
分别记录融合矩阵两半，不把weight decay变化当任务梯度。metrics保留两次fusion统计和三轮LoRA轨迹。

## 输出与完整恢复

默认外层 run root：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced48_wfuse_lora_r8_gbs128_text71437_mesh_3090/formal_<timestamp>
```

其下 formal/ 保存 runtime.json、config_audit.json、metrics.jsonl、staging_report.json、
first_recursive_backward_rank<N>.json、first_gradient_rank<N>.json、fusion_gradient_open_rank<N>.json、
training_report.json、checkpoints/。保存5771/11542/17313/23084/28855/34626步与model--epo-3.ckpt。
最终文件为 <run-root>/formal/checkpoints/model--step-34626.ckpt。

保存完整LM/routers/mapper/CED/MLP/fusion/三轮LoRA、AdamW及具名参数库存、scheduler、
GradScaler、GradNorm、八rank Python/NumPy/CPU/CUDA RNG、下一数据游标和累计解锁状态。
恢复要求本新路线相同源码/资产/文本来源/数据/依赖/配方，校验全部tensor库存/shape/dtype/finite、
optimizer参数顺序/moments/LR和游标。恢复后不重置已训练fusion/LoRA或解锁状态。
已有二循环wfuse、普通三循环、audio8/read3等checkpoint均不可作为本路线full resume。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN_ROOT=/absolute/original/formal_timestamp
bash code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced48_wfuse_lora_r8_3090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

传原外层run root，不能再追加formal/。每次恢复重建staging，从shared开始。
已完成训练不可作为resume输入。现有评测器未登记本新合同，后续评测须另行适配。

## 本地验证和同步

```powershell
python -B tests/validate_mizar_s1_text71437_ced48_wfuse_lora_r8.py
# 可选复用已有本地依赖，不下载或修改：
python -B tests/validate_mizar_s1_text71437_ced48_wfuse_lora_r8.py --dependency-dir .codex_audio_cutoff_deps
```

完整依赖环境可手动加 --require-tensors 禁止跳过；不作为formal前置。
报告 audit_reports/mizar_s1_text71437_ced48_wfuse_lora_r8_validation.json；
最终联合139项中111项执行通过、28项因缺少音频依赖明确跳过；其中新路线57项执行通过、11项跳过。
真实CPU使用Python3.13.5/Torch2.9.0+cpu/Transformers4.54.1，与远程Torch2.11/CUDA环境不同。
初次历史守卫失败独立保留于 *_initial_validation.json，后续失败/修复记录在previous_runs。
reference_audit.json仅表示本地源码对照；parent_reference_audit.json仅为母本历史。
数值helper、四次memory交互除两次value替换、完整epoch更新主体与原实现做AST对照，
15份数据/staging文件与三循环母本逐字节一致，CED/mapper/音频接口与wfuse母本逐字节一致。

真实CPU六项核心检查覆盖严格加载/初始化RNG、初始logits和梯度对齐、56调用/683位置/
三轮梯度/16router、588矩阵独立及梯度解锁、非零融合/第三轮无新注入、完整模型/AdamW/
RNG保存恢复后下一步逐值一致。collective和CUDA RNG在CPU测试中显式使用fixture。
缺少torchaudio/soundfile的音频集成用例明确skip；不等于生产权重、CUDA、3090显存或八卡训练PASS。
保护库存以本任务开始时3654份tracked Python/Shell/JSON为准；独立audio8/read3两份并行修改
记录于concurrent_source_changes.json，未由本任务改写。两个已过时母本历史库存断言由本轮库存替代，
旧测试和快照保持，其他母本回归仍执行。

在本地仓库根仅暂存本次文件，RM.md按块选择本条，不混入并行路线：

```powershell
$routePath = 'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71437_mesh_ced48_wfuse_lora_r8'
git add -- $routePath code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced48_wfuse_lora_r8_3090.sh tests/test_mizar_s1_text71437_ced48_wfuse_lora_r8_route.py tests/test_mizar_s1_text71437_ced48_wfuse_lora_r8_behavior.py tests/test_mizar_s1_text71437_ced48_wfuse_lora_r8_shared_start.py tests/test_mizar_s1_text71437_ced48_wfuse_lora_r8_staging_roots.py tests/test_mizar_s1_text71437_ced48_wfuse_lora_r8_tensors.py tests/test_mizar_s1_text71437_ced48_wfuse_lora_r8_core_tensors.py tests/validate_mizar_s1_text71437_ced48_wfuse_lora_r8.py
git add -f -- audit_reports/mizar_s1_text71437_ced48_wfuse_lora_r8_validation.json audit_reports/mizar_s1_text71437_ced48_wfuse_lora_r8_initial_validation.json
git add -p RM.md
git diff --cached --stat
git commit -m "Add isolated three-loop CED write-value fusion training"
git push
```

本任务未代为commit/push、读取远程生产资产或提交真实GPU作业。
