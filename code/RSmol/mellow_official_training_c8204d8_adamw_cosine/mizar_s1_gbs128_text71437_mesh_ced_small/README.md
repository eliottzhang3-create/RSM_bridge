# CED-small → 已训练 7–14×3–7 MeSH，GBS128 / 3090

本路线完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small`。
音频、mapper、数据、loss、优化器和真实更新循环保留原实现；文本端改为已有三循环模型，
按用户确认使用 GBS128、pdgpu-3090、64 CPU/256G。原训练路线及共享文本模型不修改。
本地 Windows 维护代码，GitHub 中转；生产权重、数据和真实训练均在远程 Linux。

## 初始化与模型

默认文本模型/config/tokenizer 来源固定为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x3_7_mesh/formal_reinit1200_seed1_20261119_084545_3432609/checkpoint-009244
```

显式使用 `recursive_model_mizar_text_7_14x3_7_mesh.MizarText7_14x3_7MeshForCausalLM`。
检查本路线架构、完成标记、9244 步及四种 schedule 别名、全部来源层映射。
严格加载完整 FP32 safetensors，逐张量核对形状/值/finite 和全部 16 个已训练 router 张量；
只允许省略一个 tied embedding/head 别名，实际 Parameter 绑定也必须保留。
不重新转换、抽层或初始化 routers，不读取文本 `training_state.pt`。
文本来源名字中的 reinit1200/seed1 只记录其历史；音频从 step0 新建 optimizer、scheduler、
GradScaler、GradNorm、RNG 和数据游标。音频 seed1234，rank 为 1234+rank。

- 28 个独立物理层，前7、中14共享执行三次、后7，共56次。
- 零基轨迹 `0..6, 7..20, 7..20, 7..20, 21..27`；后段从逻辑49开始。
- 六槽瞬时 memory `[B,6,683,576]`；slot0为完整多模态 embedding，其余五槽零。
- 四组 write/read，八个独立 Linear(576,6)，16张量/27,696参数；全LM和routers训练。
- 前段 query/value 取前段输出；循环 query 取该轮输入、write value 取该轮输出。
- 每完整forward重建memory，无KV cache。仅首个microbatch保留三轮边界供梯度审计，随后立即释放。

CED-small 默认 `/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small`，离线 strict 加载，
冻结且每个子模块保持eval。整段fixed20重采样/计算mel和冻结BN，再按1012帧分段，
逐样本dB；保持正确的segment/time/frequency排列。在线FP32编码，不使用离线hidden cache。
原 `freq_merge_126` mapper随机初始化并训练，包括频率合并MLP、global projection、
position/audio-type参数和残差refinement；dropout0.05、resampleFalse、外置projection为Identity。

## 原训练语义与本次资源

| 项目 | 配置 |
|---|---|
| 数据 | 原hybrid 1,477,492行；含MusicCaps及全部SpeechCraft splits |
| 音频 | mono/32k，单录音中心裁20秒或右补零；真实双录音保持原独立处理规则 |
| 布局 | audio126 + SEP1 + prompt256 + answer300 = 683；prefix383 |
| 文本 | 原S1模板；prompt/answer分词前追加EOS，截断后不强行补回；PAD已有 `!`，EOS/SEP0 |
| 注意力/损失 | LM attention_mask=None、labels=None、use_cache=False；外部shifted answer-only CE |
| 归约 | rank-local有效answer-token mean → GA4均值 → DDP均值 |
| 资源 | pdgpu-3090，单节点8GPU/64CPU/256G，原镜像和mellow_c8204d8环境 |
| Batch | 8 × micro4 × GA4 = GBS128 |
| 时长 | 3epochs；11542更新/epoch；34626总更新；每epoch丢弃116行 |
| LR | LM/routers最大3e-4、mapper最大6e-4；原5% warmup/cosine、floor0.1 |
| warmup | 原round公式，1731次更新；首步LR3e-5/6e-5，终点同此下限 |
| 优化 | 原AdamW(.9,.999)/eps1e-8；矩阵WD.01、bias和低维WD0；原GradNormTracker |
| 数值 | FP32前向，无autocast，保留GradScaler，关闭TF32 |
| 数据顺序 | 原seed1234 shuffle与rank连续切分，DataLoader workers0 |

三循环文本训练的BF16、LR和seed不作为音频训练配方。另一条二循环音频重新训练的
统一LR2e-4配方不适用于本路线。

## 提交正式训练

代码同步到GitHub并在远程拉取后，从仓库根运行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced_small_3090.sh
```

默认文本来源、CED路径、数据/映射、资源和3epochs均已写入入口。无需重新转换文本。
入口先在登录CPU检查文本JSON/完成标记和CED资产，准备/复用CPU index，再通过vc submit
申请GPU。计算节点严格加载真实权重。直接formal，不增加smoke或人工放行前置任务。

训练立即读取共享原始音频，默认4worker后台复制raw bytes；全量文件/字节通过后，
optimizer边界、读取microbatches之前八rank all_reduce(MIN)统一切换节点本地。
退出先stop/reap producer，再按owner/token清理本任务tmpfs。
tmpfs使用 `/dev/shm/mizar_s1_gbs256_text71437_mesh_ced_small_<token>`，其中历史gbs256前缀
为原安全守卫要求，实际训练GBS始终128。

默认输出root：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_text71437_mesh_3090/formal_<timestamp>/
```

该root下再生成 `formal/`：runtime.json、config_audit.json、metrics.jsonl、staging_report.json、
first_recursive_backward_rank<N>.json、first_gradient_rank<N>.json、training_report.json及checkpoints/。
保存更新5771/11542/17313/23084/28855/34626与最终 `model--epo-3.ckpt`，全部为完整音频状态。
最终产物为 `<run-root>/formal/checkpoints/model--step-34626.ckpt`。
日志在 `code/RSmol/log/mizar_s1_gbs128_text71437_mesh_ced_small.<timestamp>.JOB.log`。
job完整名含时间后缀少于60个字符。

可选显式保存输出root便于后续恢复：

```bash
RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_text71437_mesh_3090/formal_$(date +%Y%m%d_%H%M%S%N)"
bash code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced_small_3090.sh \
  --output-dir "$RUN_ROOT" --stage-workers 4
printf 'run-root=%s\n' "$RUN_ROOT"
```

## 完整恢复

只使用本新路线同源码/资产/文本来源/数据/依赖/训练配方的未完成音频checkpoint。
传原外层run-root，避免再多追加一层formal；将下方占位路径替换为实际路径：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN_ROOT=/absolute/original/formal_timestamp
bash code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced_small_3090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

恢复完整LM/routers/mapper/CED、AdamW、scheduler、Scaler、GradNorm、八rank RNG及下一数据游标，
累计训练至34626步。恢复前校验完整字段、route/合同SHA、batch/epochs/loss、step/scheduler/cursor及八rank RNG。
每次恢复在新节点重建staging并从shared开始；旧staging路径不进入恢复身份。
文本checkpoint仅用于初始化，其他音频路线不可full resume；已完成训练也不再作为resume输入。
当前旧评测器未登记这条新路线，后续评测需要按本路线合同单独适配。

## 本地验证与Git同步

```bash
python -B tests/validate_mizar_s1_text71437_ced_small.py
# 有真实ML依赖时可选：不下载、不跑CUDA、不提交训练；不作为formal前置。
python -B tests/validate_mizar_s1_text71437_ced_small.py --require-tensors
```

验证覆盖原源码SHA、原完整epoch更新循环/优化器/数据AST、训练来源和错架构拒绝、
全部16router库存/FP32/finite/tied校验、实际56次forward控制流的独立NumPy递推、
三轮梯度审计、全状态恢复控制流、真实Bash参数链和模拟vc、CPU index/后台复制/失败清理及实际tmpfs守卫。
NumPy模拟层、合成文件、mock collective均显式标记；真实小模型torch/autograd测试独立列出。
缺依赖测试必须明确skip，不将其或模拟提交记作生产训练PASS。
验收报告为 `audit_reports/mizar_s1_text71437_ced_small_validation.json`。
2026-11-21本地验收75项执行通过，32项真实ML依赖测试明确跳过；3166份原tracked源码LF SHA保持。
Python3.10、Bash及本文Bash命令语法检查通过。Windows沙箱拒绝临时目录/Bash管道后，
经批准在沙箱外完成离线回归；未读取远程生产资产或提交真实训练。
`reference_audit.json`保存本次官方源码对照；两个snapshot JSON均是只读测试数据，不参与正式训练门槛。

在本地Windows仓库根提交此次改动，可使用精确文件清单：

```powershell
$audioRoute = 'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71437_mesh_ced_small'
git add -- $audioRoute code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced_small_3090.sh tests/test_mizar_s1_text71437_ced_small_route.py tests/test_mizar_s1_text71437_ced_small_behavior.py tests/test_mizar_s1_text71437_ced_small_shared_start.py tests/test_mizar_s1_text71437_ced_small_staging_roots.py tests/test_mizar_s1_text71437_ced_small_tensors.py tests/validate_mizar_s1_text71437_ced_small.py audit_reports/mizar_s1_text71437_ced_small_validation.json RM.md
git diff --cached --stat
git commit -m "Add isolated CED audio training from 7-14x3-7 text checkpoint"
git push
```

本任务不代为commit/push或远程提交。生产权重完整性、3090显存、八卡训练与正式结果以服务器实际报告为准。
