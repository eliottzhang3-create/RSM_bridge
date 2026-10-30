# CED 四层特征 → 两次 MeSH audio listen，固定 γ=0.3

完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small`，母本入口为
`code/RSmol/run_mizar_s1_gbs256_recursive_mesh_ced_small_5090.sh`。
保留原 CED 最终层、官方 mapper、MeSH 主路径及训练配方，增加四个独立音频投影、
两个 prompt gate、两个 softmax listen router 和两个零初值幅度标量。
本地维护代码，权重、数据和正式环境位于远程 Linux，经 GitHub 同步。

## 架构

冻结 CED-Small 且始终 eval，同一次 `forward_features` 用临时 hooks 提取第3、6、9、12个
block 的输出，位于 encoder 最终 LN 之前。正常或异常退出都会移除 hooks。
原 `forward_features` 返回的最终特征继续进入原 mapper，hook 不替换输出。

每层按正确的时间顺序整理：

```text
[2B,252,384]
→ [B,2 segments,4 frequency,63 time,384]
→ [B,2 segments,63 time,4 frequency,384]
→ [B,126,1536]
→ LN1536 → Linear(1536,576) → LN576
→ a_i [B,126,576]
```

四套投影和输入／输出 LN 独立。全部新增 LN 均为 `eps=1e-5, elementwise_affine=False`。
只复用 mapper 的时间排列方式，不复制其位置编码、global projection 或 refine MLP。
四路投影在冻结编码器的 no_grad 区域之外，每次 LM forward 只计算一次，供两轮复用。

20个物理层仍执行 `0..14,5..14,15..19`，五槽瞬时记忆形状 `[B,5,L,576]`。
原初始化和三组 write/read routers 保留，不增加记忆槽。

| 边界 | 操作顺序 | 新 gate 输入 |
|---|---|---|
| Prelude 5层后，逻辑索引5 | 原 write₀ → listen₀ → 原 read₀ → 循环1 | Prelude 输出的有效 prompt |
| 循环1后，逻辑索引15 | 原 write₁ → listen₁ → 原 read₁ → 循环2 | 循环1输出的有效 prompt |
| 循环2后 | 原 write₂ → 原 read₂ → suffix 5层 | 无新增操作 |

母本在循环结束后仍用该次循环的**输入**计算原 write/read 权重，用循环**输出**作为原写入值。
新 listen 使用当轮边界输出，不改变原 router query。

```text
p_r = masked_mean(LN_text(H_r[:,127:383]), prompt_attention_mask)
g_r = sigmoid(Linear_r(p_r))                         # [B,4]
A_r = sum_i g_r[i] * a_i                            # [B,126,576]
P_r = softmax(ListenLinear_r(LN_query(H_r[:,:126]))) # [B,126,5]
delta_M[:,s,t] = (0.3/4) * tanh(beta_r) * P_r[:,t,s] * A_r[:,t]
```

两个 gate 为独立 `Linear(576,4)`，weight/bias全零，初始四路sigmoid均为0.5；
不使用深度softmax，不除以权重和。池化仅包含tokenizer标记的有效prompt，不含audio、SEP、
prompt PAD或answer；空／非二值mask会被拒绝。mask只用于池化，LM仍使用 `attention_mask=None`。

两个 listen router 为独立 `Linear(576,5)→softmax`，使用原router初始化规则：
std=`sqrt(2/(5*576))` 的正态权重截到±3 std，bias=0。
**β₀和β₁分别初始化为零**，由tanh幅度保证实际初始写入为零，softmax概率本身不为零。
只更新前126个音频位置的五槽记忆，SEP及文本位置不直接写入。

非affine输出LN使每个a_i的token RMS≤1；四路sigmoid和≤4，|tanh β|≤1，P≤1，
因此**单次新增写入的每槽/token RMS≤约0.3**。固定该次原read权重时，新增写入直接带入read的
分量也满足此界限。这不是主路径的30%，不限制每个维度到±0.3，也不保证两轮累计、后续网络或梯度有界。

新增 **3,551,636** 个参数、18个可训练tensors，LN无参数。先严格加载全部原文本权重，
再在 `torch.random.fork_rng(devices=[])` 中注册新模块，保护母本CPU RNG。
原共享MeSH文件、母本与其他路线均不修改。新增分支要求完整prefix重算、无KV cache。

## 保持母本训练配方

| 项目 | 设置 |
|---|---|
| 数据 | hybrid 1,477,492行，包含MusicCaps和全部SpeechCraft splits |
| 文本初始化 | `stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244` |
| 音频 | 全通道平均、32kHz、中心裁剪／右补零20秒；整段重采样16kHz、逐样本dB、冻结BN、1012帧切分 |
| 原mapper | 官方freq_merge_126，全量可训练，dropout0.05 |
| 序列 | audio126 + SEP1 + prompt256 + answer300 = 683；prefix383 |
| 文本／loss | attention_mask=None、labels=None、use_cache=False；原answer mask与非PAD选取目标，包含有效EOS |
| Batch | 8 GPU × microbatch8 × accumulation4 = GBS256 |
| 步数 | 3 epochs；5771 updates/epoch；17313 total；866 warmup；每epoch丢弃116行 |
| Loss reduction | rank-local有效answer token mean，四个microbatch均值，再DDP mean |
| AdamW | LM max LR3e-4；原mapper及所有新增参数max LR6e-4；betas=(0.9,0.999)，eps1e-8，decay0.01 |
| No decay | bias和ndim≤1，包括两个β |
| Scheduler | 原LambdaLR，5% warmup+cosine，floor0.1，optimizer后step；初始LR为0.1倍max LR |
| GradNorm | 原Tracker：initial L2=1、max=10、overdrive2.5、momentum0.995 |
| 精度 | FP32 forward，无autocast，GradScaler启用，TF32关闭 |
| 顺序 | seed1234，rank RNG=1234+rank；原shuffle、连续rank分片及恢复跳行 |
| Staging | 登录节点CPU index；共享音频立即训练、默认4线程后台复制；update边界8卡MIN确认后切换 |
| 保存 | 2886、5771、8656、11542、14428、17313步，以及epoch3最终checkpoint |
| 环境／资源 | mellow_c8204d8；pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node |

新增参数参与原GradNormTracker；它们可能改变共同缩放，这是新架构的结果，不改变裁剪算法。
β=0时投影、gate、listen Linear首步零梯度是预期；β更新后应解锁。
每rank保留首次梯度审计，要求全部10个新增模块在前32次真实更新内获得有限非零梯度。
失败即报错，不跳过optimizer更新，不重置LR或放宽母本稳定性检查。

## 正式提交

经GitHub同步后，在远程登录节点执行；仅formal，没有自动smoke或tensor验证前置任务：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer36912_listen_5090.sh --stage-workers 4
```

默认CED资产为 `/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small`，沿用离线标签配置修复及strict
safetensors加载。完整数据、resolved_paths及文本checkpoint路径继承母本。
默认run根目录为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer36912_listen_5090/formal_<timestamp>/
```

新训练要求空输出目录。支持 `--output-dir`、`--text-model-dir`、`--ced-model-dir`、
`--resolved-paths`、`--index-cache-root`、`--stage-workers`。
入口、日志、index cache、tmpfs、run输出及resume身份独立。

最终产物位于run根目录下：

```text
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/runtime.json
formal/config_audit.json
formal/first_gradient_rank<N>.json
formal/listen_gradient_open_rank<N>.json
formal/metrics.jsonl
formal/training_report.json
```

metrics的 `audio_listen` 记录两轮深度sigmoid均值及饱和比例、槽概率及熵、β/tanh β、
聚合音频RMS、最大写入RMS、直接read增量RMS及其与原read的比值。
第1、2及每25次更新记录rank0累积、unscale后、GradNorm前的各新增模块梯度L2。
梯度快照带自身update编号；其余前向指标为最后一个microbatch。

## 恢复

使用本路线中断checkpoint及**原run根目录**：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
run_root=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer36912_listen_5090/formal_<timestamp>
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer36912_listen_5090.sh \
  --output-dir "$run_root" \
  --resume-checkpoint "$run_root/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

替换为实际run时间戳及已保存checkpoint。恢复完整模型、optimizer、scheduler、GradScaler、GradNorm、
8 rank RNG、数据游标及梯度解锁状态。拒绝母本／其他架构／旧γ checkpoint，以及代码、资产、数据、
超参或optimizer/scheduler/数据游标不一致。完成的17313步checkpoint可做最终重载审计，不能重新resume。
新作业重新staging；旧节点和tmpfs路径不参与身份比较。

## 验证与证据边界

仓库根目录运行可选汇总：

```bash
python -B tests/validate_mizar_s1_mesh_ced_listen.py
```

结果写入 `audit_reports/mizar_s1_mesh_ced_listen_validation.json`。
母本源码基线为 `audit_reports/mizar_s1_mesh_ced_listen_parent_baseline.json`。
`reference_audit.json` 仅证明源码对齐，不是权重加载、训练或GPU恢复PASS。

远程 `mellow_c8204d8` 中可手动要求真实CPU tensor检查全部执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
python -B tests/validate_mizar_s1_mesh_ced_listen.py --require-tensors
```

不下载资产、不运行CUDA训练、不提交作业，也不会被formal入口自动调用。
覆盖单次CED提取／异常清理、时间顺序、冻结／梯度、零β输出/loss/原参数梯度等价、prompt与答案因果性、
两次写入、梯度解锁、幅度界限、优化器分组及CPU checkpoint续训。
CPU续训测试明确mock八卡collectives/CUDA RNG，并关闭CPU GradScaler，不代表真实8 GPU恢复通过。
本地实际执行数与依赖跳过数见汇总报告。
真实CED/文本权重、8 GPU稳定性、CUDA GradScaler/RNG恢复和17313步完成须以远程日志／产物确认。
现有普通CED／MeSH evaluator尚未适配此新增架构。
