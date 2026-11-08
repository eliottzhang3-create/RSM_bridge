# CED4/8/12：input 重注入后进行音频 cross attention

2026-11-08。完整复制隔离自原 `layer36912_crossattn` 路线，串行 input 注入参考
`layer8_4_inject_audio_reinject`。内部短目录 `in_ca4812` 用于兼容 Windows 路径长度。
本地仅维护代码，通过 GitHub 同步；权重、数据和正式 GPU 训练保留在远程 Linux。

## 架构

原非 MeSH LM 保持 20 个物理层、30 次逻辑执行，schedule 为 `0..14,5..14,15..19`。
不增加 MeSH、memory、LoRA。逻辑索引 5、15 处各执行一次 input → CA，然后进入中段循环。
第二次中段循环完成后直接进入 suffix。

冻结且始终 eval 的 CED 每音频 batch 只执行一次。只读 hooks 捕获第 4/8/12 个 block
（零基 3/7/11）的原始残差输出，位于 CED 最终 LN 之前；正常或异常退出均移除 hooks。
原最终层 LN 和 freq_merge_126 mapper 主路径保持。

每层 `[B,126,1536]` 各自经过 **affine LN1536 → Linear1536→576 → affine LN576**。
保留原 split/time/frequency 顺序；三个 bank 每完整 LM forward 各计算一次，供两处复用。
六个音频 LN 均独立，eps=1e-5、初始 weight=1/bias=0。无额外 FFN/激活/dropout。

设 `E0` 为原 mapper 音频输出，即 LM 原始 input embeddings 的前 126 个位置：

```text
# 每完整 LM forward 一次；保持 mapper 梯度
R = P_input(E0)                                      # Linear576→576，不 detach
Ai = LN_after_i(Linear_i(LN_before_i(CED_i)))          # i=4,8,12

# 两个循环入口分别重算以下内容；全部模块在两次入口共享
H = current_hidden[:, :126]
g_input = tanh(G_input(H))                            # raw pre-input hidden
Htilde = H + g_input * R
alpha = softmax(G_depth(Htilde), dim=-1)              # [B,126,3]
A_mix[:,t] = sum_i alpha[:,t,i] * Ai[:,t]              # 同时间跨深度混合
Q = LN_Q(Htilde)
C = MHA(Q, A_mix, A_mix)                              # 每个 Q 可看全部126个音频 K/V
g_CA = tanh(G_CA(Htilde))                             # raw post-input hidden
Hout = Htilde + g_CA * C
```

两个残差都只写回位置 0..125；SEP126、prompt/answer/PAD127 起保持其入口值。
原 LM causal attention 和 RoPE 保持。prompt mask 作为兼容参数仍可传入，但不参与路由。
Gate1（G_depth）和 CA gate 直接读取 Htilde，只有 CA 的 Q 使用 LN。
**无轮次偏置、无 CA 输出 Norm、无固定或可学习 residual scale。**
CA 输出幅度由其投影权重及 tanh 门控决定，不再有原 0.3 的绝对残差上限。

| 模块 | 共享、初始化 |
|---|---|
| 三个特征 Linear | 深度独立，Xavier uniform / bias0；每 forward 一次 |
| P_input | 两次入口共享，沿串行参照的 Linear 默认 weight / bias0；每 forward 一次 |
| input gate、CA gate | 各 Linear576→1 / tanh，weight/bias0 |
| depth gate | Linear576→3 / softmax，weight/bias0，初始每 token 各层 1/3 |
| Q LN | affine LN576，eps1e-5，weight1/bias0；两次入口共享 |
| MHA | width576、9 heads、head_dim64、dropout0、bias=True、batch_first=True；输出 weight 非零初始化 |

相对普通 CED 循环母本新增 **4,334,405** 参数，共 32 份参数 tensor。
严格加载和审计原文本权重后注册新模块；CPU fork_rng 保持原 LM/mapper 初值及调用方 RNG。
零门控时主路径为恒等注入。首步检查 mapper、LM、input gate、CA gate 的非零梯度，
其余新增参数允许初始 task 梯度为零，但必须已连接且有限；开门后逐一审计三个投影、
各 pre/post LN、Q LN、depth gate、MHA、input projection，最多等待 32 次更新。

## 原训练配方

| 项目 | 配置 |
|---|---|
| 学习率峰值 | 原 LM 全部参数 3e-4；mapper 和所有新增参数 6e-4 |
| AdamW | betas=(0.9,0.999)，eps1e-8，WD0.01；bias 与 ndim≤1 不衰减 |
| 调度 | 原 LambdaLR，首步 0.1×maxLR，5% rounded warmup，cosine floor0.1 |
| 精度 | FP32，无 autocast，TF32关闭，原 GradScaler / GradNormTracker |
| 数据/序列 | 原 hybrid 1,477,492 行，fixed20 在线解码；126+1+256+300=683 tokens |
| loss | answer-only；rank-local microbatch有效token均值 → GA均值 → DDP均值 |
| batch | 8 GPU × micro8 × GA4 = GBS256 |
| 时长 | 3 epochs；5771 updates/epoch，共17313；warmup866；每epoch丢弃116行 |

共享原始音频立即开始 formal，initial_steps=0；单 producer 默认4 workers 后台复制原字节。
全部完成后八 rank 在 optimizer 边界、取下个 microbatch 前以 all_reduce(MIN) 统一切本地副本。
退出先停止并 reap producer，再清理本 token 独立 tmpfs。恢复时重新建立 staging。
源数据、文本模板、随机数、sampler、保存/恢复与训练更新顺序沿母本。

默认远程资产：

```text
text=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244
ced=/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small
manifest=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
resolved=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
```

## 正式提交

将本地新增代码提交并推送 GitHub 后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4812_crossattn_audio_reinject_5090.sh --stage-workers 4
```

直接用 `vc submit` 提交一个 formal，无自动 smoke/tensor 前置作业。
沿 `pdgpu-5090 / 32CPU / 256G / 8GPU / 1node`，原容器
`docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1` 与 `mellow_c8204d8`。
CPU 索引准备/复用由 launcher 完成。训练目录、route/output/job/log/index-cache/tmpfs 及恢复身份独立。
支持 `--output-dir`、`--resume-checkpoint`、`--resolved-paths`、`--text-model-dir`、
`--ced-model-dir`、`--index-cache-root`、`--stage-workers`；fresh 输出目录须为空。

默认产物根目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer4812_crossattn_audio_reinject_5090/formal_<timestamp>/formal/
```

包含 runtime.json、config_audit.json、trainable_params.txt、first/opened_gate_gradient_rank<N>.json、
metrics.jsonl、staging_report.json、training_report.json；最终权重为
`checkpoints/model--step-17313.ckpt` 及 `checkpoints/model--epo-3.ckpt`。
中途保存2886/5771/8656/11542/14428/17313。完整保存全部模型、optimizer、scheduler、
GradScaler、GradNormTracker、八 rank RNG 与 epoch/update 游标。
metrics 记录 rank0 最后一个 microbatch 的两轮平均 depth 权重、input/CA gate 均值和绝对值、
input residual/原始 CA/CA residual/入口 audio hidden RMS，以及 CA residual token RMS 最大值。

## 完整恢复

只接受本路线、同源码/资产/数据/依赖/配方、原 run root 下的未完成 full checkpoint。
route ID 为 `ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer4812_crossattn_audio_reinject_v1`。
原36912、串行8→4及其他架构不能作为本路线 full resume。
在远程 `code/RSmol` 下替换真实 run root 后执行：

```bash
RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer4812_crossattn_audio_reinject_5090/formal_实际时间戳
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4812_crossattn_audio_reinject_5090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

## 验证范围

从仓库根目录运行 `python -B tests/validate_mizar_s1_ced_ca4812_audio_reinject.py`。
本地环境缺 torch/torchaudio/transformers/soundfile；NumPy 实际方法检查不能替代 autograd。
本次 **118项实际本地检查通过，39项真实CPU tensor检查因缺依赖跳过**（本新路线12项）。
完整结果见 `audit_reports/mizar_s1_ced_ca4812_audio_reinject_validation.json`。
验证含原文件 SHA 保护、复制源 inventory、骨干/mapper/预处理/完整更新循环 AST 对照、
实际 NumPy tokenwise mixing/前后 gate 输入/音频写回/CA 幅度/因果性、LR/模块/梯度审计拒绝、
真实 Bash 模拟 fresh/resume/失败提交、共享启动/后台复制/集体切换/清理及母本回归。
Windows 沙箱阻止临时目录后，经授权在沙箱外执行同一验证流程。

可在远程现有依赖环境手动执行真实 CPU 合成模型检查（可选，不作为正式作业前置）：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
conda activate mellow_c8204d8
python -B tests/validate_mizar_s1_ced_ca4812_audio_reinject.py --require-tensors
```

`--require-tensors` 在缺依赖、任何 skip 或失败时返回非零。真实 tensor 用例包含零门控
logits/原参数梯度等价、全部新增梯度解锁、一次投影与串行调用、音频写回和 CPU checkpoint continuation。
其中 collective/CUDA RNG 接口为 mock，不构成真实八卡恢复验收。
此次未执行 commit/push、生产权重/数据读取、GPU训练或远程作业提交。
现有 MMAU/MMAR evaluator 需要另外适配本路线新合同与 state_dict。
