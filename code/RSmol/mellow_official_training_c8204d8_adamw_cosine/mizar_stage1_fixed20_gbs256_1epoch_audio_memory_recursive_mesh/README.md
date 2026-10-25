# HTSAT stage-2 audio memory / recursive MeSH / one epoch

本目录从 `../mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh` 复制隔离。
递归模型实现位于本目录的 `recursive_model_audio_memory.py`；原训练目录和共享
`code/RSmol/recursive_model_5_10x2_5_mesh.py` 保持原样。

## 初始化与音频分支

默认加载已完成训练的 checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_recursive_mesh_5090/formal_recursive_mesh_padfix_20261025_003124507033014/formal/checkpoints/model--step-17313.ckpt
```

旧模型的所有参数和持久 buffer 必须完整匹配、逐 tensor 加载相等。只允许新增模块
缺失；不继承旧 optimizer、scheduler、GradScaler、GradNorm 或 RNG 状态。
`--resume-checkpoint` 则用于恢复本新路由的完整正式训练状态。

每条样本沿用原 fixed20 音频处理，两个 10 秒半段都送入 HTSAT。捕获每段 stage 2
最后一个 block、PatchMerging 之前的 `[B,1024,192]`；逆 HTSAT 的频率打包，得到真实
`[B,8,128,192]`。两个半段先沿时间拼成 `[B,8,256,192]`，再做：

```text
LayerNorm(192) -> Linear(192,576)
  + fixed temporal encoding [1,1,256,576]
  + learnable frequency embedding [1,8,1,576]
  -> flatten -> A [B,2048,576]
```

时间间隔为 `20/256 = 0.078125` 秒，位置中心为 `(i+0.5)*20/256` 秒。
沿用原 sinusoidal 规则，以 `time_seconds / bin_seconds` 编码，即中心 ID
`0.5..255.5`。8 个频率组共享这套时间编码；每组的 `[576]` 可学习频率向量在两个
半段的全部时间位置复用，零初始化。没有额外的 attention-logit 频率 bias。

HTSAT 继续完整执行 stage 3/4，最终 embedding 仍通过冻结的 c2l/projection 生成
原音频前缀。整个旧音频分支固定为 eval，并在 no_grad 下运行。

## 注入位置与 router 语义

每次 cross-attention 为单层、9 头、每头 64 维，无额外 FFN。两次使用独立的 Q
LayerNorm/Q projection 和独立的输出投影 `W_O`，共享一套 memory LayerNorm、K
projection 和 V projection；K/V 每次 forward 只计算一次。两个输出投影的 weight
及 bias 均零初始化，保证初始新增写入恰为零。两次均可读取全部 2048 个音频 token。

两个 perceive router 与原 router 同构，均为 `Linear(576,5)+softmax`，沿用原 router
的随机权重初始化与零 bias。零注入由 `W_O=0` 实现，不对 softmax 权重做零初始化。
每个 slot 仍保留文本位置维度 `[B,5,L,576]`，写入是逐位置加法。

```text
M  = InitSlots(X)                         # slot 0 = X，其他 slot = 0
Z0 = Prelude(X)
C0 = CrossAttn_0(Z0, A)
M  = Write_pre(M, value=Z0, query=Z0)
M  = Perceive_0(M, value=C0, query=Z0)
H0 = Read_pre(M, query=Z0)

Z1 = Core(H0)
C1 = CrossAttn_1(Z1, A)
M  = Write_loop0(M, value=Z1, query=H0)
M  = Perceive_1(M, value=C1, query=H0)
H1 = Read_loop0(M, query=H0)

Z2 = Core(H1)
M  = Write_loop1(M, value=Z2, query=H1)
H2 = Read_loop1(M, query=H1)
output = LMHead(FinalNorm(Coda(H2)))
```

仅训练新增音频模块、两个 perceive router、原六个 read/write router，以及共享
Core 的物理第 5..14 层（从 0 开始计数）。冻结 HTSAT、c2l、projection、embed_tokens、
Prelude 第 0..4 层、Coda 第 15..19 层、final norm 和 lm_head。Coda/final norm/lm_head
保留对输入的反向传播，以训练前面的 Core 和 routers。

## 数据与优化

沿用原 manifest 与 resolved-path TSV，共 1,477,492 行。8 GPUs × 每卡 microbatch 8
× 梯度累积 4，global batch 256。训练 1 epoch，共 5,771 次 optimizer 更新；warmup
289 次，drop_last 丢弃 116 行。前缀仍为 `audio[129+129]+SEP[1]+prompt[256]`，答案
300 token，总长 815；外部答案 CE、PAD/EOS 处理及数据顺序沿用原版本。

所有训练参数的最大学习率统一为 `5e-5`，cosine 最低为 `5e-6`。保留原 5% warmup、
AdamW `(0.9,0.999), eps=1e-8`、weight decay 0.01（bias/ndim<=1 除外）和 GradNorm。
使用 FP32 forward、关闭 autocast/TF32，沿用 GradScaler 与更新顺序。

## 提交与审计

同步仓库到集群后，在仓库根目录执行。默认初始化路径已经内置：

```bash
# 5 次真实 optimizer 更新；直接读取共享原始音频，不启动音频 staging。
bash code/RSmol/run_mizar_s1_audio_memory_gbs256_5090.sh --mode smoke

# 正式训练：沿用共享音频启动、后台 staging、全部完成后各 rank 一起切换的流程。
bash code/RSmol/run_mizar_s1_audio_memory_gbs256_5090.sh --mode formal
```

两者共用 `train.py`、优化器及完整 5,771 步 LR horizon，smoke 只限制真实更新数为 5。
提交资源为原 `pdgpu-5090` 容器、8 GPU、32 CPU、256G 内存；`--output-dir` 可指定新
输出目录，默认在 `mizar_s1_htsat_audio_memory_gbs256_5090/<mode>_<timestamp>` 下隔离。
CPU index 只处理 metadata/path；smoke 不复制音频、不生成音频/encoder cache。

smoke 会先运行 `runtime_checks.py`，再在真实模型/数据上检查：

- 严格旧参数加载、新参数白名单、冻结范围、完整冻结参数及 buffer 前后 SHA256。
- 两个 `W_O` 和 frequency embedding 的零初始化；开启/关闭注入的 logits 精确一致。
- 两个 HTSAT 半段、真实时频重排、时间/频率编码，stage 3/4 继续执行。
- 实际 `Z0/Z1` cross-attention query、`Z0/H0` perceive query、原 `Z0/H0/H1` read/write
  query；独立回放三次 memory 写入及读取，检查路由 softmax 和共享 Core 调度。
- 每次更新所有训练参数梯度存在且有限，冻结参数无梯度；两次 Core 的四个边界梯度。
  首步允许零初始化上游梯度为零，5 步内每个训练模块必须出现非零梯度并更新。
- 数据采样顺序、答案 loss、累积/调度顺序、五次真实更新、八个 rank 参数精确同步。
- 完整 checkpoint 保存和重读，模型 tensor、optimizer step、scheduler 和 RNG 状态。

输出包括 `<output-dir>/<mode>/runtime.json`、`config_audit.json`、
`trainable_params.txt`、`metrics.jsonl`、`training_report.json`。
smoke 成功时 report 的 `status=PASS`、`optimizer_steps=5`、`staging_used=false`。
checkpoint 为 `smoke/checkpoints/model--smoke-5.ckpt`；正式最终为
`formal/checkpoints/model--epo-1.ckpt`，并保留中间 step checkpoint。

正式中途恢复时使用原输出目录及 `--resume-checkpoint <new-route-formal-step.ckpt>`；
必须保持同一新路由代码、初始化资产和训练契约。旧训练或 smoke checkpoint 不能作为
正式完整 resume。新增音频模块随 checkpoint 保存，后续评测加载需要相应的新模型和
两份 stage 2 conditioning。
