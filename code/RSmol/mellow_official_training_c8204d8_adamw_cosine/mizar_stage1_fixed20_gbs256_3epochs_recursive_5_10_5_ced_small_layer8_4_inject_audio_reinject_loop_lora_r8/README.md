# CED-Small 非 MeSH：input → CED 8→4 + 两轮独立 LoRA

本目录完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject`。
本版本的 CED 顺序为 **8→4**，入口是 `run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8_5090.sh`。
另一个顺序有自己的完整目录、启动器、输出根和恢复身份。
本地 Windows 只维护代码、测试和文档，通过 GitHub 同步；权重、数据、GPU 正式训练位于远程 Linux。
原母本、原启动器和共享 `recursive_model_5_10_5.py` 保持。

## 模型与 LoRA 范围

20 个物理层，30 次逻辑执行，轨迹 `0..14,5..14,15..19`，无 MeSH memory/router。
前五层 0..4 和后五层 15..19 不加 LoRA，继续全量训练。
仅中间十个物理层 5..14 的以下投影加 LoRA：

- Attention：q_proj、k_proj、v_proj、o_proj。
- MLP：gate_proj、up_proj、down_proj。

每个物理层、每个投影分别保存两套 A/B；第 1 轮逻辑 5..14 用索引 0，第 2 轮逻辑 15..24 用索引 1。
两轮继续共享同一个可训练 W/bias，对完整序列计算：

```python
y = linear(x, W, bias) + (16 / 8) * linear(linear(x, A[loop]), B[loop])
```

rank=8，alpha=16，scale=2，LoRA dropout=0，A 用 Kaiming uniform（a=sqrt(5)），B=0。
LoRA max_lr=6e-4，矩阵 AdamW decay=0.01；共享 LM max_lr=3e-4。
所有参数始终可训练，不在循环之间切换 requires_grad；每次投影前向捕获自己的 A/B 张量。
每次层调用后清除选择索引，异常时也清除；不合并 adapter，不修改基础 weight.data。
保留无 KV cache、全序列重算；不支持 decoder gradient checkpointing（母本默认关闭）。

在真实 SmolLM2-135M 形状（hidden576、MLP1536、Q9/KV3 heads）下：

| 参数 | 数量 |
|---|---:|
| 单层、单轮 LoRA | 81,408 |
| 十层单套 LoRA | 814,080 |
| 两套新增 LoRA | 1,628,160 |
| 保留的 serial audio 分支 | 2,777,859 |
| 相对普通 CED 循环母本新增合计 | 4,406,019 |

正式 trainer 按实际加载的投影形状校验参数数，不符合时拒绝训练。
严格 from_pretrained 检查全部 missing/unexpected/mismatched/error 后，才注册原音频注入模块及 LoRA。
继承的 weight/bias 对象与 checkpoint key 保持；LoRA 使用独立 CPU fork_rng，不扰动原 mapper、音频分支初值或后续 RNG。
两版本在相同 seed 下对应的主干、音频分支及 LoRA 初值一致；同一版本两轮的 A 独立随机初始化。

## 保留的音频输入与串行注入

CED 冻结且所有子模块 eval，每批编码一次，同时取 final、完整 block4 和 block8 最终 norm 前输出。
原 freq_merge_126 mapper 得到 E `[B,126,576]`；E 仍是 LM 输入音频前缀。
A4/A8 分别经独立 LN1536 → Linear1536/576 → GELU → Dropout0 → Linear576/576 → LN576。
每个完整 LM forward 从原 E 一次计算共享 `A_input=audio_linear(E)`，不 detach。
每个入口依次：先共享 input gate 残差，再由更新后的 hidden 计算该 CED 分支 gate。

| 入口零基逻辑索引 | 第一步 | 第二步 | 中段 LoRA |
|---|---|---|---|
| 5 | input embeddings 残差 | CED8 残差，读取 post-input hidden | 第 1 套 |
| 15 | 同一 input 投影/gate 的残差 | CED4 残差，读取 post-input hidden | 第 2 套 |

接口始终 `(A4,A8)`，两套 MLP/gate 注册次序始终 4、8；仅入口选择改变。
音频残差只直接更新前 126 个位置，LoRA 作用于该轮完整序列；无第三次注入。
三个音频 gate 仍零初始化，共享 input 投影仍默认随机 weight/零 bias。
初始零 B、零 gate 时与母本相应参考计算对齐。
首步要求两轮 B 都有任务梯度，A 任务梯度为零；第一次更新后要求两轮 A/B 都有有效任务梯度。
保留 gate 关闭时音频上游梯度为零、打开后再审计的规则；权重衰减造成的变化不作为任务梯度证据。

## 与母本相同的训练配置

文本初始化仅继承权重/tokenizer：
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244`。
新音频 S1 的 optimizer/scheduler/scaler/GradNorm/RNG/游标从头建立。

- 原 hybrid 1,477,492 行、全部 source、manifest 与 resolved_paths。
- fixed20 原始音频在线处理；683-token 布局（126 audio+1 SEP+256 prompt+300 answer），prefix383。
- PAD `!`/17，EOS/SEP0；attention_mask=None，answer-only rank-local token mean → GA平均 → DDP平均。
- FP32，无 autocast/TF32，保留 GradScaler；8 GPU × microbatch8 × GA4 = GBS256。
- 3 epochs，5771 updates/epoch，17313 total，866 warmup，每 epoch 丢116行。
- 共享 LM max_lr=3e-4；mapper、音频 MLP/gate/input投影及两套 LoRA max_lr=6e-4。
- 原 AdamW betas(.9,.999)/eps1e-8、矩阵 decay.01、bias/ndim≤1不衰减；同一 LambdaLR 与 GradNormTracker。
- seed1234、rank RNG=1234+rank、原采样顺序；共享音频立即开训，默认4线程后台复制、optimizer边界八卡共同切换。
- 保存 step2886/5771/8656/11542/14428/17313，以及 epoch3。

## 远程正式提交

本地代码完成 GitHub 提交/推送后，在远程登录节点同步并执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8_5090.sh --stage-workers 4
```

直接通过 vc submit 提交 formal；无自动 smoke/tensor 前置任务。
环境 mellow_c8204d8；pdgpu-5090 / 32CPU / 256G / 8GPU / 1node。
登录节点只做资产检查、CPU metadata 索引和提交；CUDA/torchrun 在计算节点运行。
保留 --output-dir、--resume-checkpoint、--resolved-paths、--text-model-dir、--ced-model-dir、--index-cache-root、--stage-workers 参数。

## 输出与完整恢复

本路线 route：`ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8_v1`。
默认输出根：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8_5090/formal_<timestamp>/
```

run root 内 `formal/runtime.json` 记录注入顺序和完整 LoRA 合同；
`formal/config_audit.json` 记录真实 LoRA 数量、全部 optimizer groups 和12类参数审计。
`first_gradient_rank<N>.json`、`opened_gate_gradient_rank<N>.json` 包含两轮 LoRA B 关闭状态与执行轨迹。
最终 checkpoint 是 `formal/checkpoints/model--step-17313.ckpt` 和 `model--epo-3.ckpt`。

两版本的 route/output/log/job/index-cache/tmpfs 和完整 resume 合同隔离。
仅接受本版本同源码/资产/数据/配方、原输出根的完整 formal checkpoint。
不能用另一个顺序、无LoRA母本或其他路线的音频 checkpoint full resume。
恢复包括基础模型、全部 LoRA/音频分支、AdamW、LambdaLR、GradScaler、GradNormTracker、8rank RNG及游标；不重置已训练 LoRA。

将 RUN_ROOT 改成本版本需要恢复的原 run root：

```bash
RUN_ROOT=/absolute/path/to/this_84_loop_lora_run
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8_5090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

## 验证范围

仓库根目录统一验证：

```bash
python -B tests/validate_mizar_s1_ced_loop_lora.py
```

覆盖两版本实际源码 NumPy/stdlib 行为、非零 LoRA 轮次、scale2、post-input gate、有限差分、
优化器12类分组/全部LR位置、梯度审计失败用例、跨路线恢复拒绝、两启动器 Bash 假提交、staging隔离及母本回归。
完整 backbone 与 train 函数经过明确新增节点剥离后与母本 AST 对照；34个母本文件 SHA 保护。
18个复制源文件除隔离命名外完全保持。
本地149项实际检查通过，51项真实CPU tensor检查因缺依赖跳过（新版本共14项）。
结果见仓库 `audit_reports/mizar_s1_ced_loop_lora_validation.json`，本目录 `reference_audit.json` 保存源码对照。

缺 Torch/Transformers 时真实 tensor 测试明确 skip，不代表已做 PyTorch autograd、真实权重或GPU验证。
已有依赖的服务器环境可手动执行（不会由 formal 自动触发）：

```bash
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate mellow_c8204d8
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
python -B tests/validate_mizar_s1_ced_loop_lora.py --require-tensors
```

真实 CPU tensor 检查覆盖 A/B 独立存储、W对象继承、RNG、零LoRA输出及基础梯度对照、
非零逐投影路由、首步/更新后所有矩阵梯度、严格权重继承和完整checkpoint下一步一致性。
checkpoint 的 collective/scaler 用明确CPU fixture，不等同于真实八卡恢复。
本次未提交远程作业、未执行远程资产加载或GPU训练；未登记远程PASS。
现有评测器需要为双 LoRA 和对应注入顺序另行适配。
