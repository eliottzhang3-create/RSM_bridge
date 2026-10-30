# MeSH CED 第 4、8 层 read 后残差注入 / Mizar S1 GBS256

本路线完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small`。
保留母本的 MeSH 文本初始化、音频 prefix、数据与训练配方，在两次中段循环各自的 memory read
之后分别注入 CED 第 4、8 层信息。残差方法来自非 MeSH
`mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject`。
本地仅维护代码，真实资产和训练位于远程 Linux，经 GitHub 同步。

## 精确架构

```text
原始 audio embeddings / SEP / prompt / answer
  -> 原 MeSH prelude 5 层
  -> 原 write_routers[0] / read_routers[0] / memory read
  -> 前 126 个 audio hidden 加第 4 层残差
  -> 第一次中段 10 层循环（逻辑索引 5..14）
  -> 原 write_routers[1] / read_routers[1] / memory read
  -> 前 126 个 audio hidden 加第 8 层残差
  -> 第二次中段 10 层循环（逻辑索引 15..24）
  -> 原 write_routers[2] / read_routers[2] / memory read
  -> 原 suffix 5 层、final norm、LM head
```

20 个独立物理层、30 次逻辑执行，轨迹仍是 `0..14,5..14,15..19`。
5 个瞬时 memory slots、3 组/6 个独立 write/read routers 均保留。
最初 memory 第 0 槽仍为原始 input embeddings；新残差直接作用于读取后的 hidden，
不会额外直接写 memory。后续循环、router query 和 memory write 正常承接此 hidden；
中段后的 write/read query 仍取该循环的输入，不改为 core output。
第三次 memory read 后进入 suffix，不增加第三次注入。

## 与原 4→8 方法一致的音频分支

- 冻结且始终 eval 的 CED-small，在同一次编码里提取第 4、8 个 block 的完整输出，位于 CED 最终 LayerNorm 之前。
- `[2B,252,384]` 先恢复 `[B,2,4,63,384]`，再按 segment/time/frequency 整理为 `[B,126,4,384]`。
- 每层独立使用 `LN1536 -> Linear(1536,576) -> GELU -> Dropout(0) -> Linear(576,576) -> LN576`，得到 `[B,126,576]` 的 A4/A8。
- 原 CED final-layer mapper / `freq_merge_126` 主路径及其 dropout0.05 不变。
- 两个独立 `Linear(576,1)` gate，weight/bias 都初始化为零。

```text
h_audio = h_read[:, :126]
h_new_audio = h_audio + tanh(gate_layer(h_audio)) * A_layer
h_new = concat(h_new_audio, h_read[:, 126:])
```

每个 audio token 一个标量 gate；SEP 和文本不直接注入。非原地拼接，
原始 MLP/LN/激活/初始化与 gate 公式完整复用，不添加残差缩放或其他归一化。
新增 MLP 位于冻结 CED 的 no_grad 范围之外，完整保留梯度。
新增参数共 2,444,930：两套 MLP 2,443,776，加两个 gate 1,154。
先严格加载 MeSH 全部继承权重（包括六个 routers），再注册新 gate。
新增 MLP 与 gate 初始化使用 CPU fork_rng；不改变母本 mapper 初值及后续 CPU RNG。

每个 forward 显式传入 `(A4,A8)`，不缓存样本音频状态；只接受完整 prefix 的
inputs_embeds、`use_cache=False`，拒绝 KV cache 和缺失/错误 shape、dtype、device 的音频信息。
未来独立评测必须每样本编码一次，并在每个 token 的完整重算中显式传入同一对 A4/A8。
现有普通 MeSH CED evaluator 尚未适配本新架构。

## 保留的正式训练配置

| 项目 | 配置 |
|---|---|
| 文本初始化 | 原 MeSH `formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244` |
| 数据 | hybrid 1,477,492 行，保留 MusicCaps 和全部 SpeechCraft splits |
| 波形与 CED | 原在线 fixed20、逐样本 dB、冻结 BN、两段 1012 帧 |
| 序列 | audio126 + SEP1 + prompt256 + answer300 = 683，prefix383 |
| Loss/attention | 母本 answer mask + 非 PAD 的 rank-local token mean；attention_mask=None |
| Batch | 8 GPU × microbatch8 × accumulation4 = GBS256 |
| 更新数 | 3 epochs，5,771/epoch，17,313 total，866 warmup，每 epoch 丢弃 116 行 |
| max LR | 原 LM（含 routers）3e-4；原 mapper、新 MLP/gate 6e-4 |
| 优化器 | 原 AdamW、bias/一维不衰减规则、LambdaLR、GradNormTracker、GradScaler |
| 数值 | FP32 forward，无 autocast，TF32 关闭 |
| 数据顺序 | 原 seed1234 / rank seed1234+rank、shuffle、sampler |
| staging | 直接共享读取开始 formal，4 线程后台复制，optimizer 边界八卡一致切换 |
| 资源 | mellow_c8204d8，pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node |

新增角色独立分组，不改变原 LM/mapper 参数的 LR 与 decay。
零 gate 的首次反向：gate 有梯度，MLP 梯度为零；gate 首次更新后的下一次反向验证 MLP 梯度。
零梯度不等于参数必然不变，AdamW 的 decay 仍按原规则执行。
首次检查所有可训练参数梯度存在且有限、实际 30 层轨迹和两次注入轨迹；
分别记录六个 routers 的有限性与非零状态，非零状态仅作诊断。
metrics 定期记录 gate 幅度/饱和比例和残差范数，结束核对新增角色更新、CED 参数/buffer 不变及完整重载。

## 远程正式提交

先将本地代码经 GitHub 同步。在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_inject_5090.sh --stage-workers 4
```

直接提交 formal，没有 smoke 或 tensor 检查前置任务。登录节点只核对资产并准备/复用 CPU metadata 索引，
GPU 训练通过 vc wrapper 执行。默认文本 checkpoint 和 CED 目录沿用母本。

默认独立输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_5090/formal_<timestamp>/
```

其中的主要产物：

```text
formal/runtime.json
formal/config_audit.json
formal/first_gradient_rank<N>.json
formal/opened_gate_gradient_rank<N>.json
formal/metrics.jsonl
formal/staging_report.json
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/training_report.json
```

中间保存步骤沿用六等分：2886、5771、8656、11542、14428、17313。
若需要固定输出目录，在提交时附加 `--output-dir /完整/新空目录`。

## 完整恢复

新路线的 route/contract、源码身份、输出、日志、index cache、tmpfs 均隔离。
仅接受本路线同代码、资产、数据和配方的完整音频 checkpoint，恢复全部模型、
optimizer/scheduler、GradScaler/GradNorm、每 rank RNG 和真实数据游标。
母本 MeSH 音频和非 MeSH 4→8 音频 checkpoint 均不能直接 full resume。

```bash
# 替换为本新路线中断的实际 run 根目录与已保存 checkpoint
RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_5090/formal_替换为实际时间戳"
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_inject_5090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

## 验证与证据边界

仓库根目录可选执行：

```bash
python -B tests/validate_mizar_s1_mesh_ced_layer48.py
# 在远程具备依赖的 mellow_c8204d8 环境，禁止真实 tensor 检查被跳过：
python -B tests/validate_mizar_s1_mesh_ced_layer48.py --require-tensors
```

汇总包括：实际 MeSH forward 的 NumPy 行为验证、两条母本源码/AST 与参数/数据配方对照、
严格 loader 拒绝条件、Bash 模拟提交、共享读取/后台复制/退出清理、Python/Bash 语法，
以及使用合成 CED 和小型 MeSH 的真实 CPU tensor 检查。不会下载资产、运行 CUDA 或提交训练。
`runtime_checks.py` 只运行本路线真实 CPU tensor 检查，缺依赖直接失败，不进入 formal 调用链。

当前本地 66 项实际检查通过，16 项真实 CPU tensor 检查因缺少依赖跳过（新路线9项、原4→8路线7项）。
报告：`audit_reports/mizar_s1_mesh_ced_layer48_validation.json`；
完整 MeSH 源码差异审计：本目录 `reference_audit.json`。
真实权重加载、远程 CPU tensor、8 GPU 训练与恢复仍待服务器验证，不登记远程 PASS。
