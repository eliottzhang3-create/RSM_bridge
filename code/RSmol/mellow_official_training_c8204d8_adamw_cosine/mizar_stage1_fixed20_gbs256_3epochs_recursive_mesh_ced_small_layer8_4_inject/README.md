# MeSH CED 第 8→4 层 read 后残差注入 / Mizar S1 GBS256

本路线完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject`。
唯一模型计算差异是交换两次注入：第一次 memory read 后使用 A8/gate8，第二次使用 A4/gate4。
原 4→8 路线、普通 MeSH CED 路线、非 MeSH 8→4 路线和共享 MeSH 实现均保持原样。
本地 Windows 只维护代码、测试和文档；权重、数据与实际训练均位于远程 Linux，通过 GitHub 同步代码。

## 精确架构

```text
原始 audio embeddings / SEP / prompt / answer
  -> 原 MeSH prelude 5 层
  -> 原 write_routers[0] / read_routers[0] / memory read
  -> 前 126 个 audio hidden 加第 8 层残差（逻辑索引 5）
  -> 第一次中段 10 层循环（逻辑索引 5..14）
  -> 原 write_routers[1] / read_routers[1] / memory read
  -> 前 126 个 audio hidden 加第 4 层残差（逻辑索引 15）
  -> 第二次中段 10 层循环（逻辑索引 15..24）
  -> 原 write_routers[2] / read_routers[2] / memory read
  -> 原 suffix 5 层、final norm、LM head
```

20 个独立物理层、30 次逻辑执行，轨迹仍是 `0..14,5..14,15..19`。
5 个瞬时 memory slots、3 组/6 个独立 write/read routers 均保留。
memory 在每次完整 forward 中重建，第 0 槽仍为原始 input embeddings，其余槽初始为零。
新残差只直接作用于 memory read 后的 hidden，不额外直接写入 memory；
后续循环与普通写入照常承接该 hidden。循环后的 router query 仍取循环输入，写入值仍取循环输出。
第三次 memory read 后直接进入 suffix，无第三次注入。
实际注入轨迹为 `[[5,8],[15,4]]`，read router 顺序仍是 `[0,1]`。

## 保持一致的音频分支与初始参数

- 冻结且始终 eval 的 CED-small，一次编码同时提取第 4、8 个 block 的完整输出，位于 CED 最终 LayerNorm 之前。
- `[2B,252,384]` 恢复为 `[B,2,4,63,384]`，再按 segment/time/frequency 整理为 `[B,126,4,384]`。
- 两层分别经过独立的 `LN1536 -> Linear(1536,576) -> GELU -> Dropout(0) -> Linear(576,576) -> LN576`，得到 A4/A8。
- 原 CED final-layer mapper / `freq_merge_126` 主路径及 dropout0.05 保留。
- 两个独立 `Linear(576,1)` gate，weight/bias 全零初始化；gate4 始终对应 A4，gate8 始终对应 A8。
- MLP 与 gate 注册、初始化和参数遍历顺序仍为 **4、8**，保留 CPU fork_rng、原 mapper 初值和后续 RNG。
- `models/ced.py`、`models/mellow.py`、`models/decoder.py`、`optim.py` 及全部数据/staging 实现与 4→8 母本内容一致。

每个 forward 的显式接口仍是 `(A4,A8)`：第一次选择索引 1，第二次选择索引 0。
注入公式保持原样，每个 audio token 一个标量 gate：

```text
h_audio = h_read[:, :126]
h_new_audio = h_audio + tanh(gate_layer(h_audio)) * A_layer
h_new = concat(h_new_audio, h_read[:, 126:])
```

不做原地修改，SEP/文本不直接注入；不额外添加归一化、残差缩放、dropout 或 detach。
新增 MLP 位于冻结 CED 的 no_grad 范围之外，保留训练梯度。
新增参数仍为 2,444,930：两套 MLP 2,443,776，加两个 gate 1,154。
严格加载原 MeSH 全部文本权重及六个 routers 后才注册 gate。
零 gate 时初始 logits 与母本一致；gate 更新后训练轨迹随注入顺序改变。

只接受完整 prefix 的 inputs_embeds、`use_cache=False`，不保存样本级音频状态。
显式校验 A4/A8 的 shape、dtype、device，拒绝 KV cache 和缺失音频信息。
`generate_prefix_inference` 仍返回 prefix、音频输出及 `(A4,A8)`；未来评测需要每样本编码一次，
然后在每个生成 token 的完整重算中显式传入同一对特征。
当前 MMAU/MMAR 评测入口尚未注册本新路线，不能直接用 4→8 variant 加载新 checkpoint。

## 保留的正式训练配置

| 项目 | 配置 |
|---|---|
| 文本初始化 | MeSH `formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244` |
| 数据 | hybrid 1,477,492 行，保留 MusicCaps 和全部 SpeechCraft splits |
| 波形与 CED | 原在线 fixed20、逐样本 dB、冻结 BN、两段 1012 帧 |
| 序列 | audio126 + SEP1 + prompt256 + answer300 = 683，prefix383 |
| Loss/attention | 原 answer mask + 非 PAD 的 rank-local token mean；attention_mask=None |
| Batch | 8 GPU × microbatch8 × accumulation4 = GBS256 |
| 更新数 | 3 epochs，5,771/epoch，17,313 total，866 warmup，每 epoch 丢弃 116 行 |
| max LR | 原 LM（含六个 routers）3e-4；原 mapper、两套 MLP/gate 6e-4 |
| 优化器 | 原 AdamW、bias/一维不衰减规则、LambdaLR、GradNormTracker、GradScaler |
| 数值 | FP32 forward，无 autocast，TF32 关闭 |
| 数据顺序 | 原 seed1234 / rank seed1234+rank、shuffle、sampler |
| staging | 共享路径立即开始 formal，4 线程后台复制，optimizer 边界八卡一致切换 |
| 资源 | mellow_c8204d8，pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node |

新增角色分组和参数名保留；LM/mapper 原参数的 LR、decay、参数顺序均不改变。
零 gate 首次反向检查 gate 梯度，MLP 梯度为零；gate 更新后检查 MLP 梯度。
保留 router 梯度诊断、gate 幅度/饱和比例、残差范数、角色更新、CED 参数/buffer 冻结及完整重载审计。

## 远程正式提交

先将本地改动通过 GitHub 同步。在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer8_4_inject_5090.sh --stage-workers 4
```

直接提交 formal，无自动 smoke 或 tensor 检查前置任务。
登录节点只核对资产并准备/复用 CPU metadata 索引；GPU 训练通过 vc wrapper 执行。
默认沿用原路线的文本 checkpoint、CED 目录、hybrid manifest 和 resolved_paths.tsv。
首次从原文本权重开始新的音频训练，optimizer/scheduler/RNG/数据游标从头开始。

默认独立输出根目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer8_4_inject_5090/formal_<timestamp>/
```

run 根目录中的主要产物：

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

中间保存步骤仍为 2886、5771、8656、11542、14428、17313。
可附加 `--output-dir /完整/新空目录` 指定输出；fresh run 拒绝覆盖非空目录。
入口还支持 `--text-model-dir`、`--ced-model-dir`、`--resolved-paths`、`--index-cache-root`。

## 完整恢复

新路线身份为 `ced_small_mizar_s1_fixed20_gbs256_recursive_mesh_layer8_4_inject_v1`。
route/注入合同、源码身份、默认输出、日志、index cache、tmpfs 目录均独立。
仅接受本 8→4 路线同代码、资产、数据和配方的完整正式 checkpoint。
原 MeSH 4→8、普通 MeSH CED 或非 MeSH checkpoint 均不能直接 full resume。
恢复模型、optimizer/scheduler、GradScaler/GradNorm、各 rank RNG 和真实数据游标；已训练 gate/MLP 不会重置。

```bash
# 替换为本 8→4 路线实际中断的 run 根目录与已保存 checkpoint
RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer8_4_inject_5090/formal_替换为实际时间戳"
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer8_4_inject_5090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

## 本地验证与证据边界

仓库根目录的统一检查入口：

```bash
python -B tests/validate_mizar_s1_mesh_ced_layer84.py
# 在已有依赖环境，要求真实 CPU tensor 检查全部实际执行：
python -B tests/validate_mizar_s1_mesh_ced_layer84.py --require-tensors
```

汇总新路线和原 MeSH 4→8、普通 MeSH CED、非 MeSH 4→8 的回归检查，
包括实际 MeSH forward 的 NumPy 行为、两次 read 后的特征/gate 对应、打开 gate 后的顺序差异、
零 gate 等价性、完整源码/AST 差异白名单、LR/参数角色、严格 loader、恢复身份拒绝、
105 个原路线文件保护、Bash 模拟提交、共享读取/后台复制/退出清理和 Python/Bash 语法。
提供真实 CPU tensor 检查验证初始参数/RNG、捕获特征、读写/注入、梯度和完整 checkpoint 下一步恢复。
不下载资产、不运行 CUDA、不提交作业，也不进入 formal 前置调用链。

`runtime_checks.py` 仅运行本新路线 tensor 检查，缺依赖直接失败。
验证汇总：`audit_reports/mizar_s1_mesh_ced_layer84_validation.json`。
逐文件与共享骨干对照：本目录 `reference_audit.json`。
本地 98 项实际检查通过，26 项真实 CPU tensor 检查因缺少 Torch/Torchaudio/Transformers/SoundFile 跳过：
新路线 10 项、原 MeSH 4→8 9 项、非 MeSH 4→8 7 项。Python/Bash 语法和完整源码对照通过。
真实远程权重加载、CPU tensor、8 GPU 训练和完整恢复尚未执行，未登记远程 PASS。
