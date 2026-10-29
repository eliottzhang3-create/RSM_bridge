# CED-Small / 非 MeSH 5+10×2+5 / 第 8→4 层注入 / Mizar S1 GBS256

本路线完整复制隔离自同级 `mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject`。
唯一训练语义变化是：第一次中段循环前注入 CED 第 8 层信息，第二次中段循环前注入第 4 层信息。
原 4→8 路线、普通 CED 路线和共享递归实现均保持原文件。

本地 Windows 仅维护代码、测试和文档；模型权重、数据和训练环境位于远程 Linux，通过 GitHub 同步代码。
本地检查通过不能作为远程权重加载、真实张量计算或 GPU 训练 PASS 的证据。

## 注入结构

CED 层号从 1 开始，取完整 block 输出、位于最终归一化之前：
第 4 层为 `blocks[3]`，第 8 层为 `blocks[7]`。
一次冻结 CED forward 同时提取两层和原最终层特征，不重复运行 encoder。

每层中间特征从 `[2B,252,384]` 按原 segment/frequency/time 顺序整理成
`[B,126,4,384]`，由该层自己的 `1536→576→576` MLP 生成 `A4`、`A8`。
原最终层仍通过原 `freq_merge_126` mapper 构造音频前缀。

| 注入点 | LM 零基逻辑索引 | 使用特征 | 使用 gate |
|---|---:|---|---|
| prelude 五层输出 h0 后，第一次中段循环前 | 5 | A8 | audio_gates["8"] |
| 第一次中段循环输出 h1 后，第二次中段循环前 | 15 | A4 | audio_gates["4"] |

每次仅直接更新前 126 个音频位置：

```text
h_audio ← h_audio + tanh(Linear_576_to_1(h_audio)) * A
```

每个音频 token 一个标量 gate。SEP、prompt 和 answer 不直接注入；
其后仍正常通过因果注意力接收上下文。第二次循环结束后不再注入。
MLP/gate 按层独立，不跨层共享；不增加归一化、dropout 或 detach。

特征接口保持 `audio_information=(A4,A8)`；递归 forward 按反向索引选择 A8、A4。
两套 MLP/gate 的注册和初始化顺序仍为 4、8，保留同 seed 下与母本对应的初值及 CPU RNG。
两组 gate 权重和 bias 全零，MLP 在冻结 CED 的 no_grad 范围之外训练。
新增模块共 2,444,930 个参数，与 4→8 母本相同。
零 gate 时 MLP 梯度为零是预期行为；gate 打开后梯度再进入对应 MLP。

## 保留的训练合同

- 非 MeSH 固定 `5+10×2+5`：20 个物理层、30 次逻辑执行，轨迹 `0..14,5..14,15..19`。
- 文本初始化仍为：
  `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244`。
  只加载文本权重与 tokenizer，音频 S1 的 optimizer/scheduler/RNG/数据游标从头开始。
- 冻结且始终 eval 的 CED-Small，原 final-layer mapper 和完整 LM 的训练行为保持母本。
- hybrid 1,477,492 行；MusicCaps 和三个 SpeechCraft split 均保留。
- 原 fixed20 在线音频、整段 mel/BN 后切分；`126 audio + 1 SEP + 256 prompt + 300 answer = 683`。
- `attention_mask=None`、无 KV cache、原 answer-only CE；answer PAD 不计 loss。
- 8 GPU × microbatch 8 × accumulation 4 = GBS256；3 epochs。
- 每 epoch 5,771 次 optimizer update，共 17,313 次，warmup 866 次；每 epoch 丢弃 116 行。
- FP32；LM max LR 3e-4；mapper、两套 MLP、两套 gate max LR 6e-4。
- 原 AdamW、LambdaLR、GradNormTracker、seed 1234、数据顺序与完整保存/恢复策略。
- 共享原始音频立即开始训练，4 线程后台复制，全部就绪后在 optimizer 边界八卡共同切换。

## 正式提交

本地代码经 GitHub 同步后，在远程登录节点运行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
git pull --ff-only
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_5090.sh --stage-workers 4
```

入口只提交 formal，通过 `vc submit` 申请
`pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`，
使用原容器和 `mellow_c8204d8` 环境，无 smoke 前置任务，也不会自动运行可选张量检查。
登录节点只准备/复用 CPU metadata 索引；CUDA 训练在计算节点启动。

入口保留母本的 `--output-dir`、`--resume-checkpoint`、`--resolved-paths`、
`--text-model-dir`、`--ced-model-dir`、`--index-cache-root`、`--stage-workers` 参数。
CED 默认目录为 `/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small`；
manifest 和 resolved TSV 保持母本远程路径。

## 输出与恢复

独立 route 标识：

```text
ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer8_4_inject_v1
```

默认输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_5090/formal_<timestamp>/
```

run root 内保存：

```text
formal/runtime.json
formal/config_audit.json
formal/first_gradient_rank<N>.json
formal/staging_report.json
formal/metrics.jsonl
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/training_report.json
```

runtime 合同明确记录 `ced_layers_1based=[8,4]` 与 `logical_indices_0based=[5,15]`。
首步梯度审计必须收到 `[[5,8],[15,4]]`。
输出根、日志、job 名、CPU index cache 和 `/dev/shm` 临时目录均独立。
保存步骤保持 2886、5771、8656、11542、14428、17313，并保存最终 epoch checkpoint。

只允许从本 8→4 路线、相同代码/数据/配方的完整 formal checkpoint 恢复。
即使张量名称与形状相同，原 4→8、普通 CED、MeSH CED 或其他路线 checkpoint 仍被 route/合同校验拒绝。

```bash
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_5090.sh \
  --output-dir /absolute/path/to/the/existing/layer8_4_run \
  --resume-checkpoint /absolute/path/to/the/existing/layer8_4_run/formal/checkpoints/model--step-5771.ckpt \
  --stage-workers 4
```

正式恢复使用对应 run root；新训练要求空输出目录。
完整 checkpoint 恢复模型、两套 MLP/gate、optimizer、scheduler、GradScaler、GradNormTracker、
八个 rank 的 RNG，以及 epoch/update 游标。

## 验证和状态

本地统一验证：

```bash
python tests/validate_mizar_s1_ced_layer84.py --output audit_reports/mizar_s1_ced_layer84_validation.json
```

具备依赖的远程环境可手动执行真实小模型 CPU 张量检查：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
python tests/validate_mizar_s1_ced_layer84.py --require-tensors
```

此命令只使用合成数据，不下载权重、不运行 CUDA、不提交训练。
`--require-tensors` 会在缺少依赖或发生测试跳过时失败。
可选 `runtime_checks.py` 也指向本新验证入口，不会在 formal 之前自动执行。

验证覆盖非零 gate 下 8→4 顺序与 gate 配对、零 gate 等价、注入位置、梯度、
原路线 checkpoint 拒绝、完整恢复、Bash 模拟提交与路径隔离。
源代码审计同时比较 4→8 母本，约束变化只涉及顺序、路线命名、相应审计和记录，
并保留普通 CED 母本及共享递归骨干的原有审计。

本地验证 67 项实际执行通过，18 项真实 CPU tensor 检查因依赖缺失跳过；
Python 编译、Bash 语法、模拟提交及新旧路线源码语义对照审计通过。
原 4→8 路线和普通 CED 母本回归已纳入统一验证。
详细结果见根 README 和 validation report；真实远程权重加载与 8 GPU 正式训练尚未验证。
现有评测入口尚未适配本 8→4 架构。
