# 纯三循环文本 checkpoint → CED-Small 音频训练

这是独立的 `5+10×3+5` 非 MeSH 音频路线，代码、route、输出、日志、job、索引缓存与 tmpfs 均独立。完整复制双循环 `layer4_8_inject_audio_reinject_loop_lora_r8` 音频母本后，只扩展循环、注入入口和 LoRA 分组；原路线及共享文本实现保持原样。

默认文本来源固定为用户指定的：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x3_5/formal_1epoch_bf16_20261109_153922858684473/checkpoint-009244
```

无需再次转换。严格加载该目录的全部 LM 权重和 tokenizer，再注册全新音频模块及三组 LoRA。检查 config、纯三循环身份、9244 步最终 checkpoint 的 metadata/manifest/complete JSON；不反序列化文本 `training_state.pt`。音频 optimizer、scheduler、GradScaler、GradNormTracker、RNG 和数据游标从 step0 新建。

## 模型与注入

20 个物理 decoder 层执行 40 次：`0..4 → (5..14)×3 → 15..19`。全部基础 LM 参数继续训练，中间层基础 W/bias 在三轮共享。

| 中间循环 | 逻辑索引（0 起） | 进入该轮前的注入 | LoRA 参数组 |
| --- | --- | --- | --- |
| 第 1 轮 | 5..14 | audio input → CED block4 | A/B.0 |
| 第 2 轮 | 15..24 | audio input → CED block8 | A/B.1 |
| 第 3 轮 | 25..34 | audio input | A/B.2 |

- CED-Small 冻结并保持 eval，每批一次 encoder forward，提取 final、完整 block4/8 的 norm 前输出。原 `freq_merge_126` mapper 和两个独立 CED MLP/gate 均新初始化。
- 原 mapped audio `E:[B,126,576]` 同时作为输入前缀；共享 `audio_linear(E)` 每次完整 LM forward 只计算一次。入口 `[5,15,25]` 共用 input gate，分别读取该入口当前 hidden；不 detach。
- CED 中间层入口只在 `[5,15]`，顺序 4→8，各自 gate 读取 input 残差更新后的 hidden。第三轮没有 CED 中间层残差。
- 所有音频残差只直接更新前 126 个位置；LoRA 作用全序列。
- 仅物理层 5..14 的 q/k/v/o、gate/up/down 七投影增加三套独立 A/B，`rank=8, alpha=16, scale=2, dropout=0`。A Kaiming、B 零初始化，使用独立 CPU RNG；基础 W/bias 保留原对象。前后五层无 LoRA。
- 生产模型三组 LoRA 共 420 个张量、2,442,240 参数；串行音频分支保持 2,777,859 参数，两者合计 5,220,099（mapper 按母本另计）。
- 无 KV cache；decoder gradient checkpointing 关闭。每次调用后清理 LoRA 选择；首步及更新后梯度审计覆盖三组 A/B。

主要实现：`models/recursive_layer_inject.py`、`models/loop_lora.py`、`models/recursive_text.py`、`optim.py`、`train.py`。合同与身份在 `contract.py` / `write_config.py`。

## 完整继承的音频训练配方

| 设置 | 数值/语义 |
| --- | --- |
| 数据 | hybrid 1,477,492 行，原 manifest / resolved_paths.tsv / 采样顺序 |
| 序列 | fixed20 音频；126 audio + 1 SEP + 256 prompt + 300 answer = 683 |
| 文本与 loss | PAD `!`/17，SEP/EOS 0；LM mask=None；仅 answer loss，rank 内 microbatch 有效 token 均值 → GA 均值 → DDP 均值 |
| 训练规模 | 8 GPU × micro8 × GA4 = GBS256；3 epochs |
| 步数 | 每 epoch 5,771，合计 17,313；每 epoch 丢弃 116 行；warmup 866 |
| 学习率 | LM maxLR 3e-4；mapper / 音频注入 / 三组 LoRA maxLR 6e-4 |
| 更新 | 原 AdamW (.9,.999)/eps1e-8，矩阵 WD .01，bias/低维 WD0；原 LambdaLR floor .1 与 GradNormTracker |
| 精度 | FP32，无 autocast/TF32，保留原 GradScaler |
| 随机种子 | 1234，rank RNG 为 1234+rank |
| checkpoint | step 2886/5771/8656/11542/14428/17313；另存 epoch3 |
| 作业 | pdgpu-5090，8 GPU / 32 CPU / 256G / 1 node，原镜像与 mellow_c8204d8 |

共享音频立即开始训练，默认 4 workers 后台复制；在 optimizer 边界全 rank 共同切换到本地音频，先停止并 reap producer 再清理 tmpfs。沿用原资产检查与 CPU 索引准备后直接正式训练，不增加 smoke 前置。

## 正式训练提交（远程 Linux 登录节点）

本地仅维护核心代码；权重、数据和训练输出均在远程。先按下节提交推送，再在远程执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8_5090.sh --stage-workers 4
```

入口默认使用上面的 checkpoint-009244。可选 `--output-dir` 指定全新空目录，或显式传入 `--text-model-dir`、`--ced-model-dir`、`--resolved-paths`、`--index-cache-root`。保持原音频配方，不能在该入口改 GBS/epoch/精度。

隔离身份：

```text
route: ced_small_mizar_s1_fixed20_gbs256_text_x3_layer4_8_audio_reinject_loop_lora_r8_v1
output root: /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8_gbs256_5090/formal_<timestamp>/
actual run: <output root>/formal/
final: <output root>/formal/checkpoints/model--step-17313.ckpt
also: <output root>/formal/checkpoints/model--epo-3.ckpt
runtime: <output root>/formal/runtime.json
report: <output root>/formal/training_report.json
log: code/RSmol/log/mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8_5090.<timestamp>.JOB.log
job prefix: mizar-ced-x3-in48-lr8-g256-
index cache: /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8_gbs256_5090/index_cache
tmpfs: /dev/shm/mizar_s1_gbs256_ced_text_x3_in48_loop_lora_r8_<token>
```

job ID 和输出路径由入口打印。最终须以 `training_report.json` 的 PASS 与 17313 个更新确认完成。现有评测 registry 尚未登记这条三循环音频路线，需要后续单独适配。

## 中断后完整恢复

仅允许本新路线、相同源码/文本来源/资产/数据/依赖/配方的完整音频 checkpoint，并沿用该次训练原 output root。下面两个路径需替换为实际同一次 run 的路径；output root 是 `formal_<timestamp>`，不再追加第二个 `formal`。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN_ROOT='/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8_gbs256_5090/formal_<实际时间戳>'
AUDIO_CKPT="$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt"
bash code/RSmol/run_mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8_5090.sh \
  --output-dir "$RUN_ROOT" --resume-checkpoint "$AUDIO_CKPT" --stage-workers 4
```

恢复 LM、三组 LoRA、音频分支、optimizer/scheduler/scaler/GradNorm、八 rank RNG 和数据游标，不重置已经训练的参数。双循环、MeSH 或其他来源/配方的 checkpoint 被身份校验拒绝。

## 本地 Git 提交与推送（PowerShell）

仅纳入本次新隔离目录、入口、回归与项目记录，不执行 `git add .`：

```powershell
cd C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
git add -- RM.md code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8 code/RSmol/run_mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8_5090.sh tests/mizar_text_x3_support.py 'tests/test_mizar_s1_ced_text_x3_loop_lora_*.py' tests/validate_mizar_s1_ced_text_x3_loop_lora.py
git add -f -- audit_reports/mizar_s1_ced_text_x3_loop_lora_validation.json
git -c core.whitespace=cr-at-eol diff --cached --check
git diff --cached --stat
git commit -m "Add isolated three-loop text checkpoint CED audio training"
git push origin main
```

## 验证与边界

```powershell
python -B tests/validate_mizar_s1_ced_text_x3_loop_lora.py
```

在已有 Torch/Transformers 的环境可手动加 `--require-tensors`，缺依赖时将失败而不是把跳过当通过。所有输入均为本地合成 fixture，不读生产 checkpoint/CED/音频，不用 CUDA、不调用真实 vc，不是正式训练前置步骤。

验证覆盖 36 份母本/共享文件 SHA、16 份未变运行源码、完整 train() AST 与 loss/更新/保存恢复语义、全 17314 个 LR 位置、纯三循环 checkpoint 身份、step0 新状态与完整恢复 rank RNG、三组 LoRA 非零数值与失败轨迹、真实 Bash 提交参数及 tmpfs 守卫、并发复制和集体切换。真实 CPU tensor 项目另检查纯三循环零增量一致、实际 forward 注入顺序/一次投影、梯度、严格文本加载及恢复后的下一步更新一致。

本次本地通过/跳过明细以 `audit_reports/mizar_s1_ced_text_x3_loop_lora_validation.json` 为准。本机缺 Torch/Transformers，真实 tensor 项明确跳过；未声明生产权重加载、八卡训练或八 rank 恢复通过。`source_guard.json` 保护母本；`reference_audit.json` 是本路线当前源码审计，`parent_reference_audit.json` 仅为母本历史记录。
