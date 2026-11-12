# 文本 epoch1 已训练双 LoRA → CED4→8 音频 S1

本路线直接从用户指定的文本 checkpoint 继承完整 LM 和两套已训练 LoRA，开始新的音频训练：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_loop_lora_r8_gbs1024/epoch1/checkpoint-step-009244
```

无需转换，也无需先完成文本 epoch2。正式入口：
`code/RSmol/run_mizar_s1_ced48_text_epoch1_loop_lora_r8_5090.sh`。
本目录完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8`，
对应母本入口 `run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8_5090.sh`。
母本和文本路线保持原样，37份参考文件的归一化 LF SHA256 见 `source_guard.json`。
本目录 `.gitignore` 仅额外放行 `source_guard.json` 和 `parent_reference_audit.json`，确保小型审计清单随GitHub同步。
本地 Windows 维护核心代码，通过 GitHub 同步；权重、数据、训练输出及 GPU 作业均在远程 Linux。

## 权重继承与音频初始化

- 20物理层、30次执行：前5 → 中10第0轮 → 中10第1轮 → 后5，保持全部基础 LM 权重与 embedding/head 绑定。
- 中间物理层5..14的 q/k/v/o、gate/up/down 七投影使用两套独立 LoRA，rank8、alpha16、scale2、dropout0；前后五层无 LoRA。
- 两套共280个 A/B 张量、1,628,160参数，全部直接继承文本 checkpoint，继续全量训练。
- 显式模型类在 HF 权重加载前建立对应 A/B 槽位；加载后不再安装 LoRA，不重置任何 A/B。
- 严格拒绝 missing/unexpected/mismatched/error；音频模块注册前后均逐项比较源 HF 张量的名称、形状、FP32数值及有限性，检查绑定关系和全部280个 A/B。
- 新初始化原 freq_merge_126 mapper、CED4/8独立 MLP、两个 CED gate、共享 input projection/gate；三个 gate 均为零。CED 从原 `models/CED_small` 资产加载，冻结且 eval。
- 音频 optimizer、scheduler、GradScaler、GradNormTracker、各rank RNG及数据游标从step0开始；不反序列化文本 `training_state.pt`。

每次完整 forward 从输入音频 E 一次计算共享投影，不 detach。零基逻辑索引5：input残差后加CED4残差；索引15：同一input分支后加CED8残差。
CED gate读取已经加过input残差的hidden；两套CED MLP/gate独立。直接音频残差只更新前126位置，LoRA作用于完整序列。
所有继承 LM/LoRA 及新音频模块都可训练，只有 CED 冻结。关闭 KV cache、decoder gradient checkpointing。
首步允许并保留已训练的非零 B，此时两套 A/B 都须有有效任务梯度；保留零音频 gate 上游首步零梯度、打开 gate 后再审计的母本规则。

## 保持母本音频训练配方

| 项目 | 设置 |
|---|---|
| 数据 | hybrid 1,477,492行，原manifest、resolved_paths及全部source |
| 音频/序列 | fixed20；126audio + 1SEP + 256prompt + 300answer = 683token |
| tokenizer/loss | 已有 `!`/17作PAD，EOS/SEP0；attention_mask=None；answer-only rank-local token mean → GA4 → DDP平均 |
| batch | 8GPU × microbatch8 × GA4 = GBS256 |
| 训练长度 | 3epochs，5771updates/epoch，17313total，866warmup；每epoch丢116行 |
| 学习率 | 基础LM maxLR3e-4；两套LoRA、mapper及全部音频分支 maxLR6e-4 |
| optimizer | 原AdamW(.9,.999)、eps1e-8；矩阵WD.01、bias及低维参数WD0 |
| scheduler/clip | 原LambdaLR，floor.1；原GradNormTracker |
| 精度/seed | FP32前向，无autocast/TF32，保留GradScaler；seed1234及rank+seed |
| 音频读取 | 共享原始音频立即训练，默认4workers后台复制，optimizer边界全rank共同切换 |
| 保存 | step2886/5771/8656/11542/14428/17313，以及epoch3 |
| 环境/资源 | mellow_c8204d8，原镜像，pdgpu-5090 / 32CPU / 256G / 8GPU / 1node |

文本 config 的 `lr_policy=same_as_base` 仅保留为来源元数据；音频 optimizer 实际使用上表中的分组学习率和衰减。
训练循环、数据、optimizer、保存及完整恢复逻辑沿用母本；不添加正式训练前的 smoke/tensor 任务。
登录节点保留原CED资产检查、CPU metadata索引准备，然后直接 `vc submit`；计算节点运行 torchrun。

## 正式提交

本地提交并推送后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced48_text_epoch1_loop_lora_r8_5090.sh --stage-workers 4
```

默认文本来源就是本页开头的 epoch1 checkpoint。默认输出根：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced48_text_epoch1_loop_lora_r8_5090/formal_<timestamp>/
```

需要固定输出位置时，使用一个新的空目录：

```bash
RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced48_text_epoch1_loop_lora_r8_5090/formal_$(date +%Y%m%d_%H%M%S)
bash code/RSmol/run_mizar_s1_ced48_text_epoch1_loop_lora_r8_5090.sh \
  --text-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_loop_lora_r8_gbs1024/epoch1/checkpoint-step-009244 \
  --output-dir "$RUN_ROOT" --stage-workers 4
```

保留 `--resolved-paths`、`--ced-model-dir`、`--index-cache-root` 参数。`--text-model-dir` 仅接受此文本双LoRA路线的完整step9244 checkpoint。

## 输出与完整恢复

route为 `ced_small_mizar_s1_ced48_text_epoch1_loop_lora_r8_v1`；output/log/job/index-cache/tmpfs独立。
tmpfs为 `/dev/shm/mizar_s1_gbs256_ced48_text_epoch1_loop_lora_r8_<token>`。
2026-11-12修正此前缺失的 `mizar_s1_gbs256_` 前缀：原路径会在 `stage_data.py` 的启动和清理校验处均被拒绝。
保留原 `/dev/shm` 直属、非软链接、解析路径一致和owner/token检查；路线后缀和每次作业token仍独立。
若旧作业仅因此在发布TRAINING_READY前失败、未进入torchrun，同步修复后用上述正式命令重新提交到新输出目录。
沿母本在run root下追加 `formal/`：

- `formal/runtime.json`：完整音频配方及文本来源身份。
- `formal/config_audit.json`：源张量/LoRA SHA及数值继承审计、继承LoRA参数数、新初始化音频参数数、optimizer groups。
- `formal/first_gradient_rank<N>.json`、`formal/opened_gate_gradient_rank<N>.json`：首步及打开gate后的梯度/执行轨迹。
- `formal/training_report.json`：正式训练实际完成状态。
- 最终权重：`formal/checkpoints/model--step-17313.ckpt`、`formal/checkpoints/model--epo-3.ckpt`。

音频完整恢复要求本路线相同源码、资产、数据、配方，复用原run root：

```bash
RUN_ROOT=/absolute/path/to/this_original_audio_run_root
bash code/RSmol/run_mizar_s1_ced48_text_epoch1_loop_lora_r8_5090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

恢复LM、双LoRA、全部音频模块及optimizer/scheduler/scaler/tracker、8rank RNG/游标；每次作业重新从shared开始staging。
拒绝母本、新初始化LoRA版本、其他注入顺序及其他路线checkpoint的完整resume。文本checkpoint用于新的音频初始化，不能传给 `--resume-checkpoint`。

## 本地验证范围

```bash
python -B tests/validate_mizar_s1_ced48_text_epoch1_loop_lora_r8.py
```

本机42项通过、6项真实CPU tensor检查因缺 torch/transformers/safetensors 跳过，详见
`audit_reports/mizar_s1_ced48_text_epoch1_loop_lora_r8_validation.json`。
检查37份参考SHA、完整模型及训练循环受限差异、整个音频配方和17314个LR位置、严格加载及非零A/B继承、
首步梯度审计、跨路线恢复拒绝、共享读取/后台复制/切换/清理、Bash假vc提交资源和参数。
新增4项staging目录回归：真实Bash赋值交给原启动/清理函数，重现旧前缀拒绝、验证新前缀和作业隔离，
并验证越界/嵌套/软链接/解析别名及错误owner/token仍被拒绝；POSIX文件查询/rmtree为fixture，不触碰真实tmpfs。
`reference_audit.json` 为本路线源码对照；`parent_reference_audit.json` 仅保留母本历史审计。

远程已有依赖的环境可选运行合成CPU检查，不读取生产资产，不提交作业，也不作为formal前置：

```bash
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate mellow_c8204d8
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
python -B tests/validate_mizar_s1_ced48_text_epoch1_loop_lora_r8.py --require-tensors
```

真实tensor用例包括：文本类保存非零双A/B后由音频公开loader加载（含shards/tied weights），
零音频gate下logits与相同文本模型逐值比较，缺A/B严格拒绝，音频注册保留RNG/全部预训练张量且禁止重置，
逐投影非零LoRA轮次及反传、全部参数角色梯度/optimizer、完整CPU checkpoint下一步往返。
checkpoint collectives/scaler为明确CPU fixture；本地验证不代表真实权重加载、GPU训练或8rank恢复已验证。

## 本地 Git 提交命令

PowerShell中执行以下命令，只暂存本次新路线。工作区 `RM.md` 另有其他记录，按块选择本路线记录：

```powershell
Set-Location C:/Xlance/GZ_bridge/Recursive_SALM/RSM_bridge
git add -- code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced48_text_epoch1_loop_lora_r8 code/RSmol/run_mizar_s1_ced48_text_epoch1_loop_lora_r8_5090.sh
git add -- tests/test_mizar_s1_ced48_text_epoch1_loop_lora_r8_identity.py tests/test_mizar_s1_ced48_text_epoch1_loop_lora_r8_route.py tests/test_mizar_s1_ced48_text_epoch1_loop_lora_r8_behavior.py tests/test_mizar_s1_ced48_text_epoch1_loop_lora_r8_shared_start.py tests/test_mizar_s1_ced48_text_epoch1_loop_lora_r8_staging_roots.py tests/test_mizar_s1_ced48_text_epoch1_loop_lora_r8_tensor.py tests/validate_mizar_s1_ced48_text_epoch1_loop_lora_r8.py audit_reports/mizar_s1_ced48_text_epoch1_loop_lora_r8_validation.json
git add -p -- RM.md
git diff --cached --stat
git diff --cached --check
git commit -m "Fix CED48 pretrained LoRA audio staging prefix"
git push
```

本实现未代为commit/push或提交真实GPU作业。现有评测器是否兼容本路线需另行核对。
