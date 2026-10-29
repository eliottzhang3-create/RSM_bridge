# CED-Small / non-MeSH 5+10×2+5 / 两次循环前的音频残差注入

本路线是 `mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small` 的完整隔离副本。
母本入口 `run_mizar_s1_gbs256_recursive_5_10_5_ced_small_5090.sh`、共享递归模型及其他训练路线未修改。
本地只修改代码，经 GitHub 同步到远程 Linux；权重、音频和正式输出都留在远程。

## 新增结构

原主干保持20个物理层、30次逻辑执行，物理层轨迹仍是 `0..14,5..14,15..19`：

```text
mapper 输出 E_audio [B,126,576] ──共享 Linear(576,576)──> A
         │
         └─原样拼接 SEP、prompt、answer embeddings
                │
              前5层
                │ h0
         第一次音频残差注入 ← tanh(共享 Linear(576,1)(h0_audio)) × A
                │
           中间10层，第一次
                │ h1
         第二次音频残差注入 ← tanh(同一个 Linear(576,1)(h1_audio)) × 同一个 A
                │
           中间10层，第二次
                │
              后5层
                │
          原 final norm / LM head
```

实际残差使用非原位拼接，避免破坏 autograd：

```python
A = audio_linear(inputs_embeds[:, :126])  # 每次 forward 只计算一次
# 分别在 logical_index=5、15，即两次中段循环开始前：
audio = hidden_states[:, :126]
gate = torch.tanh(audio_gate(audio))      # [B,126,1]，每个音频位置一个标量
hidden_states = torch.cat(
    (audio + gate * A, hidden_states[:, 126:]), dim=1
)
```

两次注入共用一个投影和一个 gate，gate 根据当次隐藏状态重新计算。权重可正可负，范围为 [-1,1]。
只直接修改前126个音频位置；SEP、prompt、answer 位置不直接注入，之后仍按原因果注意力参与计算。
第二次中段循环结束后不再注入。不添加额外归一化、dropout、缩放或 detach。
A 来自本次原始 mapper 输出，不重新编码音频，也不使用前五层处理后的音频特征作为 A。

新增参数恰为四个 tensor、332,929个参数：

| 参数 | 形状 | 初始化 |
|---|---|---|
| `model.audio_linear.weight` | [576,576] | PyTorch Linear 默认随机均匀初始化 |
| `model.audio_linear.bias` | [576] | 0 |
| `model.audio_gate.weight` | [1,576] | 全0 |
| `model.audio_gate.bias` | [1] | 全0 |

先严格加载并校验原文本 checkpoint，再注册新模块；不放宽原主干的 missing/unexpected/mismatched-key 检查。
新增初始化在 CPU `torch.random.fork_rng(devices=[])` 中完成，恢复调用者 RNG，不推进原模型的随机序列。
CED 和 mapper 的构造顺序保持母本；DDP 继续广播 rank0 的完整初值。

gate 为零时新增残差为零，初始前向应与母本一致。首次反向中 gate 应获得非零梯度，
投影的任务梯度为零；第一次 optimizer update 打开 gate 后，投影应获得非零任务梯度。
零梯度不等于权重完全不更新：投影矩阵仍按母本规则参与 AdamW 的解耦 weight decay。

本模型仅支持包含完整音频 prefix 的 `inputs_embeds`、全序列重算和 `use_cache=False`；
显式拒绝 input_ids-only、KV cache/past 以及不完整音频 prefix。
原有 HTSAT/CED evaluator 尚未接入此架构，不能直接用旧 loader 评测本路线 checkpoint。

## 保留的训练设置与初始化边界

文本初始化仍为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244
```

它只提供文本权重和 tokenizer。音频 S1 从 optimizer step0 重新训练，optimizer、scheduler、
GradScaler、GradNormTracker、RNG 和数据游标均重新开始。不会继承任何旧音频训练状态。
CED 冻结且所有子模块保持 eval；官方 mapper、完整文本主干和新增四个 tensor 均可训练，无 LoRA、router 或 memory slots。

| 项目 | 本路线设置 |
|---|---|
| 数据 | 原 hybrid 1,477,492行及 resolved_paths，保留 MusicCaps 和全部 SpeechCraft splits |
| 音频 | 原 fixed20 中心裁剪/右补零，完整20秒波形经冻结 CED-Small 和官方 freq_merge_126 |
| 序列 | audio126 + SEP1 + prompt256 + answer300 = 683；prefix383 |
| PAD / EOS | 原 Mizar `!` PAD（已核对词表中为17），EOS/SEP=0 |
| attention / loss | `attention_mask=None`；原 answer mask 与非PAD共同选取目标；rank-local token mean、四次累积平均、DDP平均 |
| Batch | 8 GPU × microbatch8 × accumulation4 = GBS256 |
| 时长 | 3 epochs；5,771 updates/epoch；17,313 total；866 warmup；每轮丢弃116行 |
| 原参数 LR | LM 3e-4；mapper 6e-4，保持母本 |
| 新参数 LR | 投影和 gate 各自显式分组，均采用 mapper 的6e-4 |
| AdamW | betas=(0.9,0.999)、eps1e-8；矩阵 decay0.01，bias及 ndim<=1 不衰减 |
| 调度与裁剪 | 原 Mizar LambdaLR、5% warmup/cosine/floor0.1、原 GradNormTracker |
| 精度 | 原 FP32 forward、无 autocast、保留 GradScaler、关闭 TF32 |
| 随机与遍历 | 原 seed1234、rank seed1234+rank、shuffle、连续rank分片及实际跳行恢复 |
| staging | 立即读共享原音频，后台默认4线程复制，全部完成后在 optimizer 边界八卡共同切换 |
| 资源 | mellow_c8204d8；pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node |

数据、CED、mapper、decoder、采样和 staging 等17个文件在统一换行后与母本一致。
去除明确新增的注入节点后，整个递归主干 AST 与原 `recursive_model_5_10_5.py` 一致；
去除新增诊断后，完整 epoch/update/microbatch 循环与母本 AST 一致。
原 optimizer 参数组的 LR/decay、AdamW、LambdaLR 和 GradNormTracker 实现保持原规则。
`reference_audit.json` 只证明文件/AST/配置对照，不证明远程权重加载或训练完成。

## 远程正式提交

经 GitHub 同步后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_audio_reinject_5090.sh \
  --ced-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_audio_reinject_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

入口直接提交 formal，不增加 smoke 或 tensor 报告前置门槛。登录节点仅核对资产、准备/复用 CPU 索引并调用 vc。
真实训练由计算节点激活 `mellow_c8204d8` 后以8卡 torchrun 运行。
默认 manifest 是 `data/Mizar/s1_stage1_hybrid_filtered.json`；
默认 resolved_paths 为 `outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv`。

输出、log 文件前缀、CPU index cache、tmpfs 和 resume 身份均使用独立的 audio_reinject 名称。
新 tmpfs 为 `/dev/shm/mizar_s1_gbs256_recursive_5_10_5_ced_small_audio_reinject_<token>`，
每作业重新 staging，退出仅按 ownership token 清理本作业目录。

## 产物、梯度检查与恢复

指定 output-dir 下的 `formal/` 包含：

```text
runtime.json / config_audit.json / trainable_params.txt
first_gradient_rank<N>.json
opened_gate_gradient_rank<N>.json       # fresh run 的第2次反向
staging_report.json / metrics.jsonl
checkpoints/model--step-17313.ckpt
checkpoints/model--epo-3.ckpt
training_report.json
```

保留原保存节点：2886、5771、8656、11542、14428、17313。
完整 checkpoint 包含 CED、mapper、LM 和新增参数，以及 optimizer、scheduler、GradScaler、
GradNormTracker、8 rank RNG、epoch/update 游标和合同身份。
启动严格审计新结构；fresh run 首次反向检查零门控梯度，第2次反向检查 gate 已打开且投影收到梯度。
前两次及每25次更新记录 rank0 最后一个 microbatch 的两次 gate 绝对值、饱和比例和残差/隐藏状态范数比；
不改变训练 loss 或新增正则项。最终检查新增参数实际变化、CED 参数/buffer 未变及 checkpoint 重载一致。

只有本路线、同源码/数据/资产/配置的完整 formal checkpoint 才允许恢复：

```bash
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_audio_reinject_5090.sh \
  --output-dir /absolute/path/to/this_route_run \
  --resume-checkpoint /absolute/path/to/this_route_run/formal/checkpoints/model--step-5771.ckpt \
  --stage-workers 4
```

沿用母本在同一 output-dir 追加 metrics 的恢复行为；其余可覆盖路径须与原 run 保持相同身份。
母本、MeSH、普通 CED 或 HTSAT checkpoint 不能作为本路线的完整 resume source。

## 本地与远程验证

从仓库根目录运行汇总检查：

```bash
python tests/validate_mizar_s1_ced_audio_reinject.py
```

本地49项行为/合同/NumPy数值/Bash mock及母本回归检查通过；Python编译和两份 Bash 语法检查通过。
7项新增真实 CPU tensor 检查、3项母本 CPU tensor 检查因本地缺少 Torch/torchaudio/Transformers/soundfile 跳过；
继承的16项 runtime checks 同样未执行。详细记录在 `validation_report.json`，其 PASS 范围仅为本地代码验证。

真实 tensor 检查覆盖零门控的原主干输出/梯度等价、A只算一次与gate调用两次、
两处切片、打开门控后的投影/mapper梯度、RNG隔离、实际 from_pretrained 逐 tensor 继承、
新增参数数量，以及 checkpoint/optimizer/scheduler/随机状态恢复后的下一次更新。
CPU 测试的 collective 和 GradScaler 使用明确的 fixture，不代表8卡 CUDA 恢复验证。

可在远程 `mellow_c8204d8` 环境、仓库根目录手动运行，不下载权重、不提交GPU作业：

```bash
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate mellow_c8204d8
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
python tests/validate_mizar_s1_ced_audio_reinject.py --require-tensors \
  --output /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_audio_reinject_cpu_validation.json
```

`--require-tensors` 在依赖缺失、任何测试跳过或检查失败时返回非零，避免把跳过当作张量验收通过。
本地已实际验证此分支：相同49项检查通过、10项张量检查跳过时，严格模式返回退出码1并列出缺失依赖；这是预期的拒绝行为。
该命令是独立可选诊断，不是 formal 的前置任务。真实 CED 加载、8GPU前反向和正式3epochs仍须以远程作业为证据；
当前未提交训练，未登记远程 PASS。

修改实现后须重生成源码审计：

```bash
python code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_audio_reinject/audit_reference.py \
  --output code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_audio_reinject/reference_audit.json
```

## 设计参考

零初始化门控残差的参考是 [DiT作者实现中的门控残差与零初始化](https://github.com/facebookresearch/DiT/blob/main/models.py)；
本路线仅采用从零残差开始的原则，具体两次音频注入由上述公式定义，不引入 DiT 的 adaLN 或其他结构。
RNG隔离遵循 [PyTorch 2.11 fork_rng](https://docs.pytorch.org/docs/2.11/random.html#torch.random.fork_rng)。
投影零任务梯度时仍可能受解耦衰减影响，可参阅 [PyTorch 2.11 AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html)。
