# USAD2-Small / MeSH 5+10×2+5 / Mizar S1 GBS256

本路线完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh`。音频和 mapper 与普通 USAD2 路线 `mizar_stage1_fixed20_gbs256_3epochs_usad2_small` 完全一致，文本端使用既有两次循环 MeSH。原训练目录、共享 MeSH 实现均未修改。代码在本地维护，经 GitHub 同步至远程 Linux；权重和数据不放入仓库。

## 对齐的训练设置

音频、mapper、数据和 staging 等 16 个文件与普通 USAD2 路线逐字节相同；decoder 的 forward/prefix 和完整 epoch/update/microbatch 循环通过 AST 对照。比较双方采用不同的文本结构和初始化：普通路线使用 30 层 SmolLM2，本路线使用 checkpoint-009244 的 MeSH，因此结果不能仅归因于循环结构。

| 项目 | 固定设置 |
|---|---|
| 数据 | 原 hybrid manifest，1,477,492 行，八个来源及 resolved paths 不变 |
| Batch | 8 GPU × microbatch 8 × accumulation 4 = GBS 256 |
| 更新 | 3 epochs；5771 updates/epoch；17313 total；866 warmup；每 epoch 丢弃 116 行 |
| AdamW | mapper LR 6e-4；全量 LM（含 routers）LR 3e-4；betas (0.9,0.999)；eps 1e-8 |
| Decay | 0.01；bias 和 ndim≤1 的参数不 decay |
| LR / clipping | 原 Mizar warmup-cosine LambdaLR，floor 0.1；原 GradNormTracker |
| 精度 | FP32；autocast、TF32 关闭；GradScaler 开启 |
| 文本 | prompt 256、answer 300；既有 PAD `!` / ID 17；SEP/EOS ID 0 |
| LM 调用 | attention_mask=None、labels=None、use_cache=False；显式 answer-only CE |
| 随机性 | seed 1234；rank RNG 1234+rank；原 shuffle 和 rank 切分 |
| 音频读取 | 共享目录立即训练；后台 4 workers 复制；完整复制后在 optimizer 边界统一切换 |

## 音频和 mapper

原始音频解码为 float32，多声道取均值，必要时先重采样至 32 kHz。单音频中心裁剪/右侧补零到 20 秒；显式双原音频按原策略各取 10 秒后拼接。dataset 只返回一条 `[B,640000]` 波形，整条重采样到 16 kHz，再一次输入始终冻结且 eval 的 USAD2-Small，不切成两条分别编码。

用户已实测 20 秒输出 `x=[B,999,384]`、`x_lengths=[999]`、`mel=[B,1998,128]`。本地包显式导入 USAD2 并严格加载 safetensors，避开 HF 动态模块缓存缺少传递依赖的问题。

mapper 先在原始 999 帧之前加一枚有效帧均值，随后使用原 Mellow Projection：无 bias Linear 384→576、GELU、无 bias Linear 576→576、dropout 0.5、残差及 LayerNorm。保留投影后的全局 token，仅对时间帧做相邻 4 帧非重叠均值；末组 3 帧除以 3。最终为 `1+250=251` 个音频 token，无卷积、可训练 CLS 或第二次降采样。

```text
audio 251 + SEP 1 + prompt 256 + answer 300 = 808
prefix 508；首个 answer 的预测 logit 位于 507
```

USAD2 和 mapper 的构造顺序与普通路线相同；额外 tokenizer 检查会保存/恢复 RNG。报告记录 mapper 初始化、训练起点及最终 SHA，可比较两个实际作业的初始化是否相同。

## MeSH 初始化与环境

以下目录只用于文本模型权重初始化，不恢复其预训练 optimizer/scheduler/scaler/GradNorm/RNG；S1 从第 0 步重新开始：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244
```

继续使用 `code/RSmol/recursive_model_5_10x2_5_mesh.py`：hidden 576、20 个独立物理层、30 个逻辑层、loops 2、5 个临时 memory slots、6 个独立 routers（3 write / 3 read）。执行顺序为 `0..14,5..14,15..19`；memory 每次 forward 重置，不增加序列 token，不存入 checkpoint。共享实现 SHA 和真实加载后的结构均被审计；缺失、多余或不匹配的 checkpoint 权重会报错。全量 MeSH 与 mapper 训练，USAD2 冻结，不使用 LoRA。

保留原 MeSH 的 `answer attention_mask AND 非 PAD` CE。启动时比较 MeSH 与普通 SmolLM2 tokenizer 的完整 token-to-ID 字典、padding/truncation 方向和 16 组真实编码；检查 mask 排除的目标均为 PAD。`runtime_checks.py` 另对相同 logits 验证此 CE 与普通路线 CE 的 loss 和梯度一致；EOS 参与训练。

```text
环境：mellow_c8204d8
USAD2：/hpc_stor03/sjtu_home/jinwei.zhang/models/USAD2
普通基线 tokenizer：/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2
manifest：/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
resolved_paths：/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
```

沿用已验证版本：Torch/torchaudio 2.11.0+cu128、Transformers 4.54.1、safetensors 0.8.0、NumPy 2.2.6、soundfile 0.14.0，无需升级依赖。USAD2 目录须包含 `__init__.py`、`config.json`、`configuration_usad2.py`、`modeling_usad2.py`、`usad_model.py`、`usad_modules.py` 和 `model.safetensors`；`__init__.py` 允许为空。复制保留的 `models/htsat.py`、`models/config.py` 不在实际训练导入路径，无 torchlibrosa 依赖。

## 验证范围

新增 32 项 stdlib 回归检查覆盖几何、身份隔离、配置、tokenizer、严格加载和 Bash launcher 的成功/失败/恢复/引号传参。`reference_audit.json` 记录与原 MeSH、普通 USAD2、Mizar 官方代码的对照：16 个 USAD2 公共文件逐字节一致，1344 组文本/RNG、8 组 tokenizer 参数、17314 个 LR 位置、10000 组 GradNorm 标量比较。

本地没有 PyTorch，也没有远程权重，不能据此宣称真实模型/GPU 训练通过。另提供 13 项 CPU tensor 检查，使用假 encoder/小型随机 LM，不加载大权重；覆盖真实 MeSH 共享实现的两次循环、5 槽 memory 重置、routers 梯度、mapper、CE、优化器恢复等。可在远程环境单独运行，不是正式提交前置门禁：

```bash
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate mellow_c8204d8
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_usad2_small
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python runtime_checks.py
```

正式作业内会执行真实权重加载、各来源单样本 forward/CE、MeSH 调度检查，以及首个 optimizer update 的 mapper/全 LM/六个 routers 梯度审计。完成时检查 USAD2 权重未变、训练参数有更新，并验证最终 checkpoint。microbatch 8 的显存可行性仍须实际 GPU 作业确认；实现未擅自改变 batch 或启用 gradient checkpointing。

## 正式提交

GitHub 同步后在远程登录节点执行。先激活 `mellow_c8204d8`；提交脚本只做轻量资产检查及 CPU index 准备，然后提交正式作业：

```bash
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate mellow_c8204d8
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_usad2_small_5090.sh \
  --text-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244 \
  --comparison-text-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2 \
  --usad2-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/USAD2 \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_usad2_small_gbs256_recursive_mesh_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

资源沿用 pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node；仅运行 formal，无 smoke 作业。实际配置中固化权重、包版本、代码和数据身份。输出、日志、tmpfs、index cache 和 resume 标识与原路线隔离。comparison 目录只用于基线 tokenizer 一致性检查，不加载第二份 LM 权重；相关资产 SHA 写入契约。

保存步数与普通 USAD2 一致：2886、5771、8656、11542、14428、17313。最终 checkpoint 位于输出根目录下 `formal/checkpoints/model--epo-3.ckpt`；训练完成的判据为 `formal/training_report.json` 中 `status=PASS` 且 `optimizer_steps=17313`。

保留 `formal/runtime.json`、`config_audit.json`、`first_gradient_rank<N>.json`、`metrics.jsonl`、`training_report.json` 便于对照。checkpoint 为该路线的完整状态，推理/评测时须采用 USAD2 音频路径和同一 mapper，不能直接交给只支持 HTSAT 的旧入口。

## 断点恢复

仅支持同一路线、代码、数据、资产及训练契约的完整 checkpoint；恢复 optimizer、scheduler、scaler、GradNorm、各 rank RNG 和 sampler cursor。指定原有 run root，例如：

```bash
RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_usad2_small_gbs256_recursive_mesh_5090/formal_REPLACE_WITH_EXISTING_TIMESTAMP
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_usad2_small_5090.sh \
  --stage-workers 4 --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-REPLACE_WITH_SAVED_STEP.ckpt"
```
