# CED small / MeSH 5+10×2+5 / Mizar S1 GBS256

该目录完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh`。
只将 HTSAT/c2l/projection 换成现有 CED 基线使用的冻结 CED small 和官方
`freq_merge_126` mapper，并统一为126个单音频 token。原 MeSH 路线、标准 CED
路线和非 MeSH 5-10-5 路线均未修改。本地只有代码；真实权重、数据和训练运行在远程 Linux。

## 对比合同

直接对比入口为 `code/RSmol/run_mizar_s1_ced_small_gbs256_5090.sh`。
两条 CED 路线的音频 wrapper、离线配置 loader、mapper、Mellow 组合、数据读取、staging、
sampler、优化器代码一致。解码器的 prefix/forward 逻辑及完整 epoch/update/microbatch 循环
AST 一致。差异是文本模型架构及其初始化 checkpoint：标准 CED 基线使用30个独立物理层的
SmolLM2，本路线使用20个物理层、30次逻辑执行的 MeSH 模型。

MeSH 的原 answer-only CE 保留：answer attention mask 与非 PAD 条件共同选取有效目标；
实际 tokenizer 产生的 padding 均为 `!`，因此与标准 CED 的 ignore-PAD CE 使用同一批有效目标，
包含 EOS。可选 tensor 检查比较两种 loss 及梯度的数值一致性；不声称不同 LM 的训练轨迹逐位相同。

| 设置 | 两条 CED 路线共同值 |
|---|---|
| 数据 | hybrid 1,477,492行；保留 MusicCaps 和全部 SpeechCraft splits |
| 输入 | 全通道平均、32kHz、单录音先中心裁剪/右补零至20秒 |
| 音频特征 | 完整20秒重采样16kHz，逐样本 dB，冻结 BN 后按1012帧切分 |
| CED / mapper | CED冻结且始终eval；官方mapper全量训练，dropout=0.05 |
| 布局 | audio126 + SEP1 + prompt256 + answer300；prefix383 / total683 |
| 文本接口 | `attention_mask=None`、`labels=None`、`use_cache=False`；外部answer-only CE |
| Batch | 8 GPU × microbatch8 × accumulation4 = GBS256 |
| 时长 | 3 epochs；5771 updates/epoch；17313 total；866 warmup |
| AdamW | LM 3e-4 / mapper 6e-4；betas 0.9,0.999；eps1e-8 |
| Decay / LR | decay0.01，bias和ndim≤1不衰减；5% warmup+cosine，floor0.1 |
| GradNorm | initial L2=1、max=10、overdrive2.5、momentum0.995 |
| 数值 | FP32 forward，无autocast，GradScaler启用，TF32关闭 |
| 顺序 | seed1234，rank RNG=1234+rank；相同shuffle与rank连续分片 |
| 尾部 | 每epoch丢弃116行；DataLoader workers=0；后台复制workers默认4 |

两条路线都先构建 CED 与 mapper、再加载文本模型。在同一软件环境和 CED 资产下，相同
rank seed 与相同构造代码保持 mapper 的初始随机化顺序；DDP 广播rank0参数。
本路线在 `config_audit.json` 记录 `mapper_initial_sha256`，结束时记录mapper最终SHA。
不同文本模型的初始化及随机运算可以产生不同的后续 RNG 轨迹。

## 远程资产

CED默认目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small
```

需要 `config.json`、`configuration_ced.py`、`modeling_ced.py`、
`feature_extraction_ced.py`、`preprocessor_config.json` 和 `model.safetensors`。
不使用 `model.onnx`。严格检查384维/12层/6头/64 mel/16 stride/1012帧等几何及权重加载结果。
沿用标准 CED 已修复的离线 loader：直接复用config中的527项id2label，避免上游
CedConfig额外联网取标签CSV；不改远程资产或HF缓存。标签只用于配置初始化，训练不做音频分类监督。

MeSH文本初始化固定为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244
```

`models/recursive_text.py` 保持原路线文件不变，显式加载共享
`code/RSmol/recursive_model_5_10x2_5_mesh.py`；校验hidden576、logical30、physical20、
loops2、memory slots5、routers6和schedule `0..14,5..14,15..19`。
完整LM可训练，无LoRA。tokenizer从该checkpoint加载，按Mizar注册已有的 `!` 为PAD；
实际词表长度和token ID不得增长/改变，EOS/SEP=0。注册API返回1不代表词表新增1行。

数据路径与标准 CED 基线相同：

```text
manifest=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
resolved_paths=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
```

## 正式提交与流程

经GitHub同步后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_5090.sh \
  --ced-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

默认文本checkpoint已写入launcher；可用 `--text-model-dir` 显式指定兼容checkpoint。
资源为 `pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`，环境 `mellow_c8204d8`。
该入口仅提交formal，不先提交smoke任务。

1. 登录节点检查本地CED资产并准备/复用CPU metadata索引，此时尚未申请GPU。
2. GPU作业启动后台原始音频字节复制，训练直接从共享音频开始。
3. 写入runtime合同和资产SHA，检查MeSH checkpoint/tokenizer；加载冻结CED、mapper和MeSH LM。
4. 每来源1条样本经过真实loader/model/loss探测；探测后恢复RNG；DDP广播初始参数。
5. 每update累积4个microbatch的rank-local token mean，再执行GradNorm、AdamW、GradScaler和LambdaLR。
6. 全量复制完成后，在optimizer边界八卡MIN all-reduce一致确认后切到本地音频。
7. 完成17313次真实更新，核对CED参数/buffer不变，严格重载核验最终完整checkpoint后写训练PASS。

独立输出root下再生成 `formal/` 子目录：

```text
formal/runtime.json
formal/config_audit.json
formal/first_gradient_rank<N>.json
formal/trainable_params.txt
formal/metrics.jsonl
formal/staging_report.json
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/training_report.json
```

中间checkpoint按总步数六等分保存。checkpoint包含模型、optimizer、scheduler、GradScaler、
GradNorm、每rank RNG及数据游标；仅可用本路线相同代码/资产/数据/合同的完整checkpoint resume。
文本checkpoint只是初始化源。旧HTSAT或标准CED的音频checkpoint不能作为本路线resume。
resume时传 `--resume-checkpoint <完整ckpt路径>` 和原run的输出root。

## 验证范围

`reference_audit.json` 审核原MeSH数据/训练语义、标准CED对比合同、官方mapper类AST、
1344组文本/RNG案例、8组tokenizer调用、17314个LR位置及10000组GradNorm标量案例。
根目录新增的 `tests/test_mizar_s1_recursive_mesh_ced_small_*.py` 验证配置、资产拒绝、
launcher资源/参数、并发staging及清理、MeSH checkpoint和对比一致性。

可选tensor检查（在安装torch/torchaudio/transformers的环境中，不加载真实权重）：

```bash
cd mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small
python runtime_checks.py
```

本地无PyTorch/CUDA、远程数据或真实权重，因此本地静态检查不等于GPU训练通过。
现有HTSAT评测入口尚未接入该新CED路线，不能直接用它评估新checkpoint。
