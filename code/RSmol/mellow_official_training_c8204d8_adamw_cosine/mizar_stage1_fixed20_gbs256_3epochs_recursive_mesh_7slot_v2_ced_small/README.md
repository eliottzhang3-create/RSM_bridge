# 七槽 v2 MeSH + CED-small：Mizar S1 GBS256正式音频训练

完整复制隔离四槽CED-small音频路线，只改为本次七槽v2文本模型及独立恢复身份，
并增加最终文本checkpoint交接审计。新模型覆盖完整多模态序列；历史audio7slot的
音频七槽/文本五槽及CED4/8特征槽实验保持原样。
本地代码经GitHub同步，模型/数据/GPU执行留在远程Linux。

## 初始化与结构

先完成[七槽v2转换和完整一轮BF16文本训练](../../text_5_10x2_5_mesh_7slot_v2/README.md)。
入口必须显式传 `--text-model-dir /实际文本run/checkpoint-009244`。
CPU预检要求本v2架构、9244步、完整一轮、BF16 precision policy、完整manifest/marker和全部文件；
计算节点再次检查，将完整LM/config/tokenizer SHA与provenance写入runtime身份。
strict/offline FP32加载完整LM及已训练routers，不重新初始化router。
只继承模型/tokenizer；不恢复文本optimizer、scheduler、RNG或游标。
CED与mapper按原构建顺序初始化，CED冻结/eval，mapper随机初始化并训练。

hidden576、20物理层/30次执行、5+10×2+5，轨迹0..14,5..14,15..19。
七槽memory覆盖完整音频/SEP/prompt/answer，训练为[B,7,683,576]，每forward重建，
slot0为原始多模态embeddings，其他六槽为零。三组write/read、六个独立Linear(576,7)，
12份参数共24,234；prefix output-query/value与loop input-query/output-value、原读写公式保持。
架构 `logical_30_physical_20_5_10x2_5_mesh_7slot_v2_3router`；
音频route `ced_small_mizar_s1_fixed20_gbs256_recursive_mesh_7slot_v2`。

## 保留的音频配方

| 项目 | 配置 |
|---|---|
| 资源 | pdgpu-3090 /64CPU /256G /8GPU /1node |
| 环境 | mellow_c8204d8，原容器sjtu_wumengyue-mhl:0.0.1 |
| 数据 | 原hybrid 1,477,492行，MusicCaps与全部SpeechCraft splits保留 |
| 音频 | 全通道均值，32kHz，中心裁剪/右补零至20秒 |
| CED | 冻结/eval，16kHz、逐样本dB、冻结BN后按1012帧分段 |
| Mapper | 官方freq_merge_126，dropout0.05，全量训练 |
| 布局 | audio126+SEP1+prompt256+answer300，总长683 |
| 数值 | FP32前向、无autocast、原GradScaler、TF32关闭 |
| Batch | 8×microbatch8×GA4=GBS256 |
| 时长 | 3epochs，每轮5771步，共17313步，warmup866 |
| AdamW / LR | LM/router最大3e-4，mapper6e-4；betas(0.9,0.999)，eps1e-8 |
| Decay / scheduler | decay0.01，bias/低维不衰减；原5% warmup/cosine/floor0.1 |
| 裁剪 | 原GradNorm tracker：initial1、max10、overdrive2.5、momentum0.995 |
| 顺序 | seed1234，各rank RNG1234+rank；原shuffle/连续rank分片，每轮丢弃116行 |
| Loss | 每rank每microbatch有效answer-token均值→四次GA均值→DDP均值 |
| 保存 | 2886/5771/8656/11542/14428/17313步及最终epoch3完整checkpoint |

文本训练BF16仅覆盖其前向，不改变FP32保存权重，也不改变本音频FP32配方。
保持attention_mask=None、use_cache=False、原PAD/EOS及外部answer-only CE。
CED资产仍为 `/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small`，
manifest仍为 `/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json`，
映射仍为 `outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv`。

## 提交与输出

本地提交推送后，远程登录节点：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
# 替换为本次七槽v2完整文本训练的实际run目录。
TEXT_RUN=/absolute/path/to/stage4_5_10x2_5_mesh_7slot_v2/formal_1epoch_bf16_TIMESTAMP
bash run_mizar_s1_gbs256_recursive_mesh_7slot_v2_ced_small_3090.sh \
  --text-model-dir "$TEXT_RUN/checkpoint-009244" \
  --stage-workers 4
```

登录节点仅做CPU资产/metadata/索引准备，真实CUDA训练由vc提交的计算节点执行。
只提交一个formal，无自动smoke或tensor前置作业。
默认输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_7slot_v2_3090/formal_<timestamp>/
```

其下包含formal/runtime.json、config_audit.json、first_gradient_rank<N>.json、metrics.jsonl、
staging_report.json、training_report.json，最终为：

```text
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
```

## 共享路径启动与恢复

沿母本initial_steps=0。计算节点仅等待metadata/owner就绪，立即共享原始路径训练；
单producer默认4线程后台搬运raw bytes，不预计算CED特征。完整文件/字节数检查通过后，
八rank在optimizer边界、读取四个microbatch之前通过all_reduce(MIN)一致切换本地。
producer失败/死亡/心跳异常中止；先停止/reap producer，再清理当前token拥有的独立tmpfs。
每次resume重新staging并从shared开始；route/job/log/index-cache/tmpfs/输出全部隔离。

完整恢复仅接受本七槽v2音频路线同代码/依赖/资产/数据/配方的正式checkpoint。
使用原音频output root、原文本checkpoint绝对路径，并显式指定音频resume checkpoint：

```bash
bash run_mizar_s1_gbs256_recursive_mesh_7slot_v2_ced_small_3090.sh \
  --text-model-dir "$TEXT_RUN/checkpoint-009244" \
  --output-dir /absolute/path/to/original/audio/formal_TIMESTAMP   --resume-checkpoint /absolute/path/to/original/audio/formal_TIMESTAMP/formal/checkpoints/model--step-5771.ckpt   --stage-workers 4
```

恢复LM/router/mapper、optimizer、scheduler、GradScaler、GradNorm、八rank RNG和数据游标。
文本checkpoint只作初始化；四/六槽和历史七槽音频checkpoint均不接受为本路线full resume。

## 验证

仓库根目录 `python -B tests/validate_mesh_7slot_v2.py`：116项实际本地检查通过，
52项真实tensor检查因依赖缺失跳过。新路线26项tensor与四槽回归26项分别标记；
新Python/Bash语法、Shell LF、206原文件SHA、22个复制文件一致及完整更新循环/受限AST对照通过。
覆盖实际文本checkpoint writer→音频preflight/config、严格加载拒绝、LR全部位置、真实Bash假提交、
原四槽与历史七槽回归、共享路径启动/失败清理/八卡切换协议。
报告 `audit_reports/mesh_7slot_v2_validation.json`；--require-tensors禁止任何tensor skip。
Windows沙箱临时目录权限失败后，在沙箱外同流程复验通过。
本地缺torch/torchaudio/transformers/soundfile/safetensors，尚未生产转换、读取远程权重或真实GPU训练/恢复。
本轮未commit/push或提交真实作业；现有MMAU/MMAR evaluator需另行适配。
