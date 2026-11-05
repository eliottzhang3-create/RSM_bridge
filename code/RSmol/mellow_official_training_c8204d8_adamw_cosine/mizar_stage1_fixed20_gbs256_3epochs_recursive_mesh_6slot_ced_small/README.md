# CED-small / 六槽 MeSH 5+10×2+5 / Mizar S1 GBS256

完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small`。
模型变化仅为使用已训练的六槽 MeSH 文本模型初始化；运行资源改为
`pdgpu-3090 / 64 CPU / 256G / 8 GPU / 1 node`。原五槽训练、文本六槽实现及现有评测器保持原样。
本地维护代码，通过 GitHub 同步；权重、数据与真实训练均在远程 Linux。

## 文本初始化和结构

默认来源已固定在新的提交入口中：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh_6slot/formal_1epoch_fp32_20261031_124955519529617/checkpoint-009244
```

新路线显式加载 `code/RSmol/recursive_model_5_10x2_5_mesh_6slot.py`，
类为 `RecursiveLlama5_10x2_5Mesh6SlotForCausalLM`，架构合同为
`logical_30_physical_20_5_10x2_5_mesh_6slot_3router`。
FP32/offline 加载完整 LM 和全部已训练 routers，拒绝 missing/unexpected/mismatched/error 信息，
并检查实际物理层、独立 router 对象及参数形状；不重新初始化已加载的 routers。
tokenizer 同样从该 checkpoint 加载，按原 Mizar 方式将已有 `!` 注册为 PAD，
词表大小和 token ID 不增加、不改变，EOS/SEP=0。

- hidden576、20 个独立物理层、30 次逻辑执行，`0..14,5..14,15..19`。
- 中间十层共享权重并固定执行两次；完整 LM 与 routers 全量训练。
- 六个瞬时 memory slots，覆盖音频、SEP、prompt 和 answer 的完整序列。
  训练 memory 为 `[B,6,683,576]`，slot0 取原始多模态 input embeddings，其余五槽初始为零。
  每次 forward 重建，memory 不是持久参数或跨样本状态。
- 三组独立 write/read，共六个 `Linear(576,6,bias=True)`、十二份参数 tensors。
  routers 共20,772参数，相对五槽增加3,462个参数。
- Prelude query/value 均为 prefix output；两轮中段 query 取循环输入，write value 取循环输出。
  保持三次 write/read 和原读写公式。

仅加载文本模型和 tokenizer 作为一次新音频训练的初始化。
optimizer、scheduler、GradScaler、GradNorm、RNG、数据游标和音频训练步数从头开始。
CED 从原外部资产加载；mapper 按母本随机初始化。保留先构建 CED/mapper、再加载 LM 的顺序。

## 保持的训练配方

| 项目 | 配置 |
|---|---|
| 数据 | hybrid 1,477,492行，保留 MusicCaps 和全部 SpeechCraft splits |
| 音频输入 | 全通道均值，32kHz，中心裁剪/右补零至20秒 |
| CED | 冻结且始终eval；整段重采样16kHz、逐样本dB、冻结BN后按1012帧切分 |
| Mapper | 官方 `freq_merge_126`，全量训练，dropout0.05 |
| 布局 | audio126 + SEP1 + prompt256 + answer300；prefix383、total683 |
| 文本与loss | `attention_mask=None`、`use_cache=False`，原外部answer-only CE及PAD/EOS处理 |
| Batch | 8 GPU × microbatch8 × accumulation4 = GBS256 |
| 时长 | 3 epochs；5771 updates/epoch；17313 total；866 warmup |
| AdamW | LM/router最大LR3e-4、mapper6e-4；betas(0.9,0.999)、eps1e-8 |
| Decay / 调度 | decay0.01，bias与ndim≤1不衰减；原5% warmup/cosine、floor0.1 |
| GradNorm | initial L2=1、max=10、overdrive2.5、momentum0.995 |
| 数值 | FP32，无autocast，GradScaler启用，TF32关闭 |
| 数据顺序 | seed1234、rank RNG=1234+rank，原shuffle和连续rank分片，每epoch丢弃116行 |
| 读取 | 共享原始音频立即开始训练，后台4线程复制，完成后在optimizer边界八卡一致切换 |

音频资产仍为 `/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small`，
manifest仍为 `/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json`，
映射仍为 `outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv`。
保留 CED 离线标签配置修复、真实来源探针、首步梯度检查、CED冻结状态检查及最终完整重载审计。
首步梯度报告额外核对实际30次执行轨迹、20层、2轮、6槽和6个独立routers。

## GitHub 同步与正式提交

先将本地新增代码和文档提交并推送至 GitHub，然后在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_6slot_ced_small_3090.sh --stage-workers 4
```

沿用 `mellow_c8204d8` 和原容器
`docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`。
入口只提交一个正式训练作业，不自动提交 smoke 或 tensor 前置任务。
登录节点先做资产和CPU索引检查，真实CUDA训练由计算节点8卡wrapper执行。

默认输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_6slot_3090/formal_<timestamp>/
```

其下包括 `formal/runtime.json`、`formal/config_audit.json`、
`formal/first_gradient_rank<N>.json`、`formal/metrics.jsonl`、
`formal/staging_report.json`、`formal/training_report.json`，最终权重为：

```text
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
```

仍在2886、5771、8656、11542、14428和17313步保存完整checkpoint，并保存最终epoch文件。
新route、job、日志、index-cache、tmpfs和resume身份全部隔离。

## 完整恢复

只能恢复本六槽音频路线、相同代码/资产/数据/配方的完整正式checkpoint。
恢复使用原run输出root，并显式传入checkpoint：

```bash
bash run_mizar_s1_gbs256_recursive_mesh_6slot_ced_small_3090.sh \
  --output-dir /absolute/path/to/original/formal_<timestamp> \
  --resume-checkpoint /absolute/path/to/original/formal_<timestamp>/formal/checkpoints/model--step-5771.ckpt \
  --stage-workers 4
```

完整恢复覆盖LM、routers、mapper、optimizer、scheduler、GradScaler、GradNorm、
八个rank RNG和数据游标。上面的六槽文本checkpoint只作初始化，不能当音频resume；
原五槽音频checkpoint及其他路线checkpoint同样不能full resume。
每次作业重新staging；临时目录的路径不参与语义身份，但资产、数据及源码身份继续严格检查。

## 本地验证

```bash
python -B tests/validate_mizar_s1_mesh_6slot_ced_small.py
```

验证器覆盖原文件SHA保护、复制文件及完整train/contract AST对照、全部LR位置、
六槽严格加载拒绝、恢复身份、真实Bash假提交及64CPU/3090参数、staging和母本回归。
另提供真实CPU tensor检查：六槽memory重建、30次执行、mapper/LM/router梯度、
已训练router非零权重strict恢复和完整状态恢复后的下一步一致性。
依赖缺失会明确记录skip；加 `--require-tensors` 可禁止跳过。
报告为 `audit_reports/mizar_s1_mesh_6slot_ced_small_validation.json`。
这些验证不会下载资产、运行CUDA训练或提交远程作业，也不是formal自动前置任务。

本次本地56项实际检查通过，24项真实CPU tensor检查因缺依赖跳过，
Python/Bash语法通过；43个原文件SHA保护、23个复制文件一致性及受限AST对照通过。
首次沙箱运行遇Windows临时目录权限问题，沙箱外同命令复验通过。

本地结果与远程真实权重加载、八卡训练和恢复分开记录。
已有五槽MMAU/MMAR evaluator尚未适配本六槽合同，评测需要单独扩展。
