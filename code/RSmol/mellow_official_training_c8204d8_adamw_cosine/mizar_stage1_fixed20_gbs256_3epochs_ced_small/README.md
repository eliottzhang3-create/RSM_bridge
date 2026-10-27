# Mizar S1 / CED small / fixed20 / GBS256 / 3 epochs

## 2026-10-26 离线配置错误修复

远程 `LocalEntryNotFoundError` / `OSError` 出现在权重加载前的 CED 配置构造阶段。上游 [configuration_ced.py](https://huggingface.co/mispeech/ced-small/blob/main/configuration_ced.py) 在 outputdim=527 时无条件读取另一个仓库的 `class_labels_indices.csv`；`local_files_only=True` 并不能为这张缺失的外部标签表提供内容。官方 [config.json](https://huggingface.co/mispeech/ced-small/blob/main/config.json) 本身已包含完整标签映射。

`models/ced_offline.py` 直接复用本地 JSON 的真实标签生成临时 CSV，在原版 CED 配置和模型加载期间，仅调整配置模块内导入的 `cached_file` 绑定，提供这张表。使用原版配置类并显式传入 AutoModel，保留所有网络结构、527维原始分类头和严格权重检查。结束或异常均删除临时文件并恢复函数绑定；不改写远程资产、HF缓存源码或全局Hub函数。

预检查现在要求 `id2label` 完整覆盖0..526，并校验已提供的 `label2id`。真实标签允许重复显示名称，恢复完整 JSON 索引；官方发布配置有527项 id2label、524项 label2id。不会为缺失元数据编造标签，也不允许退回网络下载。远程应保持原来的 CED 六文件目录，无需额外下载 CSV。

本次失败尚未进入 optimizer update。同步修复后，用下面命令生成新的输出目录重新提交，不带 resume；旧输出目录和旧缓存可以保留。新代码SHA会选择新CPU索引缓存。新增6项离线回归及既有17项检查通过；另使用真实上游 CedConfig 类和官方配置执行 stdlib 回归，确认原错误复现且修复后额外 Hub 标签请求为0。该回归使用 PretrainedConfig stand-in，不涉及真实权重/tensor；GPU重跑仍待远程确认。

本路线完整复制隔离自相邻的 `mizar_stage1_fixed20_gbs256_3epochs`，将音频侧替换为 Mizar 官方 S1 的 **CED small + freq_merge_126 mapper**。原路线保持不变。本地只维护代码，经 GitHub 同步至远程 Linux；权重、数据和训练产物均留在远程。正式提交入口为上两级的 `run_mizar_s1_ced_small_gbs256_5090.sh`。

## 2026-10-27 MMAU / MMAR 对齐评测

正式 checkpoint 生成后，使用上级 `code/RSmol` 目录中的两个独立入口：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_ced_small_mmau_eval_5090.sh
bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

默认入口绑定 `formal_20261026_193830617147305/formal`、`model--step-17313.ckpt`、`runtime.json` 和本隔离训练路线。远程路径或输出路径变化时，可通过 `RSMOL_MIZAR_CED_RUN_ROOT`、`RSMOL_MIZAR_CED_CHECKPOINT`、`RSMOL_MIZAR_CED_RUNTIME_CONFIG`、`RSMOL_MIZAR_CED_ROUTE_ROOT`、`RSMOL_MIZAR_CED_MMAU_OUTPUT_DIR` 或 `RSMOL_MIZAR_CED_MMAR_OUTPUT_DIR` 覆盖。

`RSMOL_MIZAR_CED_RUN_ROOT` 指向包含 `formal/` 与 `eval/` 的外层 run 目录。两个提交入口与原评测使用相同容器、`mellow_c8204d8` 环境和 `pdgpu-5090 / 8 CPU / 32G / 1 GPU / 1 node` 资源。MMAU 与 MMAR 各完整生成1000条，不附加smoke前置任务；本地仅验证代码与控制流，真实权重加载和官方分数需在远程运行确认。

评测 runner 复用原 MMAU/MMAR 官方 metadata、parquet/audio 读取、固定 choice prompt、官方输出 key 识别、断点 raw JSONL store 和官方 `evaluation.py` scorer。CED 评测 adapter 严格验证 checkpoint 的 route、step、GBS256 geometry、训练 source SHA、CED 六个资产文件及 mapper contract，然后在线 FP32 计算固定20秒音频的 CED hidden 和 `freq_merge_126`。推理协议为 `126+1+256=383` prefix、`attention_mask=None`、全量重算 greedy argmax、无 cache、最多300 new tokens；EOS 保留在原始输出中，不做 trim、split、选项抽取或二次 rescoring。结果目录包含 `raw_generations.jsonl`、官方顺序 prediction JSON、`evaluation_report.json` 和 `official_evaluation.txt`。

## 固定训练实现

| 项目 | 本路线 |
| --- | --- |
| 音频 encoder | CED small，384 dim / 12 blocks / 6 heads；严格加载本地 safetensors |
| 冻结策略 | CED 所有参数冻结、所有子模块始终 eval；BatchNorm buffer 不更新 |
| Mapper | 官方 `CEDFrequencyMerger` + `CEDTemporalMapper`，freq_merge_126，dropout 0.05，不插值 |
| 额外投影 | Identity；mapper 已输出 576 dim |
| 文本模型 | 原始 SmolLM2-135M，30 个独立层，全量训练 |
| 序列 | audio126 + SEP1 + prompt256 + answer300 = 683；prefix383 |
| 损失 | 外部 answer-only next-token CE；logits 的 382..681 预测 300 个 answer token |
| Padding / EOS | Mizar 的 `!` 作为 PAD（现有词表 ID17），SEP/EOS 均 ID0；不扩词表 |
| Attention | `attention_mask=None`；padding 保持可见，answer PAD 仅在 loss 中忽略 |
| 优化 | AdamW：mapper 6e-4，LM 3e-4；betas=(0.9,0.999)，eps=1e-8，WD=0.01，bias/ndim≤1 不衰减 |
| LR / clip | 原 Mizar 5% warmup cosine，最低 factor0.1；原 GradNormTracker |
| 精度 | 全部 FP32，无 autocast，无 TF32；保留母本 GradScaler |
| 批量 | 8 GPU × microbatch8 × accumulation4 = GBS256 |
| 更新 | 3 epochs；5,771 updates/epoch；17,313 total；866 warmup；每 epoch 丢弃尾部116行 |

训练沿用原始数据顺序、rank seed、固定文本长度、EOS 先追加再截断、rank-local valid-token mean，以及四次 microbatch loss 均值累积、前三次 `no_sync()`、最后一次 DDP 同步。没有加入递归、MeSH、LoRA 或 SFT 逻辑。

## 音频计算顺序

1. 沿用母本全部音频读取策略：所有声道求均值，重采样到32kHz；单录音居中裁剪20秒或右补零；明确的双录音各取10秒，再按原顺序组合。
2. dataloader 的两个10秒张量先拼回完整 `[B,640000]`，然后整段重采样到16kHz。不会独立计算两个10秒的 mel。
3. 整段 mel 使用 n_fft=512、win=512、hop=160、n_mels=64、center=True，得到 `[B,64,2001]`。power dB / top_db120 对每条录音独立计算，防止不同响度样本共享截断阈值。
4. 冻结的 `init_bn` 先作用于整个 mel，再按1012帧切分；第二段989帧，在 BN 后补23帧零。
5. CED `forward_features` 得到每段 `[252,384]`，合成 `[B,504,384]`。保留 `[segment,frequency,time,hidden]` 顺序后变换为 `[B,126,4,384]`；不能把两个 segment 拼接后直接按4个频率 reshape。
6. mapper 输出 `[B,126,576]`。`no_grad` 仅包住 CED 特征提取，mapper 在外面正常反传。

Mizar 官方 S1 读取由单样本提取并保存的 FP16 CED hidden cache；本路线按既定在线训练要求直接计算 FP32 hidden，不引入缓存。独立样本的 dB 截断与官方 B=1 cache builder 一致，但不能把在线 FP32 宣称为与官方 FP16 cache 数值逐位相同。

## 远程资产与数据

默认 CED 目录：`/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small`。必须包含非空的 `config.json`、`configuration_ced.py`、`modeling_ced.py`、`feature_extraction_ced.py`、`preprocessor_config.json`、`model.safetensors`。训练加载本地 PyTorch safetensors 和目录内的 CED 模型代码；`model.onnx` 不参与训练。开启 `local_files_only=True`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，不下载权重。加载出现 missing / unexpected / mismatched keys 或 error_msgs 时直接失败，不允许随机补齐 CED。

文本目录仍为 `/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2`。环境仍为 `mellow_c8204d8`；新路线直接依赖 torch、torchaudio、transformers、safetensors、numpy、soundfile 及本地 CED 模型代码所需依赖。

数据仍为1,477,492行 hybrid manifest：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
```

来源计数：reasonaqa_oneaudio756968、AudioCaps113844、Clotho4798、MusicCaps3221、CompA-R249161、LP-MusicCaps-MTT19632、SpeechCraft287727、TACOS42141。保留已确认的 MusicCaps 和 SpeechCraft 全部 split；clean100/360 使用 LibriTTS，other500 使用 LibriTTS-R。

数据准备文件与母本逐字节相同：登录节点准备带内容 SHA 的 CPU 索引；训练从共享原始音频立即开始，4 workers 在后台复制原始字节，复制全部完成后在 optimizer boundary 通过 collective MIN 切换到 `/dev/shm`。不预解码、不预处理音频、不建立 CED hidden cache。独立 stage 根目录使用 `/dev/shm/mizar_s1_gbs256_ced_small_*`；正常结束、异常退出均先终止复制进程再按 ownership 清理。

## 正式提交

先通过 GitHub 将本地代码同步到服务器。在远程登录节点执行，实际 GPU 训练由 vc 作业运行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_ced_small_gbs256_5090.sh \
  --ced-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

提交资源与母本一致：`pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`，相同容器。提交前只做文件/config 检查及 CPU index；没有 smoke 或额外短训练前置任务。默认输出位于独立的 `mizar_s1_ced_small_gbs256_5090/formal_<时间戳>/formal/`，终端会打印输出根目录。

## Checkpoint 与恢复

完整 checkpoint 保存模型（含冻结 CED）、AdamW、scheduler、GradScaler、GradNormTracker、8 rank RNG 和数据游标。保留母本六个保存点（2886、5771、8656、11542、14428、17313），最终文件为 `formal/checkpoints/model--epo-3.ckpt`。

只接受本 CED 路线且代码、数据、包版本、权重资产、训练合同一致的正式 checkpoint。HTSAT/Mellow 或其他路线 checkpoint 不能用于 resume。恢复命令中的输出根目录可以使用原任务根目录，继续追加 metrics：

```bash
bash run_mizar_s1_ced_small_gbs256_5090.sh \
  --output-dir /完整路径/原CED运行根目录 \
  --resume-checkpoint /完整路径/原CED运行根目录/formal/checkpoints/model--step-5771.ckpt \
  --stage-workers 4
```

产物包括 `runtime.json`、`config_audit.json`、`trainable_params.txt`、`metrics.jsonl`、每 rank 首次梯度审计、checkpoints、最终 `training_report.json` 或 `failure_rank<N>.json`。训练会验证 CED 参数和 buffer 起止 SHA 一致、mapper/LM 梯度存在且有限，以及最终 checkpoint 全状态回读。原 HTSAT 评测 adapter 没有改成 CED；本次交付范围为正式训练路线。

## 已完成与待远程验证

本地通过23项 stdlib/CPU测试，包括 CED 离线标签回归、真实 Bash 提交入口的 mock、资源参数与含空格路径、资产失败拒绝、后台复制/训练并行和退出清理。Python 编译与 Bash 语法检查通过。`reference_audit.json` 记录本地官方源码对齐：两个 mapper 类 AST 完全一致；母本数据文件与关键训练函数不变；1,344组文本/RNG、8组 tokenizer 参数、17,314个 LR 位置和10,000组 GradNorm 标量比较通过。

本机没有 PyTorch/CUDA、远程权重和数据，因此没有执行真实 CED tensor 推理、梯度测试或 GPU 训练。`runtime_checks.py` 提供可选的远程 CPU tensor 测试（toy encoder，不读取真实权重），覆盖时间/频率顺序、逐样本 dB、完整20秒拼接、冻结与 mapper 梯度、严格离线加载参数、CE 对齐和 optimizer 恢复。它不自动运行，也不是提交前置条件。正式作业的资产加载、真实数据 source probes、首次梯度及完整训练报告才验证远程实际执行。

从仓库根目录可分别运行本地检查；每个脚本使用独立进程，避免不同隔离路线的同名 Python 模块互相覆盖：

```bash
python tests/test_mizar_s1_ced_small_route.py -v
python tests/test_mizar_s1_ced_small_shared_start.py -v
python tests/test_mizar_s1_ced_offline_config.py -v
```

可选 tensor 检查：在远程对应环境中进入本路线目录后运行 `python runtime_checks.py`。它无需 CUDA；不能将该结果视为真实权重加载或正式训练 PASS。

官方参考路径为本地 `C:\Xlance\assets\Mizar\Mizar_159M-main\Mizar_159M-main`；模型代码和许可证来源见 `NOTICE`。
