# Mizar 30→28 层组合搜索

这是独立的推理实验入口。新增包复用原项目的模型构造、release 校验和部分数据读取工具，不修改旧训练或评测源码。源 checkpoint 保持原始 30 层，运行时按原层号跳过两个完整 block；不训练、不改写权重、不生成 MeSH 模型。

本地 Windows 维护这些代码，通过 GitHub 同步到远程 Linux；真实 manifest、音频缓存和评测输出在远程生成。当前交付不包含远程作业的实测分数。

## 直接提交一个完整作业

在远程仓库同步代码后执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

# 可先检查实际路径、资源和协议；不读取模型/数据，不提交。
bash run_mizar_layer_pair_search_8gpu_3090.sh --check-args

# 一次 vc submit：pdgpu-3090 / 8 GPU / 1 node / 64 CPU / 256G。
bash run_mizar_layer_pair_search_8gpu_3090.sh
```

作业内依次完成：CPU 生成或核验 manifest → 8 个 GPU worker 加载完整 Mizar → rank0 预计算 100 个 prefix → 真实 BF16/掩码检查 → 378 组同步评测 → 30 层基线 → 校验并输出排名。没有额外提交检查作业，也没有按 mask 分别提交任务。

入口沿用项目 `vc` wrapper、镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1` 及 `mellow_c8204d8` conda 环境。默认 conda 根为 `/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3`，可通过 `RSMOL_CONDA_BASE` 指定其他根。权重及 tokenizer 仅从本地资产离线加载。

## pdgpu-4090 单卡入口

同步本地新增代码到远程后执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_layer_pair_search_1gpu_4090.sh
```

资源固定为 `pdgpu-4090 / 1 GPU / 1 node / 8 CPU / 32G RAM`，同样只提交一个作业。可追加 `--check-args` 预览资源和参数。单卡由一个 worker 顺序完成每组全部 100 题，完整运行 378 组以及末尾 30 层基线；BF16、正文 likelihood 求和、首组 100 条自由生成、缓存和恢复规则与八卡一致。

单卡复用同一份冻结 manifest，不重新抽样；默认独立输出目录为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_30to28_pair_search_bf16_seed1234_1gpu4090_v1
```

需要指定已有 manifest 时：

```bash
bash run_mizar_layer_pair_search_1gpu_4090.sh \
  --manifest /absolute/path/calibration100/manifest.json
```

在已分配单卡的计算作业内，已有 manifest 的直接评测入口为 `bash mizar_layer_pair_search/run.sh evaluate_single_gpu ...`。单卡产物仅有 `ranks/rank00/`，完成校验要求该 worker 每组覆盖全部 100 题。单卡与八卡的运行身份和缓存分别管理，请保留各自的独立输出目录。源码发生变化后，旧评测输出仍按原有规则拒绝跨版本续跑；冻结 manifest 的生成源码在本次单卡扩展中未变。

## 单独生成冻结 manifest

这是 CPU 数据准备入口，需要环境里的 Torch、torchaudio、Transformers、PyArrow、SoundFile；不加载模型、不初始化 CUDA。

文件和 parquet 内嵌音频字节均使用 `soundfile.read()` 解码为 FP32；torchaudio 仅用于重采样，不调用 `torchaudio.load()`，因此不需要 TorchCodec。解码后仍按“重采样 32kHz → mono → 前 20 秒 → 右补零”的顺序处理。

如果此前在 `build()` 的音频解码阶段遇到缺少 TorchCodec 的错误，最终 manifest 尚未写入；同步本次修复后直接重跑原生成命令即可。已成功冻结的旧版本 manifest 绑定旧解码源码，须使用新 manifest 路径重新生成，不能跳过身份校验混用缓存。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash mizar_layer_pair_search/run.sh manifest
```

默认产物：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/
  mizar_prune_calibration100_seed1234_v1/manifest.json
```

抽样 seed 固定为 1234，在每个数据集的合格候选池中独立无放回抽 25 题，再按来源顺序保存。总计 100 个唯一 key，固定全局索引 0…99。缺音频、解码错误不会通过补抽隐藏。manifest 记录原始字段、实际题面、候选正文、GT 映射、token IDs、截断信息、音频与波形指纹、源文件身份、选择统计和内容哈希。

| 数据集 | 默认来源与处理 |
| --- | --- |
| MMAU | `data/MMAU_test_mini/mmau-test-mini.json`；优先 `test-mini-audios/<id>.wav`，缺失时从 `test_mini.parquet` 按 ID 定位音频 |
| MMSU | `data/MMSU/MMSU data/data/train-0000{0,1,2}-of-00003.parquet`，核验 5000 行；忽略 null/空选项，保存原 slot 映射 |
| MMAR | `data/MMAR/MMAR-meta.json`；音频根 `data/MMAR/mmar-audio`，接原 `audio_path` |
| ReasonAQA | `data/reasonaqa/test.json`；仅 `AudioCaps-MCQ.json`、`Clotho-MCQ.json`，排除两个不同录音；选项从题面解析，caption 不入模 |

上述 `data/` 的完整根为 `/hpc_stor03/sjtu_home/jinwei.zhang/data/`。ReasonAQA 默认音频根为其下 `audiocaps_v2` 和 `clotho_v2_1`。

可用 `--mmau-metadata`、`--mmau-parquet`、`--mmau-audio-root`、`--mmsu-dir`、`--mmar-metadata`、`--mmar-audio-root`、`--reasonaqa-test`、`--audiocaps-root`、`--clotho-root` 覆盖来源；含空格的路径必须引用。例如：

```bash
bash mizar_layer_pair_search/run.sh manifest \
  --manifest /absolute/path/calibration100/manifest.json \
  --mmsu-dir '/hpc_stor03/sjtu_home/jinwei.zhang/data/MMSU/MMSU data'
```

已有 manifest 只核验、复用，不重新抽样。`manifest` 和 `pipeline` 模式要求复用时的来源参数与首次生成相同，防止忽略新参数。要更换数据来源或解析实现，使用新 manifest 路径。manifest 必须在评测输出目录之外。

## 评测实现与评分

- 保护原始 0-based 层号 0、29，从 1…28 中选两层跳过，共 378 组；最终执行 28 层。首组为 `[1,2]`，即 1-based 第 2、3 层。`drop_mask` 为 30 位，1 表示跳过，0 表示执行；基线全零。
- 单节点 `torchrun` 启动 8 个进程，各绑定一张可见 GPU、加载同源 Mizar。固定按 `global_index % 8` 分题，rank0…3 各 13 题，其余各 12 题。所有卡每轮运行同一个 mask。
- rank0 合并精确 100 条后保存结果、打印 `MASK_DONE` 并 flush；全体 barrier 后才切换下一 mask。accuracy 为正确数除以 100，四个子集各以 25 为分母，不平均各 rank 的 accuracy。
- 模型参数从官方 final checkpoint 严格以 FP32 加载；神经网络前向采用 CUDA BF16 autocast，输入 prefix 明确为 BF16。音频前处理与 `log_softmax`、分数求和为 FP32。关闭 TF32、flash/memory-efficient SDPA；不回退 FP16。
- 每题 encoder＋训练后的 mapper/projector 只预计算一次，缓存完整 `[1,383,576]` prefix。缓存原子落盘并绑定模型、manifest、代码和精度身份。恢复时复用已完成缓存，不重复编码。
- 每个候选的分数为 `sum_t log P(body[t] | audio, prompt, body[:t])`。正文统一 lowercase，去掉选项标签，不加 BOS、EOS 或人为前导空格；所有正文 token 参与求和。首 token 使用 prefix 最后位置的 logits，后续严格 causal shift。
- 主预测使用正文 loglikelihood **总和**的 argmax。辅助保存 token 数和均值，但均值不参与预测或排名。选项分数完全相同则选原顺序靠前者并记录 tie；mask accuracy 相同则并列排名。

原生 Mizar 输入设定来自官方 `prepare_inference.py`：按实际采样率重采样 32kHz 后转 mono，取前 20 秒、右补零；`question a) choice b) choice…` 全部 lowercase；问题不追加 EOT；prompt 右侧 padding/truncation 至 256；audio126＋SEP0＋prompt256＝383，PAD17 的 padding 可见。

首组 mask 的全部 100 题还会进行独立、无约束 greedy 自由生成：只输入 prefix，遇 EOS0 或 300 个新 token 停止。日志 `FIRST_MASK_SAMPLE` 完整打印 output、raw decode、token IDs、GT、likelihood 预测和所有候选分数。生成只供检查，不参与 accuracy。之后 377 组和末尾 30 层基线均只算 likelihood。

官方原生为 FP32/vLLM；本轮按约定使用 BF16/Transformers 和 `use_cache=False` 全量重算，二者不宣称逐 bit 等价。模型在作业内检查实际层执行轨迹，并对代表性 mask 与独立复制、重编号的 28 层模型做 BF16 logits/likelihood 容差比较。

## 已有 manifest 的作业内评测及恢复

在**已分配 8 张 GPU 的计算作业内**，可直接执行：

```bash
bash mizar_layer_pair_search/run.sh evaluate \
  --manifest /absolute/path/calibration100/manifest.json \
  --output-dir /absolute/path/search_output
```

`evaluate` 模式完全使用冻结 manifest 中的来源，不重新解析来源参数。默认完整 Mizar 资产根为 `/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M`，checkpoint 为其下 `checkpoints/final/stage3_q4_seed20260905.ckpt`，SHA256：`0c542f63e11ed55d360feb0d66e7be0e8c6a878f44cffd302a92388f915388a2`。

普通使用仍推荐顶层提交入口。在任务中断后，用同一代码、参数、manifest 和输出目录重新提交即可续跑。逐题 journal 每条 flush/fsync，末尾未完成写入可截断恢复；完整行损坏、输入身份改变、重复/缺题会报错。已完成题目的生成和 likelihood 不会重算。每个输出目录有进程锁；不要同时提交两个作业写同一目录。变更代码或协议时改用新输出目录。

## 输出与日志

默认输出根：`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_30to28_pair_search_bf16_seed1234_v1`。

```text
run_config.json                    冻结的数据/模型/代码/协议身份
cache/prefix_000.pt … prefix_099.pt 每题 BF16 prefix 与配套元数据
cache/cache_manifest.json          缓存 ready 清单和序列化一致性检查
ranks/rank00 … rank07/             每卡模型审计、逐 mask 持久 journal、完成报告
runtime_checks.json                实际 BF16/跳层等价/缓存/8 卡加载审计
masks/mask_000/likelihood.jsonl     各组完整 100 题 likelihood 原始记录
scores/mask_000.json …             各组分数与四个子集统计
first_mask_generations.jsonl       首组 100 题独立生成及 GT/likelihood
mask_scores.jsonl                  378 组候选汇总
ranking.csv / ranking.json         按 accuracy 降序，保留并列
baseline.json                      30 层无剪 likelihood 基线，独立于候选排名
search_report.json                 完整性校验与最终状态
```

提交记录写到 `code/RSmol/log/mizar-prune28.<tag>.submission.json`；调度日志为同目录下 `mizar-prune28.<tag>.JOB.log`。每组完成立即打印 `MASK_DONE`，末尾基线为 `BASELINE_DONE`。最终 `PASS` 要求 378×100 个候选决策、100 个基线决策、首组 100 条生成、所选模式全部 worker 的完成报告、缓存和真实运行检查全部通过。CPU 测试的模拟产物不作为真实 GPU 结果交付。

100 题用于选择删层组合，accuracy 步长为 1%，每个子集为 4%；选中的组合仍需在未参与选择的数据上复核。

## 本地验证入口

从仓库根执行：

```bash
python -B tests/validate_mizar_layer_pair_search.py
# 远程依赖齐全的环境可要求真实 CPU 张量及音频集成测试全部执行：
python -B tests/validate_mizar_layer_pair_search.py --require-tensors
```

此验证覆盖解析、确定抽样、mask、13/12 分片、生成与 GT 输入隔离、恢复、求和评分、完整 379 轮模拟汇总、参考源码隔离、Python 3.10 语法、Bash 语法和真实 shell 的参数转发；shell 检查使用模拟 conda 激活且仅 `--check-args`，不会调用 vc/CUDA。可选 `--reasonaqa-json /path/test.json` 只读检查真实元数据。报告默认写 `audit_reports/mizar_layer_pair_search_validation.json`，明确标注跳过项目和 `production_gpu_run: NOT_RUN`。
