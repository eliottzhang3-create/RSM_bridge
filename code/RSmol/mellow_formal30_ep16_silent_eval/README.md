# 三模型 epoch-16、静音第二槽统一评测

启动时不再遍历 1,000 个音频文件、不再读取并哈希完整 MMAU parquet，也不再在 checkpoint 完整加载后额外顺序读取它计算 SHA-256。metadata/scorer 仍做原有校验；各音频在其样本进入推理时按需读取并计算 waveform hash。MMAR 使用本隔离入口内的轻量 artifact audit 跳过全量音频路径预检，共享 evaluator 文件保持原样。

本路线完整复制当前 v5 `mellow_formal30_eval/` 后隔离修改，只接收三条 formal30 训练路线的 epoch-16 音频 checkpoint。相对母本，实验变化仅为 epoch-16 权重和第二槽输入静音波形；原 `submit_mellow_formal30_eval_5090.sh`、原评测目录、训练代码及共享数据/评分辅助代码均未修改。

## 正式提交

代码经 GitHub 同步后，在远端 `code/RSmol/` 执行：

```bash
bash submit_mellow_formal30_ep16_silent_eval_5090.sh both smollm2
bash submit_mellow_formal30_ep16_silent_eval_5090.sh both mesh_x2
bash submit_mellow_formal30_ep16_silent_eval_5090.sh both mesh_x4
```

每条命令各提交 MMAU test-mini 和 MMAR 两个独立单卡作业，合计六个。无 smoke gate。资源固定 `pdgpu-5090 -c 8 -m 32G -g 1 -n 1`，容器和 `mellow_c8204d8` conda 环境沿用 v5。`MELLOW_CONDA_BASE` / `MELLOW_CONDA_ENV` 可指定已有环境，不安装依赖。

三个别名在 [checkpoints.tsv](checkpoints.tsv) 中逐字绑定用户指定的 `model--epo-16.ckpt`：

| 别名 | 训练 run | checkpoint job |
|---|---|---|
| `smollm2` | `formal_30epochs_20261015_110258876935535` | `mellow_adamw_cosine_reasonaqa_gbs256_formal30_20_20261015_030306474730662` |
| `mesh_x2` | `formal_30epochs_20261016_080547470116428` | `mellow_adamw_cosine_reasonaqa_gbs256_formal30_mesh_20_20261016_000555705751996` |
| `mesh_x4` | `formal_30epochs_20261016_014059864603396` | `mellow_adamw_cosine_reasonaqa_gbs256_formal30_meshx4_20_20261015_191433441114827` |

也可提交单任务或显式传入文件路径：

```bash
bash submit_mellow_formal30_ep16_silent_eval_5090.sh mmau_test_mini smollm2
bash submit_mellow_formal30_ep16_silent_eval_5090.sh mmar /absolute/path/to/model--epo-16.ckpt \
  --runtime-config /same/run/root/runtime_30epochs.yaml \
  --output-dir /absolute/path/to/new/evaluation_output \
  --dataset-dir /hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR
```

`both --output-dir PATH` 使用 `PATH/mmau_test_mini/` 和 `PATH/mmar/`；`--dataset-dir` 仅适用于单任务。默认 runtime 从 checkpoint 所属 run 读取 `runtime_30epochs.yaml`；显式 runtime 也必须属于同一 run。日志和报告记录实际 checkpoint 路径、架构、`epoch_completed=16`、`num_epochs=30` 和当前步数。

## 统一输入与评分

| 项目 | 固定设置 |
|---|---|
| 精度与模式 | FP32，关闭 autocast/TF32，全模块 eval，inference mode，batch 1 |
| 第一槽 MMAU | 官方 WAV 优先，缺失时回退 parquet 音频；重采样后取前两个通道平均；短音频重复、长音频按 seed 0 + 样本 ID 固定种子裁剪，与 v5 一致 |
| 第一槽 MMAR | 沿用 v5 通道平均、重采样、首 10 秒及短音频右补零 |
| 第二槽 | `torch.zeros_like(audio1)`，FP32 `[1,320000]`；经原生模型单独编码，不把 embedding 置零，不复用第一槽 |
| prefix | 第一槽 129 + 分隔符 token 0 + 第二槽 129 + 分隔符 token 0 = 260 |
| prompt | 保留正文大小写/标点与选项顺序，按实际选项数添加 `a)` 至 `j)` |
| 分词与起点 | 调用对应 formal30 dataset 分词函数，最多 129 tokens、保留最终 EOS、不补齐；从 prompt EOS 的最后位置预测第一个答案 token |
| 生成 | 两任务均最多 300 tokens，top-p 0.8 后 argmax，temperature 1，`use_cache=False`，生成 EOS 停止 |
| 评分输入 | 同一次完整 `generated_text` 同时用于前缀评分和官方 scorer；不预先删除标签、括号或 trim |
| 前缀评分 | `prediction.split(')')[0].lower()` 与正确标签精确比较；原官方 scorer 保持原样 |

每个 benchmark 保留完整 1000 条分母；数据缺失/无法生成按母本规则记录，不缩减分母。第二槽审计写入 `audio2_policy`、`audio2_nonzero_elements=0`、`audio2_max_abs=0.0`、实际静音 waveform SHA256、`audio2_reused=false` 和两次原生编码记录。第一槽波形 hash、prompt IDs、生成 IDs 和停止原因也保留。

## checkpoint 与隔离边界

只接受 `route_variant=formal_30epochs_compact`、`epoch_completed=16`、`num_epochs=30` 的 schema-v2 checkpoint。保留架构/state keys、GBS256、global-token loss、FP32、HTSAT 冻结、学习率等母本合同审计。scheduler 的 `last_step` 必须等于 checkpoint 当前 `total_step`，当前步数允许小于总调度步数；不要求训练已完成 30 epochs。

只加载完整 `state_dict`；不恢复 optimizer/scheduler/RNG，不检查已删除的训练 staging，不绑定节点/PID。模型构造所需的 HTSAT、canonical SmolLM2 tokenizer/权重及递归文本初始化目录仍须可读。

新提交入口递归拆分 `both` 时调用本入口，作业命令只运行本目录 `run.sh`。本目录内 Python 模块使用包内导入，避免同进程与旧 evaluator 模块串用。`audio.py` / `run.sh` 与 v5 字节相同；数据与官方 scorer 辅助文件是只读依赖。复制时的原始来源哈希保存在 [parent_v5_sources.json](parent_v5_sources.json)，仅作 provenance，不作为启动 gate。

默认输出：

```text
<训练 run>/eval_formal30_ep16_silent_audio2_v2/<task>/<checkpoint-job>/model--epo-16/full/
```

独立协议标识、`.full.ep16_silent_eval_identity.json`、lock 文件和 `e16s-` 作业前缀防止与 v5 混写。同一路线同协议中断后重复命令可恢复；修复 terminal skip 后请用新输出目录重跑。

## 结果核对

两个 benchmark 各自比较三模型报告：

```bash
python mellow_formal30_ep16_silent_eval/compare.py \
  /path/to/smollm2/evaluation_report.json \
  /path/to/mesh_x2/evaluation_report.json \
  /path/to/mesh_x4/evaluation_report.json \
  --output-csv /path/to/ep16_silent_comparison.csv
```

要求三份报告均为本路线 epoch-16/30 的 comparable PASS；核对协议、数据/scorer、tokenizer、软件、attention 和非架构 LM 配置，并逐样本比较 prompt IDs、第一槽片段和第二槽静音 hash。官方分数与前缀分数均要求完整 1000 条分母。

## 本地验证边界

```powershell
python -m unittest tests.test_mellow_formal30_ep16_silent_eval -v
```

测试复制自母本并适配 epoch-16/静音槽及 v5 完整文本评分，补充精确路径、原生双槽编码、非零拒绝、六作业模拟提交、epoch/静音比较校验。另检查 Python/Bash 语法、母本及只读依赖哈希、与母本的核心函数差异。本地没有 PyTorch、远端权重和 benchmark；未运行 smoke 或真实 CUDA 评测，实际分数以远端正式作业为准。
