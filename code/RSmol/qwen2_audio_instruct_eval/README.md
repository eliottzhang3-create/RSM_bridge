# Qwen2-Audio-Instruct MMAU test-mini / MMAR 隔离评测

本目录记录 Qwen2-Audio-Instruct 对比评测路线的运行合同。代码只复用现有
`audio_5_10x2_5_mesh_mellow_shared_store_configurable_epochs` 评测器的数据、顺序、
续跑和评分层；模型加载、processor、ChatML 与生成逻辑均为隔离实现，不会读取或修改
RSmol checkpoint。

当前状态（2026-09-29）：本地代码与静态合同测试就绪；权重、数据集和官方 scorer
仅存在于远程 Linux，因此尚未登记远程 artifact preflight、GPU smoke 或 full PASS。

## 固定合同

- 模型：`/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen2Audio-Instruct`。
- 加载：`Qwen2AudioForConditionalGeneration` + `AutoProcessor`，仅使用本地文件，
  `trust_remote_code=False`，单张 `cuda:0`，参数与推理均锁定 BF16。
- 对话：一个 user turn，内容顺序固定为 audio、benchmark prompt；不添加 system prompt，
  不添加“只回答字母”等额外语义指令。
- 同一份未经预解析的 decoded prediction 同时送入 choice-label-prefix scorer 与官方 scorer。
- smoke 固定前 5 条；full 与 smoke 必须复用同一 output directory，以验证 append-only resume。

### MMAU test-mini

完全复用 Mellow 作者回复协议：parquet 物理顺序、官方 `<id>.wav` 优先、小写固定选项
prompt、129 个 Qwen prompt token 上限、32 kHz 下 repeat/random 10 秒裁剪、最多 300 个
生成 token，以及 top-p=0.8 过滤后 argmax。完整评测保持 MMAU-v05.15.25 的 1000 条
官方分母；逐样本 skip 写空预测并计错。

输出同时包含 Mellow choice-label-prefix 分数与 MMAU-v05.15.25 官方分数。主要文件为
`predictions_fixed_order.json`、`mellow_author_reply_evaluation.json`、
`official_evaluation.txt` 和 `evaluation_report.json`。

### MMAR

完全复用官方 metadata 顺序、固定 choices、动态探测的 prediction key、32 kHz 首 10 秒/
右侧补零和 1000 条完整分母。Qwen 使用 deterministic greedy、`use_cache=False`、最多
32 个生成 token。

输出同时包含 choice-label-prefix 分数与官方 MMAR 分数。主要文件为
`predictions_official.json`、`choice_label_prefix_evaluation.json`、
`official_evaluation.txt` 和 `evaluation_report.json`。

## 远程执行顺序

先在远程 `code/RSmol` 目录运行权重与 processor 预检：

```bash
bash run_qwen2_audio_instruct_eval_preflight.sh
```

报告必须为 `PASS` 后再提交 GPU 作业。MMAU smoke 与 full：

```bash
MMAU_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/qwen2_audio_instruct_eval/mmau_test_mini_dual_scoring_v1
RSMOL_QWEN2_AUDIO_MMAU_MODE=smoke RSMOL_QWEN2_AUDIO_MMAU_OUTPUT_DIR="$MMAU_DIR" \
  bash run_mmau_test_mini_qwen2_audio_instruct_3090.sh
RSMOL_QWEN2_AUDIO_MMAU_MODE=full RSMOL_QWEN2_AUDIO_MMAU_OUTPUT_DIR="$MMAU_DIR" \
  bash run_mmau_test_mini_qwen2_audio_instruct_3090.sh
```

MMAR smoke 与 full：

```bash
MMAR_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/qwen2_audio_instruct_eval/mmar_dual_scoring_v1
RSMOL_QWEN2_AUDIO_MMAR_MODE=smoke RSMOL_QWEN2_AUDIO_MMAR_OUTPUT_DIR="$MMAR_DIR" \
  bash run_mmar_qwen2_audio_instruct_3090.sh
RSMOL_QWEN2_AUDIO_MMAR_MODE=full RSMOL_QWEN2_AUDIO_MMAR_OUTPUT_DIR="$MMAR_DIR" \
  bash run_mmar_qwen2_audio_instruct_3090.sh
```

两个提交入口均使用 `pdgpu-3090`、1 GPU、8 CPU、64G memory。远程结果只有在
`evaluation_report.json` 中 inference coverage、双评分及对应 artifact audit 均为 PASS
后才可用于正式比较。
