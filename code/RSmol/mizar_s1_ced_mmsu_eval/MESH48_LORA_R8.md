# 指定 MeSH 4→8 注入 / Loop-LoRA r8 checkpoint 的 MMSU 评测

新增快捷入口 `code/RSmol/run_mizar_mmsu_mesh48_lora_r8_5090.sh`。精确绑定用户给定的：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_loop_lora_r8_5090/formal_20261104_161727489909082/formal/checkpoints/model--step-17313.ckpt
```

配套 runtime 为同一 run 的 `formal/runtime.json`；variant 为
`recursive_mesh_layer4_8_inject_loop_lora_r8`，训练 geometry 为 `8,8,4`。
训练实现目录为 `mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8`。

MMAU/MMAR 已有此精确模型适配，MMSU 通用入口通过共享 registry 调用同一严格 multilayer loader、
逐样本 context 和 greedy decoder。本次只新增固定目标提交入口，完整加载最终音频 checkpoint；
不修改模型、训练、旧提交入口或评测核心文件。原 MeSH 五槽、两循环、4→8 注入、Loop-LoRA r8 均由原合同核验。

## 提交命令

本地同步代码后，在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_mmsu_mesh48_lora_r8_5090.sh
```

这是一次完整5000题 MMSU 作业，资源为 `pdgpu-5090`、1 GPU、8 CPU、32G、1 node，
沿用原镜像、`mellow_c8204d8` 环境和离线设置，无自动 smoke 或额外 GPU 测试。
默认读取 `/hpc_stor03/sjtu_home/jinwei.zhang/data/MMSU` 下的 `MMSU data` 和 `MMSU code`。

输出目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_loop_lora_r8_5090/formal_20261104_161727489909082/eval/mmsu_step17313/
```

音频32kHz mono/fixed20、FP32、prompt256、greedy300、无KV及官方题面保持现有设置。
评分沿用用户确认的 `mmsu_answer_text_contains_v2`：忽略生成选项字母，归一化生成/完整GT文本后做包含匹配，
命中编码为对应正确字母，未命中编码为空字符串，再调用未修改官方 scorer；5000题均参与分母。
解释、否定句或多个候选中提到GT也会命中。定义与解析审计见 [RESCORE.md](RESCORE.md)。

保留原始 `raw_generations.jsonl` 和 `predictions_official.jsonl`，另存评分副本与解析详情；
最终查看 `evaluation_report.json` 和 `official_evaluation.txt`。

仅检查参数，不提交、不读权重或数据：

```bash
bash code/RSmol/run_mizar_mmsu_mesh48_lora_r8_5090.sh --check-args
```

可覆盖 `--mmsu-root`、`--dataset-dir`、`--evaluation-script`、`--output-dir` 和
`--parquet-batch-size`，支持路径空格及 `--option=value`；模型身份固定。
已有同版本输出按原断点身份规则恢复，旧版本输出使用独立目录或完成后离线重评，不放宽身份校验。

## 本地同步与验证

PowerShell 在本地仓库根自行提交推送：

```powershell
git add -- code/RSmol/run_mizar_mmsu_mesh48_lora_r8_5090.sh code/RSmol/mizar_s1_ced_mmsu_eval/MESH48_LORA_R8.md tests/test_mizar_mmsu_mesh48_lora_r8.py audit_reports/mizar_mmsu_mesh48_lora_r8_validation.json RM.md
git commit -m "Add fixed MMSU evaluation target for MeSH48 Loop-LoRA r8"
git push
```

针对性测试：

```powershell
python -B -m unittest discover -s tests -p test_mizar_mmsu_mesh48_lora_r8.py -v
```

测试执行真实新入口、原通用提交与计算 shell、实际 Python 参数解析，仅 vc/conda 使用显式 fixture，
检查精确模型/run/route/step/geometry、资源、含空格参数、固定身份、只解析不提交和现有源码摘要。
本地测试不加载生产权重，也不提交真实GPU作业。

2026-11-13 本地验收：7项全部通过，无跳过；132份现有评测源码摘要与105份训练源码LF摘要均保持。
新shell语法、测试源码Python3.10语法通过。报告：`audit_reports/mizar_mmsu_mesh48_lora_r8_validation.json`。
未访问远程生产checkpoint或执行真实评测，当前状态为代码就绪。
