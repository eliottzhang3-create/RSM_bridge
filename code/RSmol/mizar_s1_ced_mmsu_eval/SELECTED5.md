# 用户指定的五个 Mizar MMSU 测评目标（2026-11-13）

已核对现有 MMAU test mini、MMAR 的提交分支和 runtime 注册。五个目标全部复用现成适配，只新增 `code/RSmol/run_mizar_mmsu_selected5_5090.sh` 组装固定参数；既有 MMAU/MMAR、MMSU 推理和 scorer 实现均不修改。

## 目标映射

| 提交名称 | 模型与实际适配 | 固定训练 run / release |
| --- | --- | --- |
| `official_mizar` | Mizar-159M 官方最终权重；原 `mizar_official_eval/runtime.py` 严格 release loader | Stage3 / Q4 / seed20260905 / update200 |
| `base` | 原标准 CED-Small S1；`mizar_s1_ced_eval_runtime.py` | `formal_20261026_193830617147305` |
| `recursive_5_10_5` | 原非 MeSH 的 5+10×2+5 递归；`mizar_s1_ced_recursive_eval_runtime.py` | `formal_20261028_160823559722647` |
| `serial84_lora_r8` | 原 8→4 中间层注入 + audio reinjection + Loop-LoRA r8；`mizar_s1_ced_multilayer_eval_runtime.py` | `formal_20261103_165410438279511` |
| `recursive_mesh` | 原始 recursive_mesh；`mizar_s1_ced_recursive_mesh_eval_runtime.py` | `formal_20261027_081123150934256` |

`serial84_lora_r8` 的完整 eval variant 为 `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8`，对应用户指定的完整训练 checkpoint；严格加载其中的全部参数，使用原 runtime 的逐样本注入 context 与解码器。

官方 Mizar 本身沿用此前 MMAU/MMAR 的发布包位置：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M/
  checkpoints/final/stage3_q4_seed20260905.ckpt
  assets/smollm2-135m/...
  assets/ced-small/...
```

其权重与配套资产按原 `mizar_official_eval/release_manifest.json` 核验。它是官方最终发布 plain state_dict，使用自己的发布 step，不传训练用的 `runtime.json`、step17313 或 geometry。

其余四条 checkpoint 精确固定为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_5090/formal_20261026_193830617147305/formal/checkpoints/model--step-17313.ckpt
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_5090/formal_20261028_160823559722647/formal/checkpoints/model--step-17313.ckpt
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8_5090/formal_20261103_165410438279511/formal/checkpoints/model--step-17313.ckpt
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_5090/formal_20261027_081123150934256/formal/checkpoints/model--step-17313.ckpt
```

每条配套 runtime 都是相应 run 下的 `formal/runtime.json`。四条均明确检查 `total_step=17313`、训练 geometry `world_size=8,microbatch=8,gradient_accumulation=4`；评测本身始终是 batch1、单 GPU。训练实现目录与旧 MMAU/MMAR 的相应 variant 完全一致。

## 远程提交命令

本地提交并 push 后，在远程仓库根执行以下五条命令，每条提交一个独立的完整 5000 题 MMSU 作业：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# 1. 官方 Mizar-159M 最终权重
bash code/RSmol/run_mizar_mmsu_selected5_5090.sh official_mizar

# 2. 标准 CED-Small S1，step17313
bash code/RSmol/run_mizar_mmsu_selected5_5090.sh base

# 3. 普通 5-10-5 递归，step17313
bash code/RSmol/run_mizar_mmsu_selected5_5090.sh recursive_5_10_5

# 4. 串行 8→4 注入 + audio reinjection + Loop-LoRA r8，step17313
bash code/RSmol/run_mizar_mmsu_selected5_5090.sh serial84_lora_r8

# 5. 原始 recursive_mesh，step17313
bash code/RSmol/run_mizar_mmsu_selected5_5090.sh recursive_mesh
```

五条 MMSU 作业均提交至既有 MMSU 资源池 `pdgpu-5090`，每作业 1 GPU / 8 CPU / 32G / 1 node，沿用原镜像、`mellow_c8204d8` 和离线设置。此前官方 Mizar 的 MMAU/MMAR 提交分支使用 `pdgpu-4090`；本次没有改动那个分支。

数据默认使用 `/hpc_stor03/sjtu_home/jinwei.zhang/data/MMSU` 下的 `MMSU data` 与 `MMSU code`。官方大写题面、32kHz mono/fixed20、FP32、prompt256、greedy300、无 KV、保留原始 EOS、官方 scorer 和实际评分分母检查均沿用已交付 MMSU 协议。没有自动 smoke 或额外 GPU 前置任务。

## 结果目录与续跑

官方 Mizar 的结果在：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_official_stage3_q4_seed20260905_matched_ced_v1/mmsu/
```

其余四条各自在对应训练 run 下的 `eval/mmsu_step17313/`，五个输出目录相互独立。各目录中的 `evaluation_report.json`、`official_evaluation.txt`、`predictions_official.jsonl` 保存正式报告、官方 scorer 输出及逐题原始回答。

再次执行同一目标命令，会按现有 MMSU 身份检查恢复该目录的未完成样本。需要全新评测时可以显式指定新输出目录：

```bash
bash code/RSmol/run_mizar_mmsu_selected5_5090.sh base --output-dir '/新输出目录/mmsu'
```

只打印生效参数、不提交作业：

```bash
bash code/RSmol/run_mizar_mmsu_selected5_5090.sh serial84_lora_r8 --check-args
```

此模式不读取远程 checkpoint 或数据，仅验证参数绑定。正式作业由已有严格 loader 核验真实权重与资产。本地没有读取远程生产权重，不声明远程模型加载或推理已通过。

可覆盖的参数仅为 `--mmsu-root`、`--dataset-dir`、`--evaluation-script`、`--output-dir`、`--parquet-batch-size`，支持空格路径及 `--option=value`。目标 checkpoint/runtime/variant/step/geometry 固定；其他模型继续使用通用 MMSU 入口。

## 本地代码同步与回归

本轮在已有 MMSU 实现之上，只需同步以下改动。PowerShell 在本地仓库根执行：

```powershell
git add -- code/RSmol/run_mizar_mmsu_selected5_5090.sh code/RSmol/mizar_s1_ced_mmsu_eval/README.md code/RSmol/mizar_s1_ced_mmsu_eval/SELECTED5.md tests/test_mizar_mmsu_selected5.py tests/validate_mizar_s1_ced_mmsu_eval.py audit_reports/mizar_mmsu_selected5_validation.json RM.md
git commit -m "Add MMSU submission presets for five selected Mizar checkpoints"
git push
```

针对新入口的本地回归：

```bash
python -B -m unittest discover -s tests -p test_mizar_mmsu_selected5.py -v
```

测试执行真实五目标入口、通用提交入口、计算 shell 与参数解析，`vc` 和 conda 激活为显式 fixture；验证精确 checkpoint/runtime/route、8/8/4、单卡资源、五个独立输出、含空格参数、只解析不提交及错误参数拦截。测试不会加载生产权重或提交真实作业。该用例也已接入完整 MMSU validator。

本轮针对性验收合并新入口用例、已有 MMSU shell 用例及原 28 份 MMAU/MMAR 评测文件的源码校验，共 8 项全部通过；记录在 `audit_reports/mizar_mmsu_selected5_validation.json`。原 MMSU runner/runtime/storage/scoring/dataset、通用提交脚本和所有旧模型适配均未修改。
