# ReasonAQA test：补充原七项的5000条、8卡3090评测

入口为 `code/RSmol/run_reasonaqa5000_selected7_8gpu_3090.sh`。`all` 只提交本次补充的7个checkpoint，每项一个 `vc submit` 作业，资源为 `pdgpu-3090 / 8GPU / 64CPU / 256G / 1node`。与原30项合计37项，继续排除 `official_mizar`。

本地Windows维护代码，经GitHub同步到远程Linux后运行。原30项的入口、评测源码、训练/模型源码以及MMAU/MMAR/MMSU实现保持原字节，避免正在运行的任务因源码身份变化而失败。

## 提交这7项

先将本次代码同步到GitHub，然后在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_reasonaqa5000_selected7_8gpu_3090.sh all --check-args
bash code/RSmol/run_reasonaqa5000_selected7_8gpu_3090.sh all
```

`--check-args` 打印7项的完整checkpoint、runtime、输出目录和 `vc_args`，不读取生产数据/权重、不提交作业。最后一条实际调用 `vc submit`；无需重新提交原30项。

使用原镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`、conda `mellow_c8204d8` 和离线设置。job名采用 `raqa7g-<最多40字符别名>-<12位摘要>`，完整长度最多60个ASCII字符。

复用同一份已保存的抽样manifest，不重新抽样：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test_close_ended_sample5000_seed1234_v1/manifest.json
```

只有该文件尚不存在时，才先执行 `bash code/RSmol/generate_reasonaqa_test5000.sh`。原生成器核对55379条候选池（AudioCaps-MCQ 23672、Clotho-MCQ 26031、ClothoAQA yes/no 5676），用独立 `random.Random(1234)` 全池均匀无放回抽5000条，再按原行号排序。

## 固定目标

所有目标均固定 `formal/checkpoints/model--step-17313.ckpt` 和同run的 `formal/runtime.json`，训练身份geometry为8,8,4；推理每卡batch1。精确路径与训练route见同目录 `targets.json`，不选择latest、不允许覆盖模型身份。

| 提交别名 | 完整eval_variant | 固定run时间戳 |
| --- | --- | --- |
| `base` | `base` | `20261026_193830617147305` |
| `recursive_5_10_5` | `recursive_5_10_5` | `20261028_160823559722647` |
| `serial84` | `recursive_5_10_5_layer8_4_inject_audio_reinject` | `20261102_112121702891801` |
| `serial48_lora_r8` | `recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8` | `20261104_131843931946322` |
| `serial84_lora_r8` | `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8` | `20261103_165410438279511` |
| `recursive_mesh` | `recursive_mesh` | `20261027_081123150934256` |
| `mesh48_lora_r8` | `recursive_mesh_layer4_8_inject_loop_lora_r8` | `20261104_161727489909082` |

查看完整清单或单项补交，别名和完整eval_variant均支持：

```bash
bash code/RSmol/run_reasonaqa5000_selected7_8gpu_3090.sh --list-targets
bash code/RSmol/run_reasonaqa5000_selected7_8gpu_3090.sh base
```

批次提交报告为 `code/RSmol/log/reasonaqa5000.selected7.8gpu.<时间_PID>.submission.json`，GPU日志为 `reasonaqa5000.selected7.8gpu.<时间_PID>.<别名>.JOB.log`。单项提交失败后继续提交余下目标，批次最终返回非零；按失败别名补交即可。

支持通用数据路径参数 `--manifest`、`--test-json`、`--audiocaps-root`、`--clotho-root`、`--clotho-aqa-root`。单项可传 `--output-dir`，`all` 禁止共用一个输出目录。

## 推理与计分

协调器预检5000条数据及模型/源码身份后，通过同一Python的 `torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0` 启动8个worker。每个worker在导入Torch前按LOCAL_RANK只暴露一张继承可见GPU，让原严格loader的 `cuda:0` 映射到各自卡。按manifest索引mod8分片，每卡恰625题，不shuffle、不补齐、不重复。

标准CED、普通递归、原始MeSH和4项multilayer分别复用原严格loader及逐样本真实context。数值协议保持FP32、eval、seed1234、TF32off、每卡batch1、fixed20、原input小写/256token、prefix383、greedy300/EOS、无KV cache完整重算；GT/caption不进入模型。

唯一指标是完整答案规范化精确匹配：去首尾空白、移除末尾一个EOS、casefold、合并连续空白，保留选项字母/正文/括号/标点。必须8个rank全部成功、恰好覆盖原5000题且通过身份/原始生成/独占GPU核验，才合并评分，分母固定5000。

## 输出与恢复

每个run默认输出到：

```text
eval/reasonaqa_close_ended_sample5000_seed1234_step17313_8gpu_v1/
```

这7个run与原30个run不同，旧完整七项的结果目录也不同。根目录保存run_config、dataset_audit、selected_records、完整5000条raw_generations/progress、model_audit、accuracy、per_sample_results及evaluation_report；`ranks/rank00/` 至 `rank07/` 各保存625题的原始生成、进度、模型审计和rank_report。

原始生成逐题fsync。同代码、配置与资产下，重提单项会恢复每卡剩余题，已完成rank不加载模型；8个rank已完成时直接核验并汇总。仅截去末尾未写完的记录，完整坏行或身份变化拒绝。不要同时向同一目标目录提交两个作业。

## 汇总这7项

完成后在远程执行：

```bash
bash code/RSmol/mizar_s1_ced_reasonaqa5000_selected7_8gpu_eval/run.sh compare
```

默认生成 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/reasonaqa5000_selected7_seed1234_8gpu_v1.csv` 和同名JSON。逐项复核8卡完整结果、共同manifest/数据/协议/代码及逐题prompt/mask/音频输入指纹。原30项继续用原比较命令；两批分别生成汇总文件。

## 本地验证与同步

```powershell
python -B tests/validate_reasonaqa5000_selected7_8gpu_eval.py
```

报告为 `audit_reports/reasonaqa5000_selected7_8gpu_eval_validation.json`。新增回归验证7项精确绑定、与原30项互斥且合计37项、原推理函数AST一致、8个真实OS进程运行实际worker入口、5000题合并/评分/比较、rank中断13题后只补612题、错身份/坏记录/重复GPU拒绝、真实Bash入口及模拟vc提交。Torch/CUDA/模型在新增流程测试中明确模拟；缺少真实依赖的原测试明确跳过。本地验证不能代替远程严格加载和GPU推理。

2026-11-16联合验证：236项中221通过、15项因缺真实依赖跳过，`PASS_LOCAL_WITH_SKIPS`。1104份原源码SHA不变，Python3.10及7份Bash语法检查通过。

本次未代为commit/push或提交真实作业。供用户执行的同步命令：

```powershell
git add -- code/RSmol/mizar_s1_ced_reasonaqa5000_selected7_8gpu_eval code/RSmol/run_reasonaqa5000_selected7_8gpu_3090.sh tests/test_reasonaqa5000_selected7_8gpu_eval.py tests/test_reasonaqa5000_selected7_8gpu_shell.py tests/validate_reasonaqa5000_selected7_8gpu_eval.py audit_reports/reasonaqa5000_selected7_8gpu_eval_validation.json
git add -p -- RM.md
git diff --cached --stat
git commit -m "Add seven supplemental ReasonAQA5000 eight-GPU evaluations"
git push
```

RM.md仅选本次新增记录，保留其他任务的修改。
