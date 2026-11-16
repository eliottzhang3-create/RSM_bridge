# 原七项ReasonAQA5000：4090单卡评测

独立入口 `code/RSmol/run_reasonaqa5000_selected7_1gpu_4090.sh`：每项一个 `vc submit` 作业，固定 `pdgpu-4090 / 1GPU / 8CPU / 32G / 1node`。`all` 只提交原七项，不包含 `official_mizar` 或原30项。

模型绑定直接复用原七项的 `mizar_s1_ced_reasonaqa5000_selected7_8gpu_eval/targets.json`，均为step17313和同run的runtime.json。geometry=8,8,4是训练身份，单卡推理保持batch1。沿用原严格loader、FP32、seed1234、TF32off、fixed20、prefix383、greedy300/EOS、原逐样本context和完整答案规范化精确匹配。

同一张GPU上的一个Python进程顺序处理完整5000题，不启动torchrun。使用原已保存的manifest：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test_close_ended_sample5000_seed1234_v1/manifest.json
```

无需重新抽样。单卡与八卡独立输出，原30项、七项8卡及MMAU/MMAR/MMSU源码保持不变。

## 提交

代码经GitHub同步后，在远程Linux仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_reasonaqa5000_selected7_1gpu_4090.sh all --check-args
bash code/RSmol/run_reasonaqa5000_selected7_1gpu_4090.sh all
```

`--check-args` 不读权重/数据、不提交，打印七项模型与输出绑定及实际 `vc_args`，资源为 `-p pdgpu-4090 -g 1 -c 8 -m 32G -n 1`。job名采用 `raqa7s-<最多40字符别名>-<12位摘要>`，最多60个ASCII字符。

镜像和环境继续使用 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`、conda `mellow_c8204d8` 和离线设置。每项独立申请一张卡，七项可由调度器分别排队/运行。

七项别名为 `base`、`recursive_5_10_5`、`serial84`、`serial48_lora_r8`、`serial84_lora_r8`、`recursive_mesh`、`mesh48_lora_r8`；也支持完整eval_variant。查看精确清单或补交单项：

```bash
bash code/RSmol/run_reasonaqa5000_selected7_1gpu_4090.sh --list-targets
bash code/RSmol/run_reasonaqa5000_selected7_1gpu_4090.sh base
```

提交日志/报告位于 `code/RSmol/log/reasonaqa5000.selected7.1gpu4090.<时间_PID>...`。单项提交失败后继续其余目标，批次最终返回非零。提交成功不代表GPU推理通过，以各目标的evaluation_report为准。

## 结果与恢复

每个训练run的单卡结果目录：

```text
eval/reasonaqa_close_ended_sample5000_seed1234_step17313_1gpu4090_v1/
```

保存完整5000条原始生成、逐题分数、整体/三类正确率及数据/代码/模型审计。原始生成逐题fsync；同代码、数据与模型下补交同一目标只生成剩余题，完整结果可直接重评分而不加载模型。完整5000题后才评分，失败不输出部分分母。旧八卡目录与raw不复用。

完成后汇总这七项：

```bash
bash code/RSmol/mizar_s1_ced_reasonaqa5000_selected7_single_eval/run.sh compare
```

默认输出 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/reasonaqa5000_selected7_seed1234_1gpu4090_v1.csv` 及同名JSON。比较器核验完整5000题、同一manifest/协议/源码与逐题输入指纹，重算完整答案精确匹配。支持原数据路径参数；只有单目标可传 `--output-dir`。

## 本地验证与同步

```powershell
python -B tests/validate_reasonaqa5000_selected7_single_eval.py
```

报告：`audit_reports/reasonaqa5000_selected7_single_eval_validation.json`。覆盖七项绑定、1GPU资源和短job名、与原单卡生成循环AST一致、合成5000题完整评分/比较/无模型重评分、中断13题后只补4987、身份变化拒绝、真实Bash与模拟vc提交/失败继续，以及原八卡/数据/计分回归。模型和GPU明确模拟，本地通过不代表远程GPU已验证。

2026-11-16本地25项全部通过，无跳过，`PASS_LOCAL`；1120份原文件SHA保持，Python3.10、两份Bash语法及diff检查通过。

本次不代为commit/push或提交真实作业。供用户同步：

```powershell
git add -- code/RSmol/mizar_s1_ced_reasonaqa5000_selected7_single_eval code/RSmol/run_reasonaqa5000_selected7_1gpu_4090.sh tests/test_reasonaqa5000_selected7_single_eval.py tests/test_reasonaqa5000_selected7_single_shell.py tests/validate_reasonaqa5000_selected7_single_eval.py audit_reports/reasonaqa5000_selected7_single_eval_validation.json
git add -p -- RM.md
git diff --cached --stat
git commit -m "Add independent single-GPU 4090 evaluation for ReasonAQA seven targets"
git push
```
