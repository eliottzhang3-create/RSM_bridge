# ReasonAQA test 固定5000条：每个checkpoint用8卡3090评测

入口：`code/RSmol/run_reasonaqa5000_eval_8gpu_3090.sh`。一个目标对应一个 `vc submit` 作业，每个作业申请 `pdgpu-3090 / 8GPU / 64CPU / 256G / 1node`；`all` 提交30个这样的作业。

本地 Windows 只维护代码，GitHub 中转；真实数据、权重和推理仍在远程 Linux。新增独立包和入口，原单卡 ReasonAQA、完整七项 ReasonAQA、MMAU/MMAR/MMSU 与训练/模型源码保持原字节。八卡模式使用独立结果目录，撤下的旧单卡作业及其已生成文件无需删除或改名。

## 远程完整提交

将本次新增文件经 GitHub 同步后，在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_reasonaqa5000_eval_8gpu_3090.sh all --check-args
bash code/RSmol/run_reasonaqa5000_eval_8gpu_3090.sh all
```

`--check-args` 打印30项的 checkpoint/runtime/output 绑定、8卡分片策略及实际 `vc_args`（包括 `-p pdgpu-3090 -g 8 -c 64 -m 256G -n 1` 和短 job 名），不读权重/数据、不提交。

使用原镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`、conda `mellow_c8204d8` 和离线设置；不增加自动 smoke 作业。job 名为 `raqa8g-<最多40字符别名>-<12位摘要>`，完整长度不超过60个ASCII字符，摘要包含完整批次标记与完整别名。

原来已生成的5000条采样文件直接复用，不重新抽样：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test_close_ended_sample5000_seed1234_v1/manifest.json
```

只有尚未生成该文件时，才先执行原生成器：

```bash
bash code/RSmol/generate_reasonaqa_test5000.sh
```

生成器仍严格核对55379条候选池（AudioCaps-MCQ23672、Clotho-MCQ26031、ClothoAQA yes/no5676），独立 `random.Random(1234)` 全池均匀无放回抽5000条、按原始行号排序。所有模型复用同一权威 manifest。

## 目标范围与单项补交

直接复用 `mizar_s1_ced_reasonaqa5000_eval/targets.json` 的固定30项，完整别名/run/checkpoint 清单见该包 README；没有动态选择 latest。继续排除 `official_mizar` 和先前已经评测的七项。三项 step34626 的 geometry=8,4,4，其余 step17313 的 geometry=8,8,4；这些是训练身份校验，推理仍每卡batch1。

查看清单或单项提交：

```bash
bash code/RSmol/run_reasonaqa5000_eval_8gpu_3090.sh --list-targets
bash code/RSmol/run_reasonaqa5000_eval_8gpu_3090.sh init0904
```

别名和完整 eval_variant 均支持。批量提交报告保存在 `code/RSmol/log/reasonaqa5000.8gpu.<时间_PID>.submission.json`，GPU日志为 `reasonaqa5000.8gpu.<时间_PID>.<别名>.JOB.log`；失败项继续记录，其余目标继续提交，只要有提交失败最终返回非零。SUBMITTED 仅代表提交成功，GPU评测结果以各自 evaluation_report 为准。

`mesh_inject48` 仍是用户提供的候选路径，提交该项前以 CPU 调用原 route validator 核对 checkpoint/runtime/源码/合同/step/geometry；失败不提交此项，继续其余29项。确认候选run错误后可显式修正：

```bash
bash code/RSmol/run_reasonaqa5000_eval_8gpu_3090.sh all \
  --mesh-inject48-run-root '/实际确认的训练run目录'
```

或单项补交：

```bash
bash code/RSmol/run_reasonaqa5000_eval_8gpu_3090.sh mesh_inject48 \
  --run-root '/实际确认的训练run目录'
```

单候选可继续传 `--checkpoint`、`--runtime-config`；非canonical checkpoint需要同时明确 runtime-config 和 output-dir。其他29项拒绝模型身份覆盖。通用数据参数保留 `--manifest`、`--test-json`、三个音频root；单目标允许 `--output-dir`，all禁止共用一个输出目录。

## 8卡执行与评分

计算节点先运行单个父进程协调器，预检原 test.json、manifest、5000条音频路径及源码/模型身份并认领输出目录。它通过同一 conda Python 启动：

```text
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 --module mizar_s1_ced_reasonaqa5000_8gpu_eval.worker ...
```

这是8进程数据并行推理，每进程持有完整模型，不包装训练用的DDP梯度同步层，也不进行逐题collective。worker读取 `LOCAL_RANK`，在导入Torch前从继承的8张 `CUDA_VISIBLE_DEVICES` 中选一张，原严格 loader 看到的 `cuda:0` 就映射到该rank独占的卡。未设置该变量时按本机0至7分配；可见卡少于/多于8张、重复卡、错误world size或多节点配置均明确失败。

按 **manifest中的索引 % 8 = rank** 分题：rank0处理0、8、16……，rank1处理1、9、17……；每卡恰好625条，不shuffle、不补齐、不重复。每rank独立锁、独立raw、独立进度和模型审计。

八个worker全部成功后，父进程核对每rank625条、身份/record/generation SHA、实际GPU分配，再按原manifest顺序合并。必须精确覆盖原5000个ID一次，才调用原评分器；缺卡、坏行、错序、重复、缺题或模型错误均返回非零，不输出部分题量的正确率。父进程在合并前再次核对数据/音频元信息、checkpoint/runtime和源码未变化。

唯一指标继续为完整答案规范化精确匹配：去首尾空白、移除末尾一个EOS、casefold、合并连续空白，保留字母/正文/括号/标点。仅输出字母、删除答案正文或标点不一致均按原规则判错，分母固定5000。

原数值协议保持：FP32、eval/frozen、seed1234、TF32off、每卡batch1、原input小写/256token、固定20秒音频分两段10秒、prefix383、greedy300/EOS、无KV cache完整重算、原route的逐样本注入context和prompt mask。caption/GT不进入模型。没有修改任何训练模型、LoRA、router、gate或严格加载检查。

## 输出与断点恢复

每个run使用新的独立输出目录：

```text
eval/reasonaqa_close_ended_sample5000_seed1234_step<step>_8gpu_v1/
```

| 位置 | 内容 |
| --- | --- |
| 根目录 `run_config.json` | 完整模型/数据/代码/原推理协议及8卡分片身份 |
| 根目录 `selected_records.jsonl` / `dataset_audit.json` | 同一5000条原记录、真实音频路径及抽样审计 |
| `ranks/rank00/` 至 `rank07/` | 各625题的run_config、raw_generations、progress、model_audit和rank_report |
| 根目录 `raw_generations.jsonl` / `progress.jsonl` | 完整合并后的5000题原顺序输出 |
| 根目录 `model_audit.json` | 8个rank各自的GPU绑定、原严格加载及执行审计 |
| 根目录 `accuracy.json` / `per_sample_results.jsonl` | overall和三类唯一指标、逐题得分 |
| 根目录 `evaluation_report.json` | 整体PASS/FAILED、8个rank覆盖情况及错误堆栈 |

各rank原始生成逐题append+fsync。相同代码、配置、资产下重提失败目标，可恢复每个rank未完成的题；已完成的rank不再加载模型。恢复仅截去最后一次未完成的无换行写入，完整坏行/身份变化均拒绝。八个rank均已有完整PASS时，父进程直接核验并合并/重评分，不再启动torchrun或加载模型。

重提某一项仍申请8卡；不复用旧单卡raw，不因撤下作业而清除旧结果。不应同时向同一目标目录提交两个作业。已经失败的单项按别名补交即可。

## 汇总

30项完成后在远程执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/mizar_s1_ced_reasonaqa5000_8gpu_eval/run.sh compare
```

默认输出 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/reasonaqa5000_remaining30_seed1234_8gpu_v1.csv` 和同名JSON。要求30项均为8卡完整结果，保留原比较器的共同manifest/数据/协议/代码、完整5000条raw哈希、重算分数与逐题实际prompt/mask/音频输入指纹核验。

更正候选run或使用自定义输出时，用重复 `--report-dir '/真实评测结果目录'` 显式列出要比较的结果；默认比较仍读取固定30项的新8卡目录。缺失任何默认目标会报错。

## 本地验证和同步

```powershell
python -B tests/validate_reasonaqa5000_8gpu_eval.py
```

报告为 `audit_reports/reasonaqa5000_8gpu_eval_validation.json`。新回归包括八个真实OS进程运行实际worker入口（Torch/CUDA和模型为明确模拟）、完整5000题合并/评分/比较、一个rank中断13题后只补612题、坏记录/错GPU/错误覆盖及身份拒绝、30路资源和60字符job名、真实Bash入口与模拟vc、旧单卡及MMAU/MMAR/MMSU回归。1090份原源码字节SHA保持；真实Torch/音频依赖缺失时相应原测试明确跳过。本地合成通过不能代替远程8卡严格加载、GPU推理和数值验证。

2026-11-16本地联合验证：220项中205通过、15项因缺少真实依赖明确跳过，`PASS_LOCAL_WITH_SKIPS`；Python3.10语法、五份Bash语法及原源码SHA检查通过。

本次未代为commit/push或提交真实GPU作业。供用户执行的精确同步命令：

```powershell
git add -- code/RSmol/mizar_s1_ced_reasonaqa5000_8gpu_eval code/RSmol/run_reasonaqa5000_eval_8gpu_3090.sh tests/test_reasonaqa5000_8gpu_eval.py tests/test_reasonaqa5000_8gpu_shell.py tests/validate_reasonaqa5000_8gpu_eval.py audit_reports/reasonaqa5000_8gpu_eval_validation.json
git add -p -- RM.md
git diff --cached --stat
git commit -m "Add isolated eight-GPU ReasonAQA evaluation on pdgpu-3090"
git push
```

RM.md仅选本次新增记录，保留其他任务的修改。
