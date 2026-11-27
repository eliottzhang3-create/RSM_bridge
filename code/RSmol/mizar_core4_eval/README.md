# 三个 AudioMCQ ×4 最终 checkpoint：MMAU / MMAR / MMSU

此次只有 **3 模型 × 3 benchmark = 9 项**。不提交 ReasonAQA，也不重跑官方 baseline。
本地 Windows 改代码，经 GitHub 同步后在远程 Linux 提交；本地不读取生产 checkpoint/音频，不自动 commit/push 或申请 GPU。

## 固定模型与加载

`targets.json` 保存用户给出的三个完整绝对路径；默认不需要手工传 checkpoint，也不扫描 latest。

| 提交别名 | 训练输出目录 | run | 核心层（1-based） |
| --- | --- | --- | --- |
| `l9_12x4` | `mizar_final_l9_12x4_audiomcq_gbs128_3epochs_5090` | `formal_20261127_014210339558630` | 9–12 |
| `l14_17x4` | `mizar_final_l14_17x4_audiomcq_gbs128_3epochs_3090` | `formal_20261127_021300367682028` | 14–17 |
| `l19_22x4` | `mizar_final_l19_22x4_audiomcq_gbs128_3epochs_3090` | `formal_20261127_021312627006100` | 19–22 |

三者均为 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/<训练输出目录>/<run>/formal/checkpoints/model--step-16890.ckpt`。
自动使用相邻 `formal/runtime.json`，核验各自路线、合同 SHA、训练代码 SHA、720524 行、8×4×4、3epochs/16890 步及最终游标 `(3,0)`。
评测只恢复模型，不创建或恢复 optimizer/scheduler/RNG，不使用训练中“已完成不能 resume”的门槛。

训练源目录保留历史 `x3_strong_reason` 文件名，当前内容已更新为四循环 AudioMCQ 配方。
`source_manifest.json` 固定提交 `9575eafd0901c481fbd934cc90514f8b2554eaa3` 中各路线的 Python/Shell/release manifest 逐文件 SHA。
CPU 预检从对应 Git object 建立独立且内容校验的源码快照，位于 `code/RSmol/log/mizar_core4_eval_sources/`。
远程必须保有该提交；浅克隆缺少该 object 时需补齐对应 Git 历史。`--source-revision` 可指定内容完全匹配的其他 revision，不放宽 SHA。

GPU 进程以原官方 final release 严格构建完整 CED/projector/30层 LM，再严格覆盖 454 项模型状态。
检查名称/shape/dtype/finite/tied embedding 与全部冻结 tensor 逐值及 digest；保留训练后的四个核心层值，随后全模型 FP32/eval/frozen。
原训练模型直接执行 `prefix → core×4 → suffix`，30 物理层、42 次 block、无额外注入/门控/新增参数；每个完整 forward 审计真实轨迹，无跨题持久状态。

## 保持的评测协议

参考 `ced_eval_20261125` 和 `mizar_core3_eval`，旧训练、旧评测、baseline 实现均不改。

| benchmark | 完整题数 | 题面与评分 |
| --- | --- | --- |
| MMAU test-mini | 1000 | 原固定选项顺序、小写题面；原始生成直接交原官方 scorer |
| MMAR | 1000 | 原固定选项顺序、小写题面；原始生成直接交原官方 scorer |
| MMSU | 5000 | 当前紧凑小写四槽题面、null 槽保留；`mmsu_answer_text_contains_v2` 后原官方汇总 |

保持 FP32、batch1、seed1234、无 autocast、TF32 off；SoundFile/mono/32k、fixed20 中心裁剪/右补零、10+10 拼回完整 CED。
每题仅编码音频一次，随后复用 prefix。原 EOS 后缀、prompt256 右截断/补 PAD `!`=17、SEP/EOS=0、126+1+256=383，首答案 logit382。
因果 attention、PAD 可见、`attention_mask=None`；无 KV，每 token 完整重算、argmax、最多300、首 EOS 停止，保留 raw decode。
MMAU/MMAR 不加字母解析，不加候选 likelihood；MMSU 不添加仅输出字母的指令。

每项资源保持 **pdgpu-4090 / 1 GPU / 8 CPU / 32G / 1 node**，镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`，环境 `mellow_c8204d8`。
L9 训练用5090、另外两条训练用3090，不影响此次统一评测队列。无额外 GPU smoke。

## GitHub 同步与正式提交

本地精确文件清单见 `sync_files.txt`；已有共享依赖见 `reference_dependencies.txt`，需随仓库已同步。
在 Windows 仓库根，完成代码审阅后可执行（不会由助手自动执行）：

```powershell
Get-Content code/RSmol/mizar_core4_eval/sync_files.txt | ForEach-Object { git add -- $_ }
git diff --cached --check
git commit -m "Add nine evaluations for final Mizar AudioMCQ x4 checkpoints"
git push
```

确认暂存内容仅包含自己准备提交的工作。远程仓库根预览九项（只打印，不读权重、不提交）：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_core4_eval_4090.sh all --check-args
```

一次提交全部九项：

```bash
bash code/RSmol/run_mizar_core4_eval_4090.sh all
```

或者分三个模型提交，每条三项；与上面 `all` **择一**，避免重复任务：

```bash
bash code/RSmol/run_mizar_core4_eval_4090.sh l9_12x4
bash code/RSmol/run_mizar_core4_eval_4090.sh l14_17x4
bash code/RSmol/run_mizar_core4_eval_4090.sh l19_22x4
```

可先独立 CPU 预检全部三个模型，不申请 GPU：

```bash
bash code/RSmol/run_mizar_core4_eval_4090.sh all --preflight-only
```

单项提交/失败后续评示例：

```bash
bash code/RSmol/run_mizar_core4_eval_4090.sh l14_17x4 mmau
bash code/RSmol/run_mizar_core4_eval_4090.sh l14_17x4 mmar
bash code/RSmol/run_mizar_core4_eval_4090.sh l14_17x4 mmsu
```

入口每模型一次 CPU admission。预检失败，该模型三项均不调用 vc，其余模型继续；单次 vc 失败也继续后续任务，最终整体非零并保留报告。
CPU READY_FOR_GPU 仅表示 checkpoint header/runtime/source 通过；正式 tensor、资产和 CUDA 核验由各 GPU 进程完成，不能等同评测 PASS。

默认路径已经精确绑定；迁移副本时可对单个别名显式覆盖 `--checkpoint`、`--runtime-config`，仍要求相同 step16890 和模型合同。
非标准目录的副本同时传 `--output-root`；`all` 不允许单 checkpoint/runtime 覆盖。
`RSMOL_CONDA_BASE`/`MELLOW_CONDA_BASE` 可覆盖原 conda base。

## 输出、续评与比较

每项默认输出：`<run>/eval/mizar_core4_audiomcq_eval_20261127_v1/<task>/model--step-16890/`，
task 为 `mmau`、`mmar`、`mmsu_compact_lowercase_v1`。三模型位于各自原 run 下，互不覆盖。
日志：`code/RSmol/log/mizar_core4_eval/mcore4-<model>-<task>-<tag>.JOB.log`；同目录有 `<timestamp>.submission.json`。
submission 报告记录完整 checkpoint、输出、每模型预检和各 vc 返回内容；SUBMITTED 只代表调度提交成功。

`run_config.json` 固定 checkpoint/runtime/源码/数据/scorer/协议身份。
`raw_generations.jsonl` 逐题落盘，记录实际 prompt token/mask 与两段音频 SHA。
相同身份重跑只补剩余题；损坏的完整行、身份变化、缺题/重复题不会静默略过。完整覆盖后才评分；完整 MMSU 重评分不再加载模型，MMAU/MMAR 保留原流程的模型加载核验。
成功须检查 `evaluation_report.json` 的 PASS、完整1000/1000/5000分母、官方 scorer PASS 及 `model_audit.json` 的加载/执行审计。

`compare` 只读核验完整报告、原始预测、实际逐题输入、共有代码/数据/协议后输出 CSV/JSON。
允许比较此次三个 ×4 结果与已有 `mizar_core3_eval` 的 **official_mizar baseline**；分别核对原生42/30执行轨迹，比较共同推理设置。
不给旧 ×3 训练结果注册新 ×4 身份，不自动查找或重跑 baseline。

```bash
# 将两个变量设为同一 benchmark 已完成结果的绝对目录
BASELINE_REPORT=/absolute/path/to/existing/official_mizar/mmau/report_directory
X4_REPORT=/absolute/path/to/current/l9_12x4/mmau/model--step-16890
bash code/RSmol/mizar_core4_eval/run.sh compare --benchmark mmau \
  --report-dir "$BASELINE_REPORT" --report-dir "$X4_REPORT" \
  --output /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_core4_mmau_comparison.csv
```

每个 benchmark 单独比较。MMAU/MMAR 分数引用原 scorer 的完成记录；MMSU 额外按 raw 重算 answer-text 命中率。

## 本地验证边界

`tests/validate_mizar_core4_eval.py` 是离线 CPU 验收入口，报告 `audit_reports/mizar_core4_eval_validation.json`。
真实小30层 Llama eager/SDPA 验证三路线42执行、独立递推、严格模型重载、训练683/推理383及第二答案对齐、因果性和无跨题状态；真实 SoundFile 测试覆盖16k双声道解码/32k重采样与每题一次编码。
完整1000/1000/5000 runner 用明确合成模型/音频/parquet夹具，检查 raw、原 scorer、完整分母、13题中断续评及坏数据拒绝。
调度测试只模拟 vc，Bash 入口与 Git object 读取真实执行。
本地没有生产 checkpoint/数据/CUDA，远程权重、音频、4090显存/速度及正式分数尚未验证。
