# 剩余27项 MMSU：pdgpu-4090 单卡入口

2026-11-22题面更新：通用MMSU已统一改为紧凑格式＋全小写，v3协议，评分仍为答案文本v2。
默认输出现在是`<run-root>/eval/mmsu_compact_lowercase_v1/step<step>/`，旧官方题面结果不会被视为新协议已完成或可续跑。
本页27/11目标划分是此前旧题面批次的历史绑定，本次未选择新的重评范围、未修改targets/excluded，也未提交作业。
后续重评范围以用户新指定为准，不能由这份历史排除清单推定。既有旧raw离线评分入口仍可用，但不会改变题面。

入口 `code/RSmol/run_mizar_mmsu_remaining_4090.sh` 固定覆盖当前38个已适配版本中尚未完成MMSU的27项，排除用户确认已完成的11项。每项独立提交一个作业，统一 `pdgpu-4090 / 1GPU / 8CPU / 32G / 1node`，原镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`、conda `mellow_c8204d8`、HF/Transformers离线设置。

本地Windows维护代码，经GitHub同步到远程Linux；数据、权重及正式推理仍在远程。本包只增加目标绑定、调度、既有结果检查和汇总，计算节点直接执行原 `mizar_s1_ced_mmsu_eval/run.sh`。训练、模型、既有MMSU/MMAU/MMAR和ReasonAQA源码均保持原样，新包不进入原MMSU源码身份集合。

## 提交与核对

代码经GitHub同步后，在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# 打印全部27项精确绑定及实际vc_args；不读权重/数据，不写评测或提交记录，不调用vc。
bash code/RSmol/run_mizar_mmsu_remaining_4090.sh all --check-args

# 每项一个4090单卡作业；遇到单项失败继续其他项，批次有失败则返回非零。
bash code/RSmol/run_mizar_mmsu_remaining_4090.sh all
```

`all`不包含已完成11项。支持下表别名或完整 `eval_variant`，例如：

```bash
bash code/RSmol/run_mizar_mmsu_remaining_4090.sh --list-targets
bash code/RSmol/run_mizar_mmsu_remaining_4090.sh ca36912
bash code/RSmol/run_mizar_mmsu_remaining_4090.sh recursive_mesh_6slot
```

job名 `mmsu27-<最多32位别名>-<12位摘要>` 完整最多52个ASCII字符，包含前缀、分隔符和摘要后缀，严格小于60。摘要仍绑定完整别名及批次时间/PID，缩短名称后保持区分；时间/PID不直接拼入job名。提交日志及逐项状态报告位于 `code/RSmol/log/mmsu27.<时间_PID>...`。`vc`接受提交仅代表入队，实际推理状态以各run产物为准。

## 目标绑定

完整机器可读映射见 [targets.json](targets.json)。沿用已登记的精确run、训练route、step和geometry。每项固定读取：

```text
<run-root>/formal/checkpoints/model--step-<expected_step>.ckpt
<run-root>/formal/runtime.json
```

不扫描checkpoint目录、不选latest，中间步checkpoint不干扰。下表run均位于 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/` 下；完整训练目录与variant见JSON或 `--list-targets`。

| 别名 | eval_variant | run相对路径 | step |
| --- | --- | --- | --- |
| `r515_inject84` | `recursive_5_10_5_layer8_4_inject` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_5090/formal_20261030_000807991327585` | 17313 |
| `r515_inject48` | `recursive_5_10_5_layer4_8_inject` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer4_8_inject_5090/formal_20261029_210116592758429` | 17313 |
| `r515_reinject` | `recursive_5_10_5_audio_reinject` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_audio_reinject_5090/formal_20261029_110929631468450` | 17313 |
| `ca36912` | `recursive_5_10_5_layer36912_crossattn` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer36912_crossattn_5090/formal_20261030_140459065784533` | 17313 |
| `query32_ca36912` | `recursive_5_10_5_layer36912_query32_crossattn` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer36912_query32_crossattn_5090/formal_20261030_162028139130855` | 17313 |
| `mesh_inject48` | `recursive_mesh_layer4_8_inject` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_5090/formal_20261030_172359752928500` | 17313 |
| `mesh_listen48` | `recursive_mesh_layer4_8_listen` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_listen_5090/formal_20261030_172450546025072` | 17313 |
| `mesh_listen36912` | `recursive_mesh_layer36912_listen` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer36912_listen_5090/formal_20261101_113133327944502` | 17313 |
| `mesh_audio7_split3` | `recursive_mesh_layer4_8_audio7slot_split3` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_5090/formal_20261101_170211458358358` | 17313 |
| `mesh_audio7_split3_inject` | `recursive_mesh_layer4_8_audio7slot_split3_inject` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_inject_5090/formal_20261102_002211827036990` | 17313 |
| `mesh_inject84` | `recursive_mesh_layer8_4_inject` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer8_4_inject_5090/formal_20261102_112040402999384` | 17313 |
| `mesh_audio7_prompt_mix` | `recursive_mesh_layer4_8_audio7slot_split3_prompt_mix` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_prompt_mix_5090/formal_20261102_132232677352730` | 17313 |
| `serial84` | `recursive_5_10_5_layer8_4_inject_audio_reinject` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_5090/formal_20261102_112121702891801` | 17313 |
| `serial48_lora_r8` | `recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8_5090/formal_20261104_131843931946322` | 17313 |
| `serial84_r16` | `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r16` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r16_5090/formal_20261105_173212933623545` | 17313 |
| `textmesh_serial84_r16` | `recursive_textmesh_layer8_4_inject_audio_reinject_loop_lora_r16` | `mizar_s1_ced_small_gbs256_recursive_textmesh_layer8_4_inject_audio_reinject_loop_lora_r16_5090/formal_20261105_154241055340068` | 17313 |
| `textmesh_modality_r8` | `recursive_textmesh_layer8_4_inject_audio_reinject_loop_modality_lora_r8` | `mizar_s1_ced_small_gbs256_recursive_textmesh_layer8_4_inject_audio_reinject_loop_modality_lora_r8_5090/formal_20261105_173710754949913` | 17313 |
| `mesh_x4` | `recursive_mesh_5_10x4_5` | `mizar_s1_ced_small_gbs128_recursive_mesh_5_10x4_5_5090/formal_20261103_093224843011222` | 34626 |
| `mesh4` | `recursive_mesh_4slot` | `mizar_s1_ced_small_gbs256_recursive_mesh_4slot_3090/formal_20261107_203129899582015` | 17313 |
| `mesh6` | `recursive_mesh_6slot` | `mizar_s1_ced_small_gbs256_recursive_mesh_6slot_3090/formal_20261106_092419698981503` | 17313 |
| `mesh7v2` | `recursive_mesh_7slot_v2` | `mizar_s1_ced_small_gbs256_recursive_mesh_7slot_v2_3090/formal_20261107_215615398982617` | 17313 |
| `serial84_r4` | `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r4` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r4_5090/formal_20261107_215916042121396` | 17313 |
| `serial84_attn_shared_mlp_modality_r8` | `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_modality_lora_r8` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_modality_lora_r8_5090/formal_20261107_215838489409042` | 17313 |
| `serial84_attn_shared_mlp_shared_modality_r8` | `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_shared_modality_lora_r8` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_shared_modality_lora_r8_5090/formal_20261107_215949376143529` | 17313 |
| `official_serial84_r8` | `official_final_s1_5_20x2_5` | `mizar_official_final_s1_gbs128_5_20x2_5_5090/formal_20261107_003434623913704` | 34626 |
| `ca4812` | `recursive_5_10_5_layer4812_crossattn_audio_reinject` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer4812_crossattn_audio_reinject_5090/formal_20261109_005039335295104` | 17313 |
| `official_input_only` | `official_final_s1_5_20x2_5_input_only` | `mizar_official_final_s1_gbs128_5_20x2_5_input_only_5090/formal_20261107_220147075105401` | 34626 |

`mesh_x4`、`official_serial84_r8`、`official_input_only`：step34626、训练geometry `8,4,4`。其余24项：step17313、训练geometry `8,8,4`。geometry是权重身份，评测始终单GPU、batch1。

`mesh_inject48`沿用既有登记的候选路径，提交前在CPU调用原路线checkpoint validator，核对route/源码/合同/step/geometry和配套runtime。失败只阻止该项，其他26项继续；CPU通过不代替计算节点的完整模型strict加载。如果远程合同检查说明该run有误，可明确更正该候选项：

```bash
# 单项更正：替换下方路径为已核实的run根目录。
bash code/RSmol/run_mizar_mmsu_remaining_4090.sh mesh_inject48 --run-root '/absolute/verified/run'
# 批次仅更正这一候选项。
bash code/RSmol/run_mizar_mmsu_remaining_4090.sh all --mesh-inject48-run-root '/absolute/verified/run'
```

只有该候选项允许模型路径更正；不接受其他目标的模型身份覆盖。非标准checkpoint布局须显式配套runtime和独立output；标准布局可从 `--checkpoint` 推导同run。所有目标仍须通过原strict loader。

已排除完整variant清单固定在 [excluded.json](excluded.json)：`official_mizar`、`base`、普通`recursive_5_10_5`、原始`recursive_mesh`、串行8→4 LoRA r8、MeSH4→8 LoRA r8、init0904、text_epoch1、text_epoch1串行4→8 LoRA r8、ced48继承文本LoRA、init0904串行8→4 LoRA r8。`serial84`与`serial48_lora_r8`只在其他benchmark已测，仍属于本次27项。

## 保持原有推理和评分协议

直接复用 [原MMSU实现](../mizar_s1_ced_mmsu_eval/README.md)，不新增生成循环或替换模型forward。

| 项目 | 本次统一设置 |
| --- | --- |
| 数据 | 原MMSU完整5000条，三shard按原顺序；不抽样、不打乱选项、不跳题 |
| 音频 | 原bytes解码/真实采样率；mono→32kHz，中心裁剪或右补零到20秒，再连续分为两个10秒 |
| 题面 | 原MMSU官方英文题面、原大小写/null选项；训练tokenize/EOS规则、右补`!`到256tokens |
| Prefix | 126音频＋1 SEP＋256prompt＝383，首答案logit在382 |
| 数值 | FP32、batch1、seed1234、eval/frozen、inference mode、关闭TF32/无autocast |
| 解码 | attention_mask=None、无KV cache、逐token完整forward、argmax、最多300新tokens、首次EOS停止 |
| 架构 | 原strict loader、原逐样本context；保留中间层注入、cross attention/query32、MeSH memory/router和LoRA分支 |
| 输出 | 原始大小写、空白、EOS和token IDs原样保存 |
| 评分 | 继续 `mmsu_answer_text_contains_v2`：忽略生成字母，完整GT答案文本包含匹配，只转换评分副本；原固定官方scorer、完整5000分母 |

评分不改回单字母准确率，不改为ReasonAQA精确匹配。只有完整5000条唯一记录才能评分。默认资产仍为 `/hpc_stor03/sjtu_home/jinwei.zhang/data/MMSU/{MMSU data,MMSU code/mmsu_evaluation.py}`。`--mmsu-root`、`--dataset-dir`、`--evaluation-script`可指定相同资产的新位置；`--parquet-batch-size`仅改变I/O读取批量，模型仍batch1。单项可指定 `--output-dir`，批次禁止共享output。

## 既有产物、续跑与重复提交保护

默认结果沿用各训练run的 `eval/mmsu_step17313/` 或 `eval/mmsu_step34626/`。保留原raw、progress、model_audit、run_config、预测与评分文件。调度前只读检查：

- `NEW/RESUME`：使用原runner生成或同身份补齐剩余题；partial必须与当前原runner身份完全一致。只读检查不截尾，真正恢复由原runner负责。
- `COMPLETE`：核对模型/数据/源码身份、完整5000条raw/预测及v2分数，跳过GPU提交。
- `NEEDS_RESCORE`：完整生成有效但评分缺失或仍是旧v1时，直接CPU调用原offline rescore，写独立 `rescore_mmsu_answer_text_v2/`，不重新推理。
- `BUSY`：原输出锁被占用时跳过；不改已有生成。
- 身份冲突、坏完整行、错step、错误覆盖：该项失败并保留源文件，继续其他项。

新提交记录位于 `code/RSmol/log/mmsu_remaining4090_admissions/`。相同绑定在该记录中为 `SUBMITTING/SUBMITTED` 时再次调用不会重复入队，即使排队作业尚未创建输出。该记录不查询集群状态，也不会自动判断排队作业已结束。完整结果优先复核；作业停止但结果未完成时，需要单项显式重提：

```bash
# 先确认旧任务已取消或结束，再显式重提同一目标。
bash code/RSmol/run_mizar_mmsu_remaining_4090.sh ca36912 --retry-submitted
```

`--retry-submitted`仅允许单项，不绕过模型/数据/输出身份或忙锁检查。崩溃残留的SUBMITTING同样按这个方式处理。`--ledger-dir`用于迁移记录位置，正常重复调用应保持相同目录。只有本新入口的提交会登记；使用旧入口提交的尚未产生输出的排队作业不在新记录内。

只查看现有结果与提交记录：

```bash
bash code/RSmol/mizar_s1_ced_mmsu_remaining_eval/run.sh inspect all
```

inspect不调用vc、不重评分、不加载模型或修改评测源文件，只写本次独立状态报告；已有产物的核验会读取权重/runtime/数据文件SHA。

## 汇总

```bash
bash code/RSmol/mizar_s1_ced_mmsu_remaining_eval/run.sh compare
```

默认输出 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mmsu_remaining27_4090_v2.csv` 和同名JSON，含27项目标/variant/checkpoint/状态/正确数/总数/accuracy/评分目录。只为完整验证结果填分数，未完成项留空；全部27项完整可比才返回0，否则INCOMPLETE并返回1。核对同三shard、题目顺序、生成协议、v2评分规则及共有评测代码SHA；不输出部分题分母或跨模型平均。支持 `--output '/absolute/summary.csv'`；单项比较可在compare后加目标别名。

## 本地验证与同步

```powershell
python -B tests/validate_mizar_mmsu_remaining_eval.py
```

报告 `audit_reports/mizar_mmsu_remaining_eval_validation.json`。包含27项绑定与排除、原计算入口/资源/短job名、完整5000行原始生成检查与CPU重评分、只读检查/中断恢复/锁/错误身份/失败继续/重复提交保护、真实Bash与模拟vc，以及原MMSU及既有架构回归。模型和GPU仅合成模拟；缺真实依赖明确skip。`reference_audit.json`保存1768份原源码SHA及RM历史摘要，验证源码内容与原RM历史保持。

2026-11-16本地验收：195项中181通过、14项因缺Torch/Transformers/音频/Parquet等真实依赖跳过，`PASS_LOCAL_WITH_SKIPS`。新增20项全部通过；1768份原源码raw/LF SHA均保持，Python3.10与两份Bash语法通过。Windows沙箱阻止临时目录/Bash时获准在沙箱外运行合成验证；没有读取远程权重或提交生产作业。

同日按用户要求将完整job名上限从60缩短至52，严格小于60。补充长时间/PID批次与别名截断冲突回归，本包两组测试22项全部通过；实际27项的完整`-j`最长52位且互异。该后续检查单列在验证报告的`job_name_followup`，不重写此前195项验收计数。

本次不代为commit/push或提交真实作业。本地同步文件清单：

```powershell
git add -- code/RSmol/mizar_s1_ced_mmsu_remaining_eval code/RSmol/run_mizar_mmsu_remaining_4090.sh tests/test_mizar_mmsu_remaining_eval.py tests/test_mizar_mmsu_remaining_shell.py tests/validate_mizar_mmsu_remaining_eval.py audit_reports/mizar_mmsu_remaining_eval_validation.json RM.md
git diff --cached --stat
git commit -m "Add remaining 27 MMSU targets on single 4090"
git push
```

用户完成上述同步后，再在远程pull并运行提交命令。
