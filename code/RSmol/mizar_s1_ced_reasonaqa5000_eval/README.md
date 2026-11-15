# ReasonAQA test：固定随机 5000 条、其余 30 个 checkpoint

本地 Windows 准备代码，经 GitHub 同步；数据生成、权重校验和评测在远程 Linux 使用现有环境完成。新增独立包和入口，只复用已有严格模型 loader、greedy decoder 与逐题注入 context。原完整 ReasonAQA 七项以及 MMAU/MMAR/MMSU 的实现和默认输出均不修改。

## 数据与计分

源文件固定为 `/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test.json`，按原文件顺序筛选：

| 类别 | 条件 | 已确认候选池条数 |
| --- | --- | ---: |
| `audiocaps_mcq` | subtype 为 `AudioCaps-MCQ.json` | 23672 |
| `clotho_mcq` | subtype 为 `Clotho-MCQ.json` | 26031 |
| `clotho_aqa_yesno` | subtype 为 `ClothoAQA.json`，answer 去首尾空白、casefold 后等于 yes/no | 5676 |
| 合计 | | 55379 |

生成器先核对以上计数，再用独立的 `random.Random(1234).sample(range(55379), 5000)` 全池均匀无放回抽样，按原始行号排序后保存。**不分层、不按比例固定各类条数、不去重原题**；每条 ID 为 `reasonaqa_test:<原文件0起始行号>`。原始重复题若处于不同位置，仍是不同候选行。所有目标读取同一个 manifest，评测不会重新抽样。

默认采样目录：`/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test_close_ended_sample5000_seed1234_v1/`。

| 文件 | 用途 |
| --- | --- |
| `manifest.json` | 权威清单；保留 5000 条原始记录、源行号、ID、抽样协议、源 SHA256 和清单 SHA256 |
| `sampled_test.json` | 5000 条原格式 JSON，供人工检查 |
| `selected_records.jsonl` | 同一批记录及其行号、类别 |
| `dataset_audit.json` | 全池与抽样类别计数、排除计数、哈希 |
| `run_config.json` | 采样输出目录归属 |

同源、同目录重复生成幂等；数据或身份变动拒绝覆盖。评测和提交时从原 test.json 复核 SHA、抽样算法及全部入选原始记录。源文件搬家可传 `--test-json`，内容 SHA 必须一致。

唯一指标为**完整答案规范化精确匹配**：去首尾空白、移除末尾一个 `<|endoftext|>`、Unicode casefold、合并连续空白。保留选项字母、括号、正文和标点。例如 GT 为 `B) A person snoring`，仅输出 `B` 或 `A person snoring` 均算错；GT 为 `no` 时 `no.` 算错。空回答保留在分母内。只在全部 5000 题生成完毕后计算 overall 微平均和三类 correct/total/accuracy，accuracy 为 0–1 比例；原始生成和 token IDs 完整保存。

## 30 个固定目标

范围为当前 MMAU 已适配的 38 路减去 `official_mizar` 和已经评测的七项。`official_input_only`、`official_serial84_r8` 是独立的已训练 official-final 路线，仍在本次范围内。以下表格由 `targets.json` 生成；完整 variant、训练实现、run、step 和 geometry 都固定在该文件中，不自动选择 latest。

run 表中目录统一位于 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/`；checkpoint 为各 run 下 `formal/checkpoints/model--step-<step>.ckpt`，配套配置为同 run 的 `formal/runtime.json`。step=34626 的三项训练 geometry=8,4,4，其余为 8,8,4；推理始终单卡 batch1。

| 别名 | run 相对目录 | step |
| --- | --- | ---: |
| `init0904_serial84_r8` | `mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_gbs256_5090/formal_20261113_221451337688842` | 17313 |
| `init0904` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_init0904_5090/formal_20261111_130037935014521` | 17313 |
| `text_epoch1` | `mizar_s1_ced_small_text_epoch1_gbs256_5090/formal_20261112_103104688616376` | 17313 |
| `text_epoch1_serial48_r8` | `mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_gbs256_5090/formal_20261112_103717262469368` | 17313 |
| `text_lora_epoch1_ced48_r8` | `mizar_s1_ced48_text_epoch1_loop_lora_r8_5090/formal_20261112_111357380584730` | 17313 |
| `r515_inject84` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_5090/formal_20261030_000807991327585` | 17313 |
| `r515_inject48` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer4_8_inject_5090/formal_20261029_210116592758429` | 17313 |
| `r515_reinject` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_audio_reinject_5090/formal_20261029_110929631468450` | 17313 |
| `ca4812` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer4812_crossattn_audio_reinject_5090/formal_20261109_005039335295104` | 17313 |
| `official_input_only` | `mizar_official_final_s1_gbs128_5_20x2_5_input_only_5090/formal_20261107_220147075105401` | 34626 |
| `mesh4` | `mizar_s1_ced_small_gbs256_recursive_mesh_4slot_3090/formal_20261107_203129899582015` | 17313 |
| `mesh6` | `mizar_s1_ced_small_gbs256_recursive_mesh_6slot_3090/formal_20261106_092419698981503` | 17313 |
| `mesh7v2` | `mizar_s1_ced_small_gbs256_recursive_mesh_7slot_v2_3090/formal_20261107_215615398982617` | 17313 |
| `serial84_r4` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r4_5090/formal_20261107_215916042121396` | 17313 |
| `serial84_attn_shared_mlp_modality_r8` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_modality_lora_r8_5090/formal_20261107_215838489409042` | 17313 |
| `serial84_attn_shared_mlp_shared_modality_r8` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_shared_modality_lora_r8_5090/formal_20261107_215949376143529` | 17313 |
| `official_serial84_r8` | `mizar_official_final_s1_gbs128_5_20x2_5_5090/formal_20261107_003434623913704` | 34626 |
| `ca36912` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer36912_crossattn_5090/formal_20261030_140459065784533` | 17313 |
| `query32_ca36912` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer36912_query32_crossattn_5090/formal_20261030_162028139130855` | 17313 |
| `mesh_inject48`（候选） | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_5090/formal_20261030_172359752928500` | 17313 |
| `mesh_listen48` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_listen_5090/formal_20261030_172450546025072` | 17313 |
| `mesh_listen36912` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer36912_listen_5090/formal_20261101_113133327944502` | 17313 |
| `mesh_audio7_split3` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_5090/formal_20261101_170211458358358` | 17313 |
| `mesh_audio7_split3_inject` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_inject_5090/formal_20261102_002211827036990` | 17313 |
| `mesh_inject84` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer8_4_inject_5090/formal_20261102_112040402999384` | 17313 |
| `mesh_audio7_prompt_mix` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_prompt_mix_5090/formal_20261102_132232677352730` | 17313 |
| `mesh_x4` | `mizar_s1_ced_small_gbs128_recursive_mesh_5_10x4_5_5090/formal_20261103_093224843011222` | 34626 |
| `serial84_r16` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r16_5090/formal_20261105_173212933623545` | 17313 |
| `textmesh_serial84_r16` | `mizar_s1_ced_small_gbs256_recursive_textmesh_layer8_4_inject_audio_reinject_loop_lora_r16_5090/formal_20261105_154241055340068` | 17313 |
| `textmesh_modality_r8` | `mizar_s1_ced_small_gbs256_recursive_textmesh_layer8_4_inject_audio_reinject_loop_modality_lora_r8_5090/formal_20261105_173710754949913` | 17313 |

排除的八个完整 variant：

```text
official_mizar
base
recursive_5_10_5
recursive_mesh
recursive_5_10_5_layer8_4_inject_audio_reinject
recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8
recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8
recursive_mesh_layer4_8_inject_loop_lora_r8
```

`mesh_inject48` 对应 `recursive_mesh_layer4_8_inject`，使用用户暂定的路径：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_5090/formal_20261030_172359752928500/formal/checkpoints/model--step-17313.ckpt
```

目前尚未在远程核实该文件。提交器在提交该项前以 CPU 读取 checkpoint，调用已有 route validator 核对配套 runtime、训练源码、结构合同、17313 步和 8,8,4 geometry。失败会在提交报告记录该项 FAILED，并继续其他目标；只要有提交失败，批量入口最终返回非零。CPU 通过只标记 `PASS_CPU_CONTRACT_ONLY`，不能代替计算节点上的完整模型 strict load 和 GPU 推理验证。

## 远程数据生成与评测提交

本地将本次文件提交、push 后，在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/generate_reasonaqa_test5000.sh
bash code/RSmol/run_reasonaqa5000_eval_5090.sh all --check-args
bash code/RSmol/run_reasonaqa5000_eval_5090.sh all
```

第一条数据命令在登录节点 CPU 生成采样文件，打印实际 5000 条的类别计数和 manifest SHA。`--check-args` 只解析所有绑定，不读数据/权重、不提交。最后一条用 `vc submit` 提交 30 个独立目标；候选结构校验失败时其余 29 项继续提交。资源为 pdgpu-4090 / 1GPU / 8CPU / 32G / 1node、原镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`、conda `mellow_c8204d8`，不增加自动 smoke 作业。为兼容已有命令，入口文件名仍保留 `_5090.sh`，实际提交分区已改为 `pdgpu-4090`。

查看全部绑定或仅提交一个目标，支持表中别名与完整 variant：

```bash
bash code/RSmol/run_reasonaqa5000_eval_5090.sh --list-targets
bash code/RSmol/run_reasonaqa5000_eval_5090.sh init0904
```

单项与 all 按需使用，避免同时向同一个输出目录重复提交。重新提交同一项会在身份一致时续跑；不要为恢复某个失败项重复提交整个 all。

候选 checkpoint 也可先独立执行 CPU 校验：

```bash
bash code/RSmol/mizar_s1_ced_reasonaqa5000_eval/run.sh inspect_checkpoint --target mesh_inject48
```

若用户确认候选 run 有误，批量时可显式修正该项；以下占位路径需替换为真实 run：

```bash
bash code/RSmol/run_reasonaqa5000_eval_5090.sh all \
  --mesh-inject48-run-root '/实际确认的训练run目录'
```

或单独补交候选项：

```bash
bash code/RSmol/run_reasonaqa5000_eval_5090.sh mesh_inject48 \
  --run-root '/实际确认的训练run目录'
```

单候选还接受 `--checkpoint`、`--runtime-config`；canonical `formal/checkpoints/` 路径可自动推导 run。非 canonical checkpoint 须同时提供 runtime-config 和 output-dir。其他 29 项拒绝模型身份覆盖。

通用数据参数为 `--manifest`、`--test-json`、`--audiocaps-root`、`--clotho-root`、`--clotho-aqa-root`；只解析已选 5000 行的音频路径。默认三个音频根目录分别为 `data/audiocaps_v2`、`data/clotho_v2_1`、`data/clotho_aqa_audio/audio_files`，均位于远程 home 下。支持原逻辑反斜杠/空格路径；缺失、歧义和越界路径失败，不跳题。

## 推理、输出与恢复

复用当前 MMAU/MMAR 的原 CED 推理设置：原 input 小写后训练 tokenizer 固定 256 tokens，126 audio + 1 SEP + 256 prompt = 383 prefix，首答案 logit 位于 382；FP32、eval/frozen、seed1234、TF32 关闭、batch1、greedy argmax 300 tokens、遇 EOS 停止、无 KV cache、完整重算、LM attention_mask=None。原 route 所需 prompt mask 和每样本注入 context 正常传入。caption、GT 和任务元数据不进入模型。

音频调用原训练 mono/resample32k/fixed_len：单文件居中裁剪或右补零到 20 秒，再分两个 10 秒；filepath2 非空时两文件各独立固定 10 秒。checkpoint/runtime/模型源码/训练合同、初始化资产和 tokenizer 沿用原严格加载核验，不重置训练后的 gate、router 或 LoRA。

每个目标输出到自己的 run 下：`eval/reasonaqa_close_ended_sample5000_seed1234_step<step>_v1/`。

| 文件 | 内容 |
| --- | --- |
| `run_config.json` | 模型/数据/代码/抽样和推理协议身份 |
| `dataset_audit.json` / `selected_records.jsonl` | 抽样审计、5000 条原记录和真实音频路径 |
| `raw_generations.jsonl` | 原始输出、token IDs、prompt/mask/音频输入 SHA |
| `progress.jsonl` / `model_audit.json` | 可恢复进度、原严格加载与执行审计 |
| `accuracy.json` / `per_sample_results.jsonl` | 唯一指标及逐题得分 |
| `evaluation_report.json` | PASS/FAILED、覆盖数量和异常堆栈 |

raw 逐题 append+fsync，进程锁拒绝并发写；中断后只生成剩余题目。仅恢复最后一次不完整的无换行写入，完整坏行/错序/身份不一致均拒绝。完整 5000 条 raw 重评分不加载模型。源码、模型、配置或数据身份改变则拒绝复用旧输出；单目标可传独立 `--output-dir`，all 不允许共用输出目录。

提交器逐项保存 `code/RSmol/log/reasonaqa5000.<时间_PID>.submission.json`，区分 SUBMITTED/FAILED、vc 调用是否发生和候选校验结果；也可传 `--submission-report`。SUBMITTED 仅表示提交成功。GPU 日志为同目录 `reasonaqa5000.<时间_PID>.<别名>.JOB.log`。

`-j` 使用 `raqa5k-<最多40字符的别名>-<12位摘要>`，完整名称最多 60 个 ASCII 字符。摘要由完整批次时间/PID 标记及完整别名生成，长别名前缀相同时仍可区分，日志保留完整批次标记和别名。

30 项全部评测结束后，CPU 汇总命令如下：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/mizar_s1_ced_reasonaqa5000_eval/run.sh compare
```

默认生成 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/reasonaqa5000_remaining30_seed1234_v1.csv` 和同名 JSON。核对完整覆盖、清单/协议/代码一致、raw 哈希、重算分数和逐题 prompt/mask/音频输入指纹后才汇总。

若更正了候选 run 或自定义了输出目录，比较时须重复 `--report-dir '/真实评测输出目录'` 显式列出全部要比较的结果；默认比较仍读取固定注册表中的 30 个目录。缺失任何默认结果会报错，不生成残缺的默认 30 项汇总。

## 本地验证与代码同步

```powershell
python -B tests/validate_reasonaqa5000_eval.py
```

报告为 `audit_reports/reasonaqa5000_eval_validation.json`。包含真实 55379 行合成候选池、5000 条完整流程、全答案评分、清单篡改拒绝、中断恢复、30 项真实参数解析、真实 Bash 配合模拟 conda/vc、原三种 benchmark 回归；以本次修改前 1070 份源码 SHA 验证原实现未变。模型生成在合成流程中被模拟，依赖缺失的真实 tensor/音频测试明确跳过；本地通过不表示生产 checkpoint 或远程 GPU 推理已经通过。

供用户执行的精确同步命令：

```powershell
git add -- code/RSmol/mizar_s1_ced_reasonaqa5000_eval code/RSmol/generate_reasonaqa_test5000.sh code/RSmol/run_reasonaqa5000_eval_5090.sh tests/test_reasonaqa5000_eval.py tests/test_reasonaqa5000_shell.py tests/validate_reasonaqa5000_eval.py audit_reports/reasonaqa5000_eval_validation.json audit_reports/reasonaqa5000_rm_history.json audit_reports/reasonaqa5000_documentation_validation.json
git add -p -- RM.md
git diff --cached --stat
git commit -m "Add fixed 5000-row ReasonAQA evaluation for remaining 30 targets"
git push
```

RM.md 分块仅选本次新增记录，保留工作区其他任务的修改；未代为 commit/push 或提交真实评测。
