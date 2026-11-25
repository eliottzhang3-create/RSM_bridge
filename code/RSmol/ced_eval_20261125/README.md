# 2026-11-25：audio8/read3和HRM PrefixLM，两个checkpoint的八项评测

本地Windows修改代码，通过GitHub同步，生产权重/数据及GPU作业在远程Linux。
本入口只绑定用户本次指定的两个checkpoint，不搜索latest、不提交其他历史模型。

| 别名 | 完整训练run（前缀 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/`） | 原生模型 |
|---|---|---|
| audio8_read3 | `mizar_s1_ced48_audio8_read3_lora_r8_gbs128_text71437_3090/formal_20261124_002101117114787` | 28物理/56执行，三循环MeSH、八槽read3、三套LoRA r8 |
| hrm_prefixlm | `mizar_s1_hrm_bp5_prefixlm_ced_small_3090/formal_20261124_184519991594964` | 20物理/50执行，H2L3/BP5、PrefixLM383 |

均固定 `/formal/checkpoints/model--step-34626.ckpt`，同run `/formal/runtime.json` 自动绑定；完整绝对路径在 `targets.json`。
每个模型四项：ReasonAQA5000、MMAU test-mini、MMAR、MMSU，共8项。
全部 `pdgpu-4090 / 1GPU / 8CPU / 32G / 1node`，原镜像、`mellow_c8204d8`环境。

## 推理与评分对齐

两模型共用原数据、原题序、原音频及分词流程、greedy解码规则和评分，实现复用已有正式评测器：

- ReasonAQA：原seed1234固定5000 manifest；完整答案exact。
- MMAU test-mini / MMAR：各1000题；原始生成直接交原官方scorer，不做新答案抽取或仅字母改写。
- MMSU：最新紧凑全小写四槽题面，例 `which country? a) india b) none c) china d) japan`。
  原顺序及null占位保持、无only-letter指令，GT不进入输入；原 `mmsu_answer_text_contains_v2` 和官方5000分母保持。
- FP32、eval、batch1、seed1234、无autocast、TF32关闭；音频mono32k/fixed20中心裁剪或右补零，然后拆10+10。
  Mellow在CED之前拼回原整段，whole mel/BN/正确时频顺序与训练一致。
- audio126+SEP1+prompt256=prefix383；首答案取logit382，原EOS后缀、PAD `!`、截断/右补齐保持。
- 无KV cache、每生成token全序列重算、argmax、最多300 token，首EOS停止；保留包含EOS的原始decode文本。
- 不跳过坏题、不用部分分母，完整覆盖才评分；同身份raw续跑、锁和坏行拒绝保持。

模型内部使用各自训练时的注意力机制，报告如实区分：

| 项目 | audio8_read3 | hrm_prefixlm |
|---|---|---|
| Attention | 原causal，`attention_mask=None` | 训练原生PrefixLM383，显式validity |
| Prompt padding key | 可见 | 屏蔽；query382保留计算 |
| Prefix范围 | 原因果顺序 | audio/SEP/prompt有效位置全双向，不能看任何answer |
| Answer范围 | causal | 有效prefix全可见＋当前/以前answer |
| 每次执行 | 56次/三循环MeSH | 50次/P5+(L5×3 H5)×2+C5，固定BP5 |
| 单题上下文 | 传入该题A4/A8至每次LM forward | 传入该题原始prompt IDs及tokenizer validity |

HRM直接调用固定训练源码 `DecoderModel.inference_logits(prefix,texts_enc,generated_ids)`，
使用训练同一 `prefixlm_mask`、BP5和位置编号；每步从初始prefix及全部已生成ID重算。
每完整forward从已保存的固定zL_init和该次Prelude输出重新开始，不跨题/跨token携带隐藏状态。
生成token（包含可能生成的 `!`）按原native接口作为有效历史token；prompt padding依原mask处理。
不会重新转换Norm、清零已训练gate、重建初始状态值或恢复optimizer/训练RNG/EMA。

audio8/read3完整保留八槽 `[E,0,0,0,0,0,A4,A8]` 与文本六槽，A4/A8只读；
四组write只更新前六槽；执行7/21/35次后前三audio read各读八槽，49次后最终audio read仍原六槽read3。
11router/22张量/41544参数，三轮各98共294投影、588独立LoRA矩阵/3419136参数，
两MLP2443776加三audio read13848，共新增5876760；每题只编码一次音频，context显式随每次forward传递。
真实 `validate_audio_read3(model)` 只传model，不传其不支持的fresh参数；训练后全部分支值保留。

## 固定源码、严格加载与审计

| 目标 | Git版本 | 训练code_sha256 |
|---|---|---|
| audio8_read3 | `2c7d4485979f30f23c96196533c6de5914c47888` | `3c3a4619d3ab1d019fd9d3fc288c2f4ebe81d8d6c3cf707b3af64d18627e7f2d` |
| hrm_prefixlm | `ee6f54474139fa5b86a9189617a2d2c0b9c84492` | `0814180a787ab29ab439403fff5befab2b870e4bb6b6cb50b07822a677c1ca58` |

通过本地Git object `git archive`提取必要训练后端/共享代码到
`code/RSmol/log/ced_eval_20261125_sources/<revision>/`，逐文件SHA、库存和训练/共享摘要固定核对。
HRM的code_digest还包含顶层训练Shell；共享摘要包含两个HRM包所有直接Python源码及原raw_audio审计器，快照一并完整保留。
只导入对应快照，核对模型/合同实际路径；不切换checkout，不更改已有训练目录，不修改共享环境。
当前评测器、紧凑MMSU dataset和原scorer保持当前checkout，不使用历史旧题面。

CPU每个checkpoint预检一次：正式header/route/step、完整runtime合同SHA、训练geometry8×4×4、源码/原生函数签名。
GPU再核对外部text/CED资产、完整state_dict库存/shape/FP32/finite、tied aliases、冻结CED；
HRM额外核对固定zL_init与来源一致，20独立层/102420288 LM参数、MagicNorm、已训练gate及原生音频PrefixLM类。
只加载完整音频checkpoint模型权重，文本source仅作构造/tokenizer/资产校验，不覆盖音频训练后值。

只读hooks检查每个生成forward的56次三轮LoRA/router轨迹，或HRM50次相同PrefixLM mask、padding/边界/无训练梯度。
hooks不改变forward或数值，mask参考仅在当前forward存活，报告只保存元数据。
`READY_FOR_GPU`仅表示CPU预检；`SUBMITTED`仅表示vc接受提交；完整结果以 `evaluation_report.json` 为准。

## 远程提交命令

本轮文件经GitHub同步后，在远程仓库根执行这8条：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# 三循环audio8/read3/LoRA r8
bash code/RSmol/run_ced_eval_20261125_4090.sh audio8_read3 reasonaqa5000
bash code/RSmol/run_ced_eval_20261125_4090.sh audio8_read3 mmau
bash code/RSmol/run_ced_eval_20261125_4090.sh audio8_read3 mmar
bash code/RSmol/run_ced_eval_20261125_4090.sh audio8_read3 mmsu

# HRM BP5 PrefixLM
bash code/RSmol/run_ced_eval_20261125_4090.sh hrm_prefixlm reasonaqa5000
bash code/RSmol/run_ced_eval_20261125_4090.sh hrm_prefixlm mmau
bash code/RSmol/run_ced_eval_20261125_4090.sh hrm_prefixlm mmar
bash code/RSmol/run_ced_eval_20261125_4090.sh hrm_prefixlm mmsu
```

也可一次提交8项，**与上面的8条择一，避免重复排队**：

```bash
bash code/RSmol/run_ced_eval_20261125_4090.sh all
```

按模型一次提交四项的等价替代：

```bash
bash code/RSmol/run_ced_eval_20261125_4090.sh audio8_read3
bash code/RSmol/run_ced_eval_20261125_4090.sh hrm_prefixlm
```

可选预览/仅CPU合同检查，不申请GPU：

```bash
bash code/RSmol/run_ced_eval_20261125_4090.sh all --check-args
bash code/RSmol/run_ced_eval_20261125_4090.sh all --preflight-only
```

一次all时每个checkpoint只预检一次。某目标预检失败则其四任务不申请GPU，其余目标继续；vc某次失败也继续其余任务。
整体任一失败退出非零，记录具体错误。不增加额外GPU smoke作业，不自动commit/push或提交真实作业。

## 日志、结果、续跑和比较

```bash
ls -lt code/RSmol/log/ced_eval_20261125/
```

- `<时间>_<pid>.submission.json`：两个目标预检、八项checkpoint/输出、完整vc参数和返回stdout/stderr。
- `ced25-<target>-<task缩写>-<hash>.JOB.log`：GPU日志，对实际文件运行 `tail -f`。
- 新结果：`<各自run>/eval/ced_eval_20261125_v1/<task>/model--step-34626/`。
  task为 `reasonaqa5000`、`mmau`、`mmar`、**`mmsu_compact_lowercase_v1`**。
- `run_config.json`记录数据/模型/源码/协议身份；`model_audit.json`记录真实架构/注意力/forward；
  `raw_generations.jsonl`保留逐题生成，`evaluation_report.json`记录完整分母和评分结果。

保留所有历史评测目录。新批次中断后确认原作业已退出，再执行相同单项命令；
模型/数据/scorer/代码/协议身份相同才续跑。完整输出可复用raw评分，不能强改旧run_config来复用其他协议结果。

本包compare可读取本批次及已注册的历史完成结果：严格核对同benchmark的manifest/题序/数据、分词输入、
原scorer、共同解码设置和共享数据/评分源码；原生attention/padding按注册策略分别验证，并在CSV中显式展示。
不会宣称HRM和MeSH的内部mask相同。MMSU旧官方题面、不同token上限、原始记录/评分矛盾或未完成覆盖均拒绝。

```bash
bash code/RSmol/ced_eval_20261125/run.sh compare --benchmark mmsu \
  --report-dir /absolute/completed/audio8_result \
  --report-dir /absolute/completed/hrm_result \
  --output /absolute/new/comparison.csv
```

## 本地验证与Git同步

使用既有本地依赖执行验证，不修改生产环境：

```powershell
.\.hrm_test_runtime\Scripts\python.exe -B tests/validate_ced_eval_20261125.py --dependency-dir .codex_audio_cutoff_deps --dependency-dir .hrm_test_runtime/audio_packages --require-tensors
```

报告 `audit_reports/ced_eval_20261125_validation.json` 保留各轮结果和依赖版本。真实CPU测试包括：
两个架构的生产576维审计、三轮LoRA/真实read3校验、非初始参数重载、HRM原生greedy及50次mask、
首/第二答案训练-推理logits、未来答案隔离、padding/query382、跨题无状态carry；
真实训练导入图、合成WAV解码/重采样，以及四种实际生成适配器每题一次encoder/context路径。
完整尺寸两组Reason5000/MMAU1000/MMAR1000/MMSU5000使用明确模型/音频/Parquet/CUDA夹具，bulk fsync模拟；
另验证真实Git快照、严格合同拒绝、加载控制流、真实Bash参数链/mock vc、续跑和完整原评分。
本地CPU不代表生产checkpoint/4090 CUDA/显存或实际分数已通过；本轮未读取远程生产资产。

精确同步清单见本包 `sync_files.txt`，新入口、整个评测包及当前共享Reason/MMSU/原scripts必须一起可用。
已有训练、共享模型和旧评测实现保持；模型权重和数据无需拷回本地。

工作期间仓库提交 `42f7fdf...` 更新了独立audio_cutoff1000实验的identity/submit两份文件；
本轮保留它们，前后摘要单列 `audit_reports/ced_eval_20261125_concurrent_sources.json`，
不将其记为本评测改动。任务开始的3130份已有生产源码中，其余3128份摘要逐一保持。
初次库存数量断言使用了全仓而非生产目录的预期值，修正夹具后继续；首轮联合库存检查捕获上述并行提交，
登记精确摘要后完整复验。各轮失败/通过证据均保留在验收报告中。

最终联合验收38/38通过、无跳过，包含6项真实CPU张量/模型用例和7项既有MMSU紧凑题面回归。
本地版本Python3.13.5/Torch2.9.0+cpu/Transformers4.54.1/Torchaudio2.9.0+cpu/SoundFile0.14.0；
与远程Torch2.11/CUDA不同，不能据此声称远程GPU结果已通过。
Python3.10语法、两入口及文档6块Bash语法、whitespace和32份同步清单检查通过。
