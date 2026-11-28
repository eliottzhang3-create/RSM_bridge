# S3 projector＋全 LM 半轮微调：MMAU／MMAR

仅评测用户指定的 **step1066**：MMAU test-mini完整1000题、MMAR完整1000题，共两项。
新入口 `run_mizar_s3_full_finetune_eval_4090.sh`；原训练、数据准备和其它评测文件保持。
本地Windows编辑，通过GitHub同步；正式checkpoint、音频和结果留在远程Linux，不自动commit/push或提交GPU。

## 指定模型与加载

`target.json` 固定完整路径，无需再传长参数，不扫描latest：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_final_s3_full_finetune_halfepoch_gbs128_3090/formal_20261128_194549346557872/formal/checkpoints/model--step-1066.ckpt
```

自动使用同run `formal/runtime.json`。核对schema/formal、路线 `mizar_final_s3_full_projector_lm_halfepoch_gbs128_v1`、合同SHA、训练源码SHA、完整数据选择身份与八来源计数。
272796行池、136398条半轮预算、8×4×4、1066步、`num_epochs=.5`、最终游标`(0,1066)`；最新max LR5e-6/min1e-6/5% warmup的原生semantic_contract逐项核验。
只恢复模型状态，不创建或恢复optimizer/scheduler/训练RNG，不套用训练“已完成不能resume”的门槛，不重新prepare训练manifest。

原生30个独立LM层各执行一次，无循环/LoRA/gate/bias/新增参数，454项状态。
原官方final release及构造资产逐文件核验后构建模型，strict覆盖此次checkpoint中完整projector和全LM，包括embedding、final norm、tied head。
检查全部名称/shape/dtype/finite/tied值、恢复后逐tensor等值，冻结部分按本路线parameter_role核验官方值与digest。
**CED冻结；projector和全LM均允许为训练后值。** 覆盖后全模型FP32/eval/frozen，每个解码forward审计原生30层顺序及final norm一次。

`source_manifest.json` 固定Git `f826bf9024138addaf5d2e3f8902b713d273fac2`，训练code SHA `5059578543c1ea005bf6b0c8daf5adada696a1f560bd055a13dade60fc22767a`。
快照完整保留repo相对目录：训练包、release manifest、外层训练入口、S3数据准备Python和外层prepare脚本，共39份code_digest依赖。
从匹配Git object读取并逐文件核验到 `code/RSmol/log/mizar_s3_full_finetune_eval_sources/<SHA>/`，检查transitive import实际路径与完整原code_digest。
远程需保有对应Git历史；浅克隆缺object时补齐历史。可用 `--source-revision` 指定内容完全匹配版本，SHA检查保持。

## 与此前模型保持一致的评测协议

S3训练输入与此次统一评测输入分开记录，不能直接用S3训练数据类替换此前评测：

| 项目 | S3训练合同 | 此次MMAU/MMAR评测 |
| --- | --- | --- |
| 长音频 | resample32k→mono→first20 | mono→resample32k→center20，沿用原评测 |
| 短音频 | 右补零20s | 右补零20s |
| prompt | 固定256，无EOS | 原小写题面＋空格EOS后缀，右截断/补PAD到256 |
| 答案 | 小写＋EOS，固定96 | 原greedy最多300，首EOS停止 |
| 布局 | prefix383，总479 | prefix383，逐token完整重算 |

`eval_audio.py`、`audio_io.py`、`eval_text.py` 保留此前统一评测的数值函数；不调用S3的first20/no-prompt-EOS预处理。
MMAU/MMAR原固定选项顺序，不添加仅字母指令，不使用候选likelihood；raw生成不改写，交原官方scorer。
FP32、eval、batch1、seed1234、无autocast、TF32 off；32k单声道20s、两个10s拼回whole CED，每题只编码音频一次。
PAD `!`=17、SEP/EOS=0、audio126+SEP1+prompt256=383、首答案logit382；原因果attention、PAD可见、attention_mask=None，无KV，argmax/raw decode保持。
各项 **pdgpu-4090／1GPU／8CPU／32G／1node**；原镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`、`mellow_c8204d8`、Transformers4.54.1，无额外GPU smoke。
训练3090队列不影响统一评测4090。保持此前baseline比较条件，此次不重跑baseline。

## 同步与提交

`sync_files.txt` 列本次交付，`reference_dependencies.txt` 列需在仓库已有的共享依赖与训练源码。
本地PowerShell仓库根，审阅后自行提交（助手不自动执行）：

```powershell
Get-Content code/RSmol/mizar_s3_full_finetune_eval/sync_files.txt | ForEach-Object { git add -- $_ }
git diff --cached --check
git diff --cached --stat
git commit -m "Evaluate S3 half-epoch Mizar checkpoint on MMAU and MMAR"
git push
```

远程Linux仓库根同步后，分别提交：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# MMAU test-mini，完整1000题
bash code/RSmol/run_mizar_s3_full_finetune_eval_4090.sh mmau

# MMAR，完整1000题
bash code/RSmol/run_mizar_s3_full_finetune_eval_4090.sh mmar
```

参数预览和CPU预检：

```bash
bash code/RSmol/run_mizar_s3_full_finetune_eval_4090.sh all --check-args
bash code/RSmol/run_mizar_s3_full_finetune_eval_4090.sh all --preflight-only
```

`all`可一次提交两项，与两条单项命令择一。每次调用只做一次CPU预检，失败不调用vc；all内一项vc失败仍继续另一项，整体返回非零并保留报告。
CPU READY_FOR_GPU只确认header/runtime/source；GPU任务再验证生产资产、完整tensor及CUDA。SUBMITTED只表示提交成功，正式PASS以结果报告为准。
保留 `RSMOL_CONDA_BASE`／`MELLOW_CONDA_BASE` 覆盖。副本可显式 `--checkpoint`、`--runtime-config`、`--output-root`，仍只接受step1066及本合同；非标准目录必须同时指定runtime和output。

## 输出、续评与日志

默认在指定run下：

```text
formal_20261128_194549346557872/eval/mizar_s3_full_finetune_eval_20261128_v1/
  mmau/model--step-1066/
  mmar/model--step-1066/
```

保留 `run_config.json`、`raw_generations.jsonl`、`progress.jsonl`、`model_audit.json`、`evaluation_report.json`、原官方预测与评分日志。
run_config固定checkpoint/runtime/源码/数据/scorer/协议；raw逐题保存实际prompt token/mask、两段音频SHA。
同单项命令重跑只补剩余题；完整坏行、冲突progress、错身份、缺题/重复拒绝，未完成末行恢复；完整1000且无skipped才评分。
延续原MMAU/MMAR流程，重评分仍加载模型核验；evaluation_report须PASS、完整1000分母且官方scorer成功。
日志：`code/RSmol/log/mizar_s3_full_finetune_eval/ms3full-<mmau或mmar>-<tag>.JOB.log`；同目录timestamp `.submission.json`记录确切输出路径、CPU预检和vc返回。

## 本地验证边界

入口 `tests/validate_mizar_s3_full_finetune_eval.py`，报告 `audit_reports/mizar_s3_full_finetune_eval_validation.json`。
真实小30层Llama eager/SDPA、原S3 Mellow/Decoder方法验证原生30层/final norm一次、无跨forward状态；相同实际token输入下479训练与383/384推理首/第二答案logits/argmax及未来因果性。
严格加载训练后mapper/全LM/embedding/norm、冻结CED与错state拒绝；真实loader控制流中生产资产/CUDA为明确CPU夹具。
真实本地release tokenizer验证原EOS/右截断/补PAD与S3 no-EOS的区分；真实24s双声道16k音频验证mono32k中心裁剪、区别于S3 first20、路径/波形两分支相同输入指纹、每题一次编码。
完整1000题AU/AR runner用明确模型/音频/parquet合成夹具，原始预测/分母/13题中断后补987/坏行拒绝；真实Git object/外层依赖篡改拒绝、真实Bash入口与空格参数，vc仅模拟。
本地没有远程生产checkpoint/音频/CUDA；本地验证不代表生产454tensor加载、4090显存速度或正式分数通过。
