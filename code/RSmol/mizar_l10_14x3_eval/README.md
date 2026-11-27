# L10–14×3 / A9 gate / bias3 / L15：AU、AR 评测

仅评测指定 **step2542** 的 **MMAU test-mini（AU）1000题** 和 **MMAR（AR）1000题**，共两项。
独立入口与命名空间，原训练、baseline、此前×3/×4及其它评测文件保持。
本地Windows改代码，经GitHub同步；生产权重/数据/输出在远程Linux，本地不执行正式GPU任务。

## 精确 checkpoint 与模型

`target.json` 绑定：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_final_l10_14x3_strong_a9gate_bias3_l15_gbs128_5090/formal_20261127_171041273199107/formal/checkpoints/model--step-2542.ckpt
```

自动读取同run `formal/runtime.json`。不扫描latest，不把2542预先当作训练终点；实际strong行数N、总步数ceil(N/128)、1epoch及游标由原合同计算。
CPU核对原route、完整合同SHA、源码SHA、strong-only来源/视图、8×4×4、step2542、457状态库存；中间和已完成的合法游标均可model-only评测。
不创建/恢复optimizer、scheduler或训练RNG，不要求原训练音频/manifest重新prepare。

训练源码仍在历史目录 `mizar_l9_12x3_strong_reason/`，当前route为 `mizar_final_l10_14x3_a9_gate_bias3_l15_strong_gbs128_v1`。
固定Git版本 `816b48e269c4e51235e8617892868eaa24a6cae9`，全部30份Python/Shell/release manifest逐文件SHA及训练code_digest见 `source_manifest.json`。
从匹配Git object建立内容校验快照到 `code/RSmol/log/mizar_l10_14x3_eval_sources/`，不依赖后续可能原地改动的训练目录。远程需保有该Git提交；浅克隆缺object需补齐历史。
可显式 `--source-revision` 指定内容完全匹配的其它版本，仍逐文件验SHA。

模型原样执行：`L1..9 → (L10..14)×3 → L15 → L16..30 → final norm`，30物理层、40次block。
A9是原L9 raw post-block的audio126。第一轮H9+b1；后两轮音频先按 `A9+tanh(Linear(previous_audio))*previous_audio` 重构，非音频继承前轮，再给全部token加当前bias。
共享gate两次，三个bias向量存为 `[3,576]`；L15训练但不循环，每次forward重新构建A9，无跨题状态。

GPU以原官方final完整454状态构建CED/projector/30层LM，按原loader仅一次注册gate/bias，然后strict覆盖457项checkpoint。
覆盖后不再执行 `enable_training/enable_adapters`，确保训练后的L10–15/gate/三个bias完整保留。检查全部名称/shape/dtype/finite、tied embedding/head、冻结tensor逐值和digest。
随后全模型FP32/eval/frozen；每个真实解码forward审计40次层轨迹、两个gate调用、三轮bias入口及final norm一次。

## 与原评测可比的设置

复用原 `mizar_core4_eval` 的MMAU/MMAR评测控制流和数值函数，`greedy_decode`、prompt/generate、音频fixed20、原官方scorer调用的AST对照不变。

- AU原test-mini完整1000，AR完整1000；原固定选项顺序、小写题面，不增加仅回答字母指令，不使用候选likelihood。
- FP32、eval、batch1、seed1234、无autocast、TF32 off；原SoundFile/mono/32k、fixed20中心裁剪/右补零、10+10拼回whole CED，每题音频仅编码一次。
- 原空格EOS后缀，prompt256右截断/右补PAD `!`=17，SEP/EOS=0；126+1+256=383，首答案logit382。
- 原因果attention、PAD可见、attention_mask=None；无KV，每token完整重算，argmax最多300、首EOS停止，raw decode原样交原官方scorer。
- 各项统一pdgpu-4090/1GPU/8CPU/32G/1node，原镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`，conda `mellow_c8204d8`、Transformers4.54.1，无额外GPU smoke。

## 同步与两条提交命令

`sync_files.txt` 列本次新增实现/入口/测试/报告和RM；`reference_dependencies.txt` 列已存在且需随仓库保有的共享依赖与训练源码。
本地PowerShell仓库根，审阅后自行提交（助手不自动执行）：

```powershell
Get-Content code/RSmol/mizar_l10_14x3_eval/sync_files.txt | ForEach-Object { git add -- $_ }
git diff --cached --check
git diff --cached --stat
git commit -m "Evaluate L10-14x3 gated Mizar checkpoint on MMAU and MMAR"
git push
```

远程Linux仓库根：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# AU：MMAU test-mini完整1000题
bash code/RSmol/run_mizar_l10_14x3_eval_4090.sh mmau

# AR：MMAR完整1000题
bash code/RSmol/run_mizar_l10_14x3_eval_4090.sh mmar
```

路径已固定，无需手写长checkpoint参数。可先只预览参数或做CPU预检：

```bash
bash code/RSmol/run_mizar_l10_14x3_eval_4090.sh all --check-args
bash code/RSmol/run_mizar_l10_14x3_eval_4090.sh all --preflight-only
```

`all`也可一次提交两项，与两条单项命令择一。每次调用CPU预检一次，失败不调用vc；all中单项vc失败仍继续另一项，最终整体非零并保存诊断。
READY_FOR_GPU只表示CPU header/runtime/source通过，生产资产和tensor/CUDA检查仍由GPU任务完成；SUBMITTED仅表示调度提交成功。
副本迁移可用 `--checkpoint`、`--runtime-config`、`--output-root`，仍只接受文件名step2542和原路线合同；非标准目录需同时明确runtime和output。
保留原 `RSMOL_CONDA_BASE`/`MELLOW_CONDA_BASE` 覆盖。

## 输出、日志与续评

默认结果在指定run下：

```text
formal_20261127_171041273199107/eval/
  mizar_l10_14x3_a9gate_bias3_l15_eval_20261127_v1/
    mmau/model--step-2542/
    mmar/model--step-2542/
```

每项保留 `run_config.json`、`raw_generations.jsonl`、`model_audit.json`、`evaluation_report.json` 和原官方预测/评分日志。
`run_config`固定checkpoint/runtime/源码/数据/scorer/协议身份；raw保存实际prompt token/mask与两段音频SHA。
相同单项命令重跑，只生成剩余题；原始完整坏行、冲突progress、身份变化/缺题/重复不会静默通过，未写完末行可恢复。
MMAU/MMAR保持原流程，续评/重评分仍加载模型核验；只有完整1000覆盖、无skipped、原官方scorer成功并确认完整分母才评测PASS。

日志目录 `code/RSmol/log/mizar_l10_14x3_eval/`：`ml1014-gb3-<mmau或mmar>-<tag>.JOB.log`；同目录有timestamp `.submission.json`，记录确切checkpoint、输出、CPU预检及vc返回内容。
既有baseline及×4分数仅在共同数据/题面/音频/生成/scorer一致时可对照；此次不额外重跑baseline。

## 本地验证边界

入口 `tests/validate_mizar_l10_14x3_eval.py`，报告 `audit_reports/mizar_l10_14x3_eval_validation.json`。
真实小30层Llama eager/SDPA对照非零gate/bias独立公式、40次执行/两gate/三bias/final norm一次、无跨forward状态、训练683与推理383首/第二答案及因果性；严格重载与冻结漂移拒绝。
实际loader控制流用明确小模型/官方资产/CUDA夹具；真实SoundFile双声道16k→mono32k/fixed20与一次音频编码。完整AU/AR1000题runner用明确模型/音频/parquet合成夹具，vc仅模拟，Bash及Git object读取真实执行。
本地没有生产checkpoint/音频/CUDA，不能代替远程457项生产tensor、4090显存/速度及正式分数验证。
