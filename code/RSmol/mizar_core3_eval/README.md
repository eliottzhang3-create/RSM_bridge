# 原始 Mizar baseline 与三条三循环模型的四项评测

本地 Windows 修改核心代码，GitHub 中转；权重、数据、GPU 作业和结果位于远程 Linux。
独立入口：code/RSmol/run_mizar_core3_eval_4090.sh。参考 ced_eval_20261125，
不修改旧训练、旧评测和历史结果。

| target | 模型 |
|---|---|
| official_mizar | 官方 final，完整已训练 CED/projector/LM；30 物理层、30 次执行 |
| l9_12x3 | 显式 checkpoint；1..8,(9..12)x3,13..30；38 次，保留已训练 A8 门控 |
| l14_17x3 | 显式 checkpoint；1..13,(14..17)x3,18..30；38 次，无额外注入 |
| l19_22x3 | 显式 checkpoint；1..18,(19..22)x3,23..30；38 次，无额外注入 |

三条训练产物不预设路径或最终步数，不扫描 latest。
prepare_mizar_l9_12x3_strong_reason.sh 是训练数据准备器，不产出模型，也不需要为评测重新运行。

## 四项数据和推理协议

| task | 范围与评分 |
|---|---|
| reasonaqa5000 | 原 seed1234 固定 5000 manifest；完整答案规范化 exact |
| mmau | 原 test-mini 全 1000；原始生成直接交原官方 scorer |
| mmar | 原全 1000；原始生成直接交原官方 scorer |
| mmsu | 原全 5000；紧凑全小写四槽、null 保留；mmsu_answer_text_contains_v2 和原官方汇总 |

沿用 FP32/eval/batch1/seed1234、无 autocast、TF32 关闭；mono32k，
fixed20 中心裁剪/右补零、10+10，然后整段在线 CED。每题一次音频编码。
原 EOS 后缀、prompt256、右截断/右补齐、PAD!=17、SEP/EOS0；
126+1+256=383，首答案 logit382。原因果 attention_mask=None、PAD 可见。
无 KV cache、每 token 全序列重算、argmax、最多 300 个新 token、首 EOS 停止，
保留原始 decode/EOS/空格，不改成字母抽取或 likelihood。
各循环模型按原训练 forward 执行，anchor/隐藏状态每次完整 forward 重建。

baseline 是本项目参考评测协议下的官方 final 对照。四个模型共 16 个独立作业。
每项 pdgpu-4090、8CPU/32G/1GPU/1node、原镜像和 mellow_c8204d8；
无额外 GPU smoke，不修改共享环境，支持 RSMOL_CONDA_BASE / MELLOW_CONDA_BASE。

## Git 同步与四模型提交

本地精确交付清单见 sync_files.txt；既有参考评测、共享依赖及三条训练源码也需已同步。
远程仓库根：

~~~powershell
git add -- code/RSmol/mizar_core3_eval code/RSmol/run_mizar_core3_eval_4090.sh tests/test_mizar_core3_eval.py tests/test_mizar_core3_eval_tensors.py tests/test_mizar_core3_eval_workflow.py tests/validate_mizar_core3_eval.py audit_reports/mizar_core3_eval_validation.json
git add -p -- RM.md
git commit -m "Add Mizar baseline and three core-loop checkpoint evaluations"
git push
~~~

~~~bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# baseline 可以先独立提交四项
bash code/RSmol/run_mizar_core3_eval_4090.sh official_mizar
~~~

三个训练产物产生后，把下面变量替换为真实完整路径：

~~~bash
CKPT9='/实际/L9_12/run/formal/checkpoints/model--step-实际步数.ckpt'
CKPT14='/实际/L14_17/run/formal/checkpoints/model--step-实际步数.ckpt'
CKPT19='/实际/L19_22/run/formal/checkpoints/model--step-实际步数.ckpt'

bash code/RSmol/run_mizar_core3_eval_4090.sh l9_12x3 all --checkpoint "$CKPT9"
bash code/RSmol/run_mizar_core3_eval_4090.sh l14_17x3 all --checkpoint "$CKPT14"
bash code/RSmol/run_mizar_core3_eval_4090.sh l19_22x3 all --checkpoint "$CKPT19"
~~~

每条提交该模型四项。单项补交时以 task 替换 all，例如：

~~~bash
bash code/RSmol/run_mizar_core3_eval_4090.sh l9_12x3 reasonaqa5000 --checkpoint "$CKPT9"
bash code/RSmol/run_mizar_core3_eval_4090.sh l9_12x3 mmau --checkpoint "$CKPT9"
bash code/RSmol/run_mizar_core3_eval_4090.sh l9_12x3 mmar --checkpoint "$CKPT9"
bash code/RSmol/run_mizar_core3_eval_4090.sh l9_12x3 mmsu --checkpoint "$CKPT9"
~~~

其他 target 同理。official_mizar 单项不需要 checkpoint。不要与同模型 all 同时重复提交。

## 参数预览、CPU 预检与源码

~~~bash
bash code/RSmol/run_mizar_core3_eval_4090.sh official_mizar --check-args
bash code/RSmol/run_mizar_core3_eval_4090.sh l19_22x3 all --checkpoint "$CKPT19" --check-args
bash code/RSmol/run_mizar_core3_eval_4090.sh l19_22x3 all --checkpoint "$CKPT19" --preflight-only
~~~

--check-args 不读生产权重、不申请 GPU。每模型正式提交前只预检一次；
失败则该模型四项均不提交。某次 vc 失败仍继续其余项，整体非零并保存错误。
READY_FOR_GPU 仅为 CPU 预检，SUBMITTED 仅为提交成功，正式结果看 evaluation_report.json。

官方 baseline 核对 18 个 release 文件 SHA/大小，严格加载原始 454 状态。
训练产物核对 route、同 run runtime/contract SHA、训练源码、动态 geometry、step/epoch 游标和
454/456 状态库存；GPU 再核对全部 shape/dtype/finite、tied aliases、冻结状态与官方来源逐值一致。
接受完整保存的中间或最终 checkpoint，不沿用训练 resume 的“完成即拒绝”条件；
不写死 34626/5089，不创建或恢复 optimizer/scheduler/训练 RNG。

source_manifest.json 固定三个未修改训练包的源码摘要；首次 CPU 预检制作内容校验的不可变快照：
code/RSmol/log/mizar_core3_eval_sources/<code_sha256>/<训练包>/。
全部原训练 import 从对应快照加载。若以后 checkout 的训练代码演进，可显式给出匹配本版摘要的 Git revision：

~~~bash
bash code/RSmol/run_mizar_core3_eval_4090.sh l9_12x3 all \
  --checkpoint "$CKPT9" --source-revision '对应训练源码Git提交'
~~~

只读 Git object，不切换分支、不放宽摘要。真实训练代码/配方发生变化需要独立适配。
标准 checkpoint 自动定位 formal/runtime.json；非标准位置必须同时传 --runtime-config 和 --output-root。
--model-dir 默认 /hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M，覆盖后仍严格核对来源。

## 输出、续跑与比较

默认 baseline：
outputs/RSmol/mizar_official_core3_baseline/eval/mizar_core3_eval_v1/<task>/stage3_q4_seed20260905/。

默认训练产物：
<原run>/eval/mizar_core3_eval_v1/<task>/<checkpoint-stem>/。

MMSU 的 task 目录为 mmsu_compact_lowercase_v1。
显式 --output-root ROOT 时写 ROOT/<target>/<task>/<checkpoint-stem>/。

日志在 code/RSmol/log/mizar_core3_eval/：*.submission.json 含 admission、路径、vc 参数/stdout/stderr；
mcore-*.JOB.log 为 GPU 日志。各结果含 run_config.json、raw_generations.jsonl、
model_audit.json、evaluation_report.json；ReasonAQA 另有 accuracy.json。
逐题持久化、输出锁、身份保护、坏完整行拒绝和未完成尾行恢复继承原实现；
完整覆盖后才评分。中断后确认旧作业已停止，再重提原单项命令，只补剩余题。
不能修改旧 run_config 绕过身份；同身份完整 raw 可重评分。

完成某项后比较四模型：

~~~bash
bash code/RSmol/mizar_core3_eval/run.sh compare --benchmark mmsu \
  --report-dir /实际/baseline/mmsu结果目录 \
  --report-dir /实际/l9/mmsu结果目录 \
  --report-dir /实际/l14/mmsu结果目录 \
  --report-dir /实际/l19/mmsu结果目录 \
  --output /实际/新目录/mmsu_core3_comparison.csv
~~~

benchmark 可换 reasonaqa5000/mmau/mmar；核对完整分母、相同题序/数据/共享代码和协议，
检查原始回答与评分输入一致，CSV/JSON 记录 30/38 次执行。
不要修改正在运行任务依赖的评测代码。

## 本地验证

~~~powershell
.\.hrm_test_runtime\Scripts\python.exe -B tests/validate_mizar_core3_eval.py --dependency-dir .codex_audio_cutoff_deps --dependency-dir .hrm_test_runtime/audio_packages
~~~

audit_reports/mizar_core3_eval_validation.json 保留每轮结果和环境。
真实 CPU Llama 检查原始 30 层、三条 38 次轨迹、独立递推、已训练 gate/state、
383/683 首/第二答案位置与因果性、严格加载拒绝、真实 SoundFile/重采样及四个生成适配器。
完整尺寸控制流显式使用合成模型/音频/Parquet/CUDA夹具；bulk fsync 仅测试模拟。
已有 torchaudio 未用 native 扩展的 ABI 差异只在测试进程适配，生产代码不加适配。
CPU 通过不代表远程真实 checkpoint、CUDA、4090 显存/吞吐或实际分数通过。

Final local acceptance: 23/23 CPU tests passed, no skips. See the validation JSON for previous runs and explicit fixture boundaries.
