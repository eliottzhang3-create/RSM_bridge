# ReasonAQA test close-ended 完整评测

本入口独立于原 MMAU/MMAR/MMSU 的数据读取、题面与评分链路，只调用原有严格 CED 模型 loader、greedy decoder 和逐样本注入 context。原三类评测文件、训练实现和模型源码均不修改。本地 Windows 只准备代码，经 GitHub 同步后，在远程 Linux 使用现有权重、音频、环境和 GPU。

## 数据和唯一评分口径

默认数据：`/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test.json`。

按原 JSON 顺序筛选，不去重，不改原题、答案或 ID：

| 原始 subtype | 纳入范围 | 统计名称 |
| --- | --- | --- |
| `AudioCaps-MCQ.json` | 全部 | `audiocaps_mcq` |
| `Clotho-MCQ.json` | 全部 | `clotho_mcq` |
| `ClothoAQA.json` | 答案经首尾去空白、casefold 后严格等于 `yes` 或 `no` | `clotho_aqa_yesno` |

其他数据只记录排除数量。ID 为 `reasonaqa_test:<原始行号，从0开始>`，保留整个原始 record。实际题量由远程 test.json 审计得出，不根据示例推定。

唯一指标为**完整答案的规范化精确匹配**。预测和标准答案各自依次执行：去首尾空白、移除末尾一个 `<|endoftext|>`、Unicode casefold、连续空白合并为空格。保留选项字母、括号、答案正文、标点，不提取字母、不使用包含关系或语义匹配。

例如标准答案 `B) A person snoring` 可以匹配 ` b) A  person snoring<|endoftext|> `；`B`、`A person snoring`、`B) A person snoring.` 均判错。标准答案 `no` 不匹配 `no.`。原始生成及 token IDs 完整保存，规范化只用于评分副本。

输出 overall 微平均正确率及三类各自的 correct/total/accuracy，accuracy 为 0–1 的比例。空回答和格式不匹配均算错、保留在完整筛选分母中；空类别 accuracy 为 null。必须生成全部筛选题目后才评分。任何路径、解码、模型或存储错误均停止并返回非零，不跳题缩减分母。

## 音频和推理设置

逻辑路径支持反斜杠和空格，沿用项目 ReasonAQA 原音频映射方式：

| 数据 | 默认真实根目录 | 示例映射 |
| --- | --- | --- |
| AudioCaps | `/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2` | `AudioCapsLarger/test/x.wav` → `test/x.wav` |
| Clotho | `/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1` | `ClothoV21/evaluation/Freezing Rain.wav` → `evaluation/Freezing Rain.wav` |
| ClothoAQA | `/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_audio/audio_files` | `ClothoAQA\audio_files\dripping.wav` → `dripping.wav` |

ClothoAQA 优先精确文件名，必要时使用项目原有 Unicode/标点规范化文件 stem 唯一匹配。缺失、多个候选、路径越出类别根目录均失败。启动时解析全部入选行的路径并记录 size/mtime_ns，每次真实解码前后复核；逐题保留重采样后及两个实际音频 half 的波形 SHA256。续跑按完整记录身份核对，七模型比较另核对实际模型输入指纹。

各目标使用自己的训练 `load_mono` / `fixed_len`：mono、32 kHz resample；单文件整段居中裁剪或右补零到20秒，再分两个10秒；真实 filepath2 非空时，两文件分别裁剪或右补零到10秒。不复制第一段、不用随机音频补失败样本。

推理参考既有 CED MMAU/MMAR 协议：仅将原始 `input` 转小写后用训练 tokenizer 固定到256 tokens，不追加新指令，不把 GT/caption/taskname 等字段输入模型；126 audio + 1 SEP + 256 prompt = 383 prefix，首个答案 logit 在382。FP32、eval/frozen、seed1234、TF32关闭、batch1、300-token纯 argmax、完整重算、无 KV cache、LM attention_mask=None，首个生成 EOS 停止。注入/reinjection/Loop-LoRA/MeSH 使用原 runtime 的每样本 context，不更改其模型行为。checkpoint、配套 runtime、训练源码/合同、资产和 tokenizer 均使用原 loader 的严格校验。

## 七个固定目标

所有目标固定 `model--step-17313.ckpt` 和训练 geometry `8,8,4`。训练 geometry 用于核验合同，评测本身始终单卡 batch1。下表 run 均位于 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/`：

| TARGET | 模型 | run 相对目录 |
| --- | --- | --- |
| `base` | 标准 CED-Small S1 | `mizar_s1_ced_small_gbs256_5090/formal_20261026_193830617147305` |
| `recursive_5_10_5` | 普通5-10-5递归 | `mizar_s1_ced_small_gbs256_recursive_5_10_5_5090/formal_20261028_160823559722647` |
| `serial84` | 8→4注入、audio reinjection | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_5090/formal_20261102_112121702891801` |
| `serial48_lora_r8` | 4→8注入、audio reinjection、Loop-LoRA r8 | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8_5090/formal_20261104_131843931946322` |
| `serial84_lora_r8` | 8→4注入、audio reinjection、Loop-LoRA r8 | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8_5090/formal_20261103_165410438279511` |
| `recursive_mesh` | 原始 MeSH | `mizar_s1_ced_small_gbs256_recursive_mesh_5090/formal_20261027_081123150934256` |
| `mesh48_lora_r8` | MeSH 4→8注入、Loop-LoRA r8 | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_loop_lora_r8_5090/formal_20261104_161727489909082` |

每个 run 下取 `formal/checkpoints/model--step-17313.ckpt`、`formal/runtime.json`。完整绑定见 `config.py`，不自动找 latest，不用残留模型环境变量改变目标身份。预设拒绝 checkpoint/runtime/variant/step/geometry/route 覆盖。

## 本地同步

在本地仓库根核对并提交本次文件；以下命令供用户执行，本次未代为 commit/push：

```powershell
git add -- code/RSmol/mizar_s1_ced_reasonaqa_eval code/RSmol/run_mizar_s1_ced_small_reasonaqa_eval_5090.sh code/RSmol/run_mizar_reasonaqa_selected7_5090.sh tests/test_mizar_s1_ced_reasonaqa_eval.py tests/test_mizar_s1_ced_reasonaqa_shell.py tests/validate_mizar_s1_ced_reasonaqa_eval.py audit_reports/mizar_s1_ced_reasonaqa_eval_validation.json RM.md
git diff --cached --stat
git commit -m "Add isolated ReasonAQA test close-ended evaluation for seven CED checkpoints"
git push
```

## 远程提交

本地代码 push 后，在远程 Linux 仓库根执行。一键提交七个独立的完整评测作业：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_reasonaqa_selected7_5090.sh all
```

或逐条提交，以下与 all 二选一，避免重复提交同一输出：

```bash
bash code/RSmol/run_mizar_reasonaqa_selected7_5090.sh base
bash code/RSmol/run_mizar_reasonaqa_selected7_5090.sh recursive_5_10_5
bash code/RSmol/run_mizar_reasonaqa_selected7_5090.sh serial84
bash code/RSmol/run_mizar_reasonaqa_selected7_5090.sh serial48_lora_r8
bash code/RSmol/run_mizar_reasonaqa_selected7_5090.sh serial84_lora_r8
bash code/RSmol/run_mizar_reasonaqa_selected7_5090.sh recursive_mesh
bash code/RSmol/run_mizar_reasonaqa_selected7_5090.sh mesh48_lora_r8
```

每作业：`pdgpu-5090`、1GPU / 8CPU / 32G / 1node，镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`，conda `mellow_c8204d8`，HF/Transformers offline。不加自动 smoke。日志为 `code/RSmol/log/mizar_s1_ced_reasonaqa_5090.<时间_PID>.JOB.log`。

只检查七个目标的有效参数，不提交、不读数据/权重、不做推理：

```bash
bash code/RSmol/run_mizar_reasonaqa_selected7_5090.sh all --check-args
```

允许覆盖 `--test-json`、`--audiocaps-root`、`--clotho-root`、`--clotho-aqa-root`，以及单目标的 `--output-dir`；支持 `--option=value` 和含空格路径。all 不接受共用 output-dir。需要重跑到新目录可用：

```bash
bash code/RSmol/run_mizar_reasonaqa_selected7_5090.sh base \
  --output-dir '/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/reasonaqa_base_rerun_v2'
```

通用入口也支持已登记的这七种架构，用户显式绑定其他训练 run 时使用：

```bash
bash code/RSmol/run_mizar_s1_ced_small_reasonaqa_eval_5090.sh \
  --eval-variant base --run-root '/明确的训练run目录' \
  --expected-step 17313 --expected-geometry 8,8,4
```

## 输出、失败与续跑

每个目标默认结果目录为自己的训练 run 下：`eval/reasonaqa_close_ended_step17313_v1/`。

| 文件 | 内容 |
| --- | --- |
| `run_config.json` | checkpoint/runtime/test文件SHA、源码摘要、协议、筛选/解析记录身份 |
| `dataset_audit.json` / `selected_records.jsonl` | 全数据筛选计数及所有入选原始行、真实音频路径 |
| `raw_generations.jsonl` | 逐题原始生成、token IDs、完整题目、波形摘要、实际模型输入指纹 |
| `progress.jsonl` | 已持久化的顺序进度 |
| `model_audit.json` | 原 runtime 的严格加载、资产/合同/结构和执行审计 |
| `accuracy.json` | 唯一完整答案精确匹配指标，overall和三类别 |
| `per_sample_results.jsonl` | 原始/规范化预测及答案、correct |
| `evaluation_report.json` | 整轮 PASS/FAILED、完整覆盖、指标、失败堆栈 |

原始记录逐条 append+fsync；OS进程锁拒绝并发写同一目录。再次执行同一目标命令会在身份完全一致时续跑。仅截去最后一次未完成的无换行写入；完整损坏记录、顺序/内容/hash 不一致均硬失败。progress 从 raw 重建；完整 raw 的重新评分不加载模型。

checkpoint、runtime、test JSON、入选音频 size/mtime、源码或协议改变时，旧目录被拒绝，不覆盖旧报告；使用新目录，或在原代码和原资产下续跑。缺少/损坏音频也失败，不跳过。首次启动预检失败会打印错误并返回1；若尚未认领输出目录，不会把失败报告写进旧目录。

七个作业完成后可在远程汇总。比较工具验证各轮 PASS、完整覆盖、相同 test/筛选/协议/评测源码、原始记录hash及重算分数，并逐题比较 prompt token IDs 和两个真实音频 half 的输入指纹；不满足就拒绝汇总。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate mellow_c8204d8
python -B -m mizar_s1_ced_reasonaqa_eval.compare
```

默认生成 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/reasonaqa_selected7_step17313_v1.csv` 和同名 JSON。可用重复 `--report-dir '/具体结果目录'` 及 `--output '/汇总路径.csv'` 比较自定义输出。CSV 中 accuracy 为0–1比例。

## 本地验证

```powershell
python -B tests/validate_mizar_s1_ced_reasonaqa_eval.py
```

验证包含筛选/完整答案评分、路径解析、真实训练 fixed_len 在明确 NumPy tensor fixture 上的数值结果、生成题面隔离、七路 loader/context 调用、完整流转/中断恢复/坏记录拦截/汇总一致性，以及真实 Bash 入口+模拟 vc/conda、原 MMAU/MMAR/MMSU 回归和132份原评测文件字节SHA核对。

报告：`audit_reports/mizar_s1_ced_reasonaqa_eval_validation.json`。真实 Torch/音频或其他依赖缺少时相关测试明确跳过；合成测试通过不表示远程模型或完整 test 推理已通过。Windows 沙箱限制临时目录时需在获准的沙箱外运行。本次不访问远程生产资产，不提交真实 GPU 作业。
