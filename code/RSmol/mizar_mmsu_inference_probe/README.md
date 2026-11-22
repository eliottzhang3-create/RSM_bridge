# MMSU：题面对照与训练/推理数值核验

用户报告多个架构 MMSU 只有个位数，而相同 checkpoint 在其他 benchmark 回答正常；
三个 accent_identification 原始回答出现无关内容和重复，问题发生在 scorer 之前。
本目录提供三题隔离诊断，不改生产评测、训练模型或既有结果，不输出完整 benchmark 分数。

2026-11-22用户随后授权正式MMSU改为紧凑格式＋全小写，现已接入共享dataset.prompt。
本工具的official_original/official_lowercase继续显式使用保留的历史official_prompt，
因此不会随着正式默认输入改变而丢失旧模板对照；compact_lowercase与新正式输入一致。
历史Git后端没有official_prompt时仍读取该快照当时的dataset.prompt，保留原诊断可复现性。

## 2026-11-22：用户回传真实三题对照

用户回传71427 base / step34626诊断，报告路径：
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_text71427_mesh_3090/formal_20261119_141531507533438/eval/mmsu_probe_20261122_112042_1279907/diagnostic_report.json`。
以下证据来自用户贴出的作业日志，未在本地独立读取远程完整报告。

三题 runner_parity 均为true；四种题面共12项 training_layout_passed 均为true，
最终状态PASS_NUMERICAL_PARITY。按该诊断实现，最终PASS还要求三题双音频解码器逐位一致。
这支持三题上同输入runner一致、首/第二teacher-forced答案位置logits在既定容差内对齐，
首预测argmax一致；不代表所有生成步/所有架构验证完成，也不代表回答正确。

| 题面 | 用户回传的三题行为 |
|---|---|
| 官方原大小写 | 重现此前无关文本/反复ppe、ibe等异常 |
| 官方题面只转小写 | 仍出现复述或反驳字母指令、混杂选项等异常 |
| 紧凑题面原大小写 | south continent、dered sibling、humour，仍有非选项回答 |
| 紧凑题面全小写 | `c) india`、`a) new zealand`、`c) singapore`，均正常结束于EOS |

此前用户给出的GT分别为India、New Zealand、Australia，所以紧凑小写为两题正确、一题错误；
第三题恢复正常回答格式并不等于答对。这是针对已知异常三题的诊断，不是随机样本分数或完整5000题准确率。
官方小写第三题同时提到australia和singapore，原v2文本包含指标会计对，仍不能据此称其生成已恢复正常。

控制同一checkpoint、音频和生成设置，只改变题面即可改变退化行为，支持**题面格式/大小写触发退化**的判断。
仅lowercase不足；四个case没有分别隔离instruction/换行/Question前缀/选项符号，不能单独归罪其中某一项。
新旧MMSU一致只能说明复现旧协议，不能保证官方题面适合该模型的训练输入分布。
正式重评宜统一采用现有MMAU构造器的紧凑小写题面，保留原greedy/音频/模型和既有评分规则，
另立prompt协议与输出目录，对全部待比较架构使用同一模板；不能把新旧题面分数混作同协议比较。
本次仅解释和记录用户回传证据，尚未修改正式MMSU题面或执行5000题重评。

## 已确认的源码结论

新 `text71427_ced_eval/mmsu_runtime.py` 的 read_audio、decode_audio、generate、
prefix_and_decode 与原 `mizar_s1_ced_mmsu_eval/runtime.py` 的函数 AST 相同。
二者复用相同 dataset、parsing、scorer；execute/run/summarize 也相同。
旧实现一致性只能证明代码复用，不能证明远程真实数值正确或题面合适。

| 项目 | 音频训练 | 原/新 MMSU 推理 | 源码检查 |
|---|---|---|---|
| 音频 | SoundFile FP32、mono→32k、fixed20 中心裁剪/右补零、10+10秒重拼后完整 CED | 同样顺序；内嵌音频 bytes 优先 | 一致，真实音频另做双解码器对照 |
| prefix | audio126 + SEP1 + prompt256 | 383 tokens | 一致 |
| 分词 | 原 tokenizer，追加 ` <\|endoftext\|>`、右截断、右 PAD `!` | 相同训练 tokenize 方法 | 一致 |
| 可见 padding | attention_mask=None | attention_mask=None | 一致；不要改为取最后非 PAD logits |
| 首答案 | answer_loss 用 logits[:,382:-1] 对齐 answer300 | prefix 的 logits[:,-1] 即382 | 静态位置一致，工具比较真实数值 |
| LM | FP32、无 autocast、TF32 关闭 | 同样；eval、mapper dropout 关闭 | 推理关闭 dropout 是预期差别 |
| 循环/分支 | 各自真实模型 forward | 相同训练类、strict 完整 state 加载、每 token 全序列重算 | 既有 layer/router/LoRA hooks 核验 |
| 题面 | 普通 QA 经过 text_pair 小写转换；训练数据具体格式随来源 | MMSU 官方大写题面、`Question:`、换行、A. 等、额外仅答字母指令 | **不同**；与正常 MMAU/MMAR 紧凑小写格式也不同 |
| 输出/评分 | 混合任务答案文本 | 要求只答 A/B/C/D，但 v2 判断完整 answer_gt 文本是否出现 | **指令与指标不协调**；此前明确选定的旧口径 |

指令/评分冲突可以令正确字母回答被 v2 判错，但不能单独解释无关重复文本。
目前最需要实测的是题面格式/大小写敏感性；没有真实对照前不把它写成已确认根因。
工具不增加 temperature、repetition penalty，不改 prefix、PAD/EOS、KV 或答案位置来掩盖症状。

## 一次作业检查什么

默认使用用户贴出的三个精确 ID，每题四种独立题面（其他架构也可按相同方式分别运行）：

1. 官方题面原大小写。
2. 相同官方题面全小写：只隔离大小写。
3. 原 MMAU/ReasonAQA 紧凑题面，保留原大小写。
4. 同一紧凑题面全小写：与其他 benchmark 使用的构造器相同。

所有选项位置及 None 槽保留；构造题面不读取 answer_gt。
每题四次自由生成全部完成后，才将 GT 文本用于单独的 teacher-forced 诊断，绝不流入自由生成或正式预测。
另用原 MMAU/MMAR runner 对完全相同的小写官方题面推理，核对 token IDs，排除 runner 包装差异。

每种题面比较真实 `Mellow.forward(audio, prompt256, answer300)` 的683长度输出与：

- prefix383 的 LM forward：首答案 logits 与真实生成第一 token。
- prefix383 + 一个 teacher-forced answer embedding 的 LM forward：第二答案 logits。

两侧均 eval/FP32、无cache、同一权重和音频；额外记录 token IDs/mask、PAD/EOS、prefix/波形 SHA、
首答案实际位置、logits 最大差、argmax、GT 文本 NLL（仅诊断）、layer/router/LoRA 实际审计。
teacher forcing 用来查训练布局对齐，不能当作该样本的生成正确率。

`PASS_NUMERICAL_PARITY` 只表示三题数值/runner/音频对照通过，不表示答案正确或低分问题解决。
数值/runner/音频不一致为 `DIFFERENCE_FOUND`，加载/覆盖/数据错误为 `FAILED`。
源码审计和题面差异成立，也不能替代 checkpoint 的真实 GPU 对照。

## 先诊断用户提供的71427纯MeSH两循环

base 训练目录在2026-11-21已改为“从旧音频 checkpoint 重训、统一 LR2e-4”的新配方，
旧 step34626 的 strict source hash 与当前目录不再相同。
因此此处明确取变更前的 Git `8a9f1f76722e77cba4be429a578c63fdcf6ee527`，
在共享 log 子目录解包 code/RSmol 快照，保留原完整 loader 验证，当前训练代码和分支不切换。
该历史 revision 的 base .py/.sh 与原评测 protected_sources 清单相符；服务器 ckpt 合同仍由 loader 最终核验。
快照需保留到作业结束；不要把源快照放在仅登录节点可见的 /tmp。

经 GitHub 同步本次工具后，在服务器仓库根执行，一个4090作业最多15条自由生成，另有数值 forward：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
CKPT='/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_text71427_mesh_3090/formal_20261119_141531507533438/formal/checkpoints/model--step-34626.ckpt'
RUN="${CKPT%/formal/checkpoints/*}"
DIAG="$RUN/eval/mmsu_numeric_probe_$(date +%Y%m%d_%H%M%S)_$$"
bash code/RSmol/run_mmsu_inference_probe_4090.sh \
  --source-revision 8a9f1f76722e77cba4be429a578c63fdcf6ee527 \
  --backend text71427 --eval-variant base --checkpoint "$CKPT" --output-dir "$DIAG"
printf 'report: %s/diagnostic_report.json\n' "$DIAG"
```

结果写入新的 `$DIAG/diagnostic_report.json`；日志在
`code/RSmol/log/mmsu-probe.<timestamp_pid>.JOB.log`。pdgpu-4090/1GPU/8CPU/32G，原镜像和环境。
`--source-revision` 是诊断提交 shell 的选项；`run.sh` 直接计算入口对应 `--implementation-root <快照/code/RSmol>`。
新输出必须不存在，工具拒绝覆盖，也不会续跑或修改任何正式 MMSU raw/scorer 结果。

同一入口适用于其余71427路线，只须正确的 eval-variant/checkpoint。
旧5–10×2–5等架构使用 `--backend legacy`，参数与旧 MMSU 一致，例如：

```bash
bash code/RSmol/run_mmsu_inference_probe_4090.sh --backend legacy \
  --eval-variant recursive_mesh --run-root /实际训练run根目录 \
  --expected-step 17313 --expected-geometry 8,8,4 --output-dir /实际新诊断目录
```

旧架构与71427的 base 别名含义不同，不可混用。旧模型仍用它自己的 loader；不凭名称猜架构。
可加 `--sample-id` 指定其他题，每个ID一次；默认就是异常的三题。
`--check-args` 预览不加载权重、数据或模型；显式 `--source-revision` 会先写入独立源码快照。

## 如何据结果处理

数值与双 runner 都通过、只有紧凑小写恢复正常，才能把证据指向 MMSU 题面分布。
小写官方也恢复，则大小写就是其中一个因素；仅改变格式恢复则不能单独归因于大小写。
同一题面双 runner 不同，或训练布局/推理 logits 不一致，应先修调用链，再测全量。
所有题面都退化时，继续查具体模型/数据/原结果身份；三题不能代表5000条准确率。
若确认采用新题面，应另立协议和新目录，所有架构统一重评，保留旧分数用于历史复现。
不要悄悄替换官方题面、换成字母指标或覆盖旧 raw 来造成不对齐的分数。

本地测试：`python -B -m unittest discover -s tests -p test_mmsu_inference_probe.py -v`。
合成数值模型用于检查诊断工具的索引/拒绝路径；另有真实小型MeSH CPU测试，缺ML依赖时明确跳过。
本地无生产权重、MMSU音频和CUDA；真实对照结论要看上述服务器报告。

2026-11-21联合验收：新工具12项 + 原三题诊断6项 + 原MMSU22项，共40项，39通过、1项真实CPU测试缺依赖跳过。
补全真实模型测试依赖声明后，新工具12项定向复验为11通过/1跳过。
2880份本轮开始时的既有生产Python/Shell/JSON摘要保持，原MMSU/其他benchmark/训练/模型均未改。
Python3.10语法、真实Bash预览/模拟vc资源和含空格参数、历史Git快照独立进程导入通过。
报告：`audit_reports/mmsu_inference_review_validation.json`；GPU/真实checkpoint明确未执行。

本地同步本次工具（先审阅再提交）：

```powershell
git add -- code/RSmol/mizar_mmsu_inference_probe code/RSmol/run_mmsu_inference_probe_4090.sh tests/test_mmsu_inference_probe.py audit_reports/mmsu_inference_review_validation.json
git add -p -- RM.md
git diff --cached --stat
git commit -m "Audit MMSU prompt differences and add training-logit probe"
git push
```
