# MeSH 5+10×4+5 MMSU 三题诊断

针对 `recursive_mesh_5_10x4_5` 在 MMSU 约 1%、原始回答重复退化的问题。
这是独立的真实权重对照工具，不生成 benchmark 分数，也不修改生产评测、训练或模型源码。
用户贴出的三条回答本身没有正确选项文本，单独调整答案解析无法修复这些生成。
目前本地源码检查没有发现漏循环或漏 router；根因仍待真实 checkpoint 对照确认。

用户随后贴出的服务器 `recursive_forward_audit` 显示 `checked_forwards=649923`、
`use_cache=false`，最后一次完整执行 `0..4,(5..14)×4,15..19` 及位于
5/15/25/35/45 的五组 write/read，GPU 为 RTX 4090。该记录支持已执行完整四循环，
`last_branch_events=[]` 符合本路线没有额外音频中间层注入的结构。
这段审计未附 checkpoint/runtime/源码身份，仍需与异常输出目录及诊断结果对照。

## 固定模型与输入

- checkpoint：`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_recursive_mesh_5_10x4_5_5090/formal_20261103_093224843011222/formal/checkpoints/model--step-34626.ckpt`
- 同 run 的 `formal/runtime.json`，训练 geometry `(8,4,4)`。
- 直接复用原严格 loader、模型及 tokenizer；不选择 latest。
- 精确使用用户报告的三个 `accent_identification` ID，见 `probe.DEFAULT_IDS`。
- 同一解码音频比较四次生成：原 MMSU 题面、相同题面小写、原 MMAU 紧凑格式小写，以及通过原 MMAU/MMAR runner 对相同小写题面的生成。
- 额外用原 MMAU bytes 解码器读取相同音频，比较 mono/resample 后波形是否逐位相同。

所有生成保持 FP32、eval、inference mode、seed1234、TF32 关闭、mono32k/fixed20、
prompt256/prefix383、无 KV cache、逐 token 完整 forward、greedy300 和首次 EOS 停止。
MeSH 的 50 次逻辑层调用、4 次循环和 10 次 router 调用由原 trace hooks 实时核验。
原题面保留大小写；其他题面仅供诊断，不能计作同协议的正式 MMSU 结果。

## 远程提交

先将代码经 GitHub 同步到服务器，在远程仓库根执行下面的命令。一个 4090 作业最多生成 12 条回答。
完整 `-j` 使用短名称，包含时间/PID 后缀后仍小于 60 字符。诊断目录必须不存在，拒绝覆盖旧结果。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
tag="$(date +%m%d%H%M%S)_$$"
diag="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_recursive_mesh_5_10x4_5_5090/formal_20261103_093224843011222/eval/mmsu_probe_${tag}"
mkdir -p log
vc submit -p pdgpu-4090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 -j "mx4-mmsu-probe-${tag}" \
  -d "$PWD" JOB=1:1 "$PWD/log/mx4-mmsu-probe.${tag}.JOB.log" \
  --cmd "bash mizar_mmsu_meshx4_diagnostics/run.sh --output-dir '$diag'"
printf 'diagnostic report: %s/diagnostic_report.json\n' "$diag"
```

参数预览（不读权重、数据或写输出）：

```bash
bash mizar_mmsu_meshx4_diagnostics/run.sh --output-dir /tmp/new_meshx4_probe --check-args
```

## 查看结论

`diagnostic_report.json` 包含生产模型/数据/协议身份、诊断源码 SHA、真实模型 audit、
每题各题面的原始回答、token IDs、prompt tokens/mask、波形摘要和音频解码对照。

- 相同小写输入的两个 runner token IDs 不同：继续检查包装调用或模型状态。
- 音频与 runner 一致，原题面退化而小写题面恢复：提示大小写敏感，需进一步验证。
- 只有紧凑题面恢复：提示格式/指令分布敏感，不能仅归因于大小写。
- 所有题面仍退化：需继续核对真实权重、初始化资产、服务器版本，以及 MMAU/MMAR 原始输出。

`PASS` **仅代表**三题覆盖完整、相同输入的 runner token IDs 一致且音频解码一致，
不代表答案正确或 1% 问题解决。执行异常为 `FAILED`，对照不一致为 `DIFFERENCE_FOUND`。
三题不足以推断完整 MMSU 准确率；该工具不调用 scorer。

要核对异常正式作业，还需其 `model_audit.json`、`run_config.json`（或至少
`recursive_forward_audit`、checkpoint、eval_variant 和 protocol 字段）。如果最终确认题面需调整，
应使用新协议和新目录，并为其他架构安排同协议比较，不能覆盖现有结果。

本地验证使用合成样例及 NumPy/模型 fixture；本地没有生产权重和 Torch/CUDA，不能据此宣布真实模型通过。
