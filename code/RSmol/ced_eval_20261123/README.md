# 2026-11-23：三循环MeSH和修复后的audio7_read2，八项4090评测

本地Windows维护代码，经GitHub同步到远程Linux。本入口只提交用户这次指定的两个checkpoint，每个四项，共8项。
完整路径固定在 `targets.json`，不搜索latest，不默认其他checkpoint已经训练完成，也不自动重投上一批19项。

| 别名 | 训练run（前缀均为 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/`） | 架构 |
|---|---|---|
| mesh_71437 | `mizar_s1_ced_small_gbs128_text71437_mesh_3090/formal_20261121_150120579947686` | 28物理层、56次执行、三循环、六槽、八个write/read router |
| audio7_read2 | `mizar_s1_ced48_audio7_read2_lora_r8_gbs128_text71427_3090/formal_20261120_111448563391250` | 28物理层、42次执行、双循环LoRA r8、CED4/8七槽read0/1 |

每个run后接 `/formal/checkpoints/model--step-34626.ckpt`，同run的 `/formal/runtime.json` 自动绑定。
四项为 ReasonAQA5000、MMAU test-mini、MMAR、MMSU。全部4090单卡、8CPU、32G、单节点，原镜像与mellow_c8204d8。

## 本次修复与三循环适配

原read2失败发生于完整音频checkpoint加载后的结构审计：评测器向真实历史函数
`validate_audio_read2(model)` 传入 `fresh=False`，Python在进入函数体前报TypeError。
已在 `text71427_ced_eval/runtime.py`、`ced_eval_20261122/runtime.py` 和本次入口修正为只传model。
普通read2校验本来就检查训练后router的数量、形状和存储独立性，没有初始化零值检查。
ar6及wfuse各自支持fresh的调用保持原语义，不通过捕获/吞掉TypeError跳过审计。
训练源码和固定Git快照不改；full state_dict严格覆盖、dtype/finite/绑定、冻结CED与资产一致性等检查保持。

新三循环使用真实 `MizarText7_14x3_7MeshForCausalLM` 和本路线训练loader，核对56次执行：
`0..6, 7..20, 7..20, 7..20, 21..27`。
六槽memory、四组write/read、八个独立Linear(576,6)、16张量/27,696参数，四次路由边界在执行7/21/35/49层后。
保留训练后的所有28层与router；不转换、重新初始化或恢复优化器/RNG/数据游标。memory每次完整forward重建。
read2完整保留A4/A8、两轮LoRA的392矩阵、每轮98次投影和训练后的两个七槽audio read router。

只读hooks不改forward、权重或激活值，仅记录形状、执行顺序和计数。
CPU预检新增从固定源码解析校验函数签名，申请GPU前检查调用接口；实际模型/张量/forward仍在计算进程严格审计。
`READY_FOR_GPU`仅表示CPU预检通过，`SUBMITTED`仅表示vc接受提交。

## 相同评测设置

八项复用既有正式数值流程和评分。训练forward不改，生成/预处理/评分核心函数AST与上一版一致。

- FP32、eval、seed1234、TF32关闭、无autocast，逐题推理。
- mono32k，fixed20中心裁剪/右补零，再拆成10+10；训练Mellow在CED前拼回原整段。
- audio126+SEP1+prompt256=prefix383；首答案取logit382。原EOS后缀、PAD `!`、分词/截断/右补齐顺序保持。
- attention_mask=None、无KV cache、每token全序列重算、argmax、最多300 tokens、首EOS停止。
- read2每题编码一次音频，每次LM forward显式传入本题A4/A8；不跨题复用context。
- ReasonAQA沿用seed1234固定5000 manifest及完整答案exact；MMAU/MMAR各1000，原始输出交原官方scorer。
- MMSU继续最新紧凑全小写，如 `which country? a) india b) none c) china d) japan`，无only-A/B指令，四槽原顺序保留。
- MMSU评分仍为 `mmsu_answer_text_contains_v2`及原固定官方scorer、完整5000分母；GT不进入推理输入。
- 完整覆盖校验、raw留存、坏行拒绝、同身份断点续跑保持。未完成生成不产生完整benchmark分数。

## 固定源码与独立输出

三循环训练源码固定 `75dcda3a7bcd8765044b14ecd4e408fe33dfdda6`，code_digest为
`ea3677917df06121dfb0cbbc6254f2cd5b707ea4787bfd3eb9e5f4daa1fece9d`。
read2固定 `8a9f1f76722e77cba4be429a578c63fdcf6ee527`，code_digest为
`f0ab5b6556c4a2144f2dd86bc1d8ad13cd79c60c4ed0da81fec790ed40efaa1d`。
远程runtime/ckpt仍须与各自源码摘要精确相同，不能以目录名替代合同核对。

仅通过本地 `git archive` 取所需训练后端、相应x2/x3共享模型/合同和release manifest，
存放 `code/RSmol/log/ced_eval_20261123_sources/<revision>/`，逐文件SHA及源码库存固定校验。
新缓存不扩写或覆盖上一批按revision缓存的不同文件清单。新旧模型共享包导入隔离并核对实际路径。
当前评测器与MMSU紧凑小写dataset始终来自当前checkout，不从历史Git恢复旧题面。

新输出：`<run>/eval/ced_eval_20261123_v1/<task>/model--step-34626/`。
task为 `reasonaqa5000`、`mmau`、`mmar` 或 **`mmsu_compact_lowercase_v1`**。
保留原失败/半程输出，本次八项首次提交从新目录生成；同批次后续中断可在身份不变时续跑。
修复代码改变旧评测器的源码身份，因此本轮使用本页新入口，不把旧目录的run_config手工改成新协议。

## 远程提交：八条单项命令

将本轮全部文件通过GitHub同步后，远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# 新三循环MeSH：四项
bash code/RSmol/run_ced_eval_20261123_4090.sh mesh_71437 reasonaqa5000
bash code/RSmol/run_ced_eval_20261123_4090.sh mesh_71437 mmau
bash code/RSmol/run_ced_eval_20261123_4090.sh mesh_71437 mmar
bash code/RSmol/run_ced_eval_20261123_4090.sh mesh_71437 mmsu

# 修复后的audio7_read2：四项重新跑
bash code/RSmol/run_ced_eval_20261123_4090.sh audio7_read2 reasonaqa5000
bash code/RSmol/run_ced_eval_20261123_4090.sh audio7_read2 mmau
bash code/RSmol/run_ced_eval_20261123_4090.sh audio7_read2 mmar
bash code/RSmol/run_ced_eval_20261123_4090.sh audio7_read2 mmsu
```

也可一次提交8项，**与上面八条择一，避免重复排队**：

```bash
bash code/RSmol/run_ced_eval_20261123_4090.sh all
```

只提交某模型的全部四项，也可分别使用：

```bash
bash code/RSmol/run_ced_eval_20261123_4090.sh mesh_71437
bash code/RSmol/run_ced_eval_20261123_4090.sh audio7_read2
```

参数预览不加载权重/不提交；CPU预检只读合同/权重header，不申请GPU：

```bash
bash code/RSmol/run_ced_eval_20261123_4090.sh all --check-args
bash code/RSmol/run_ced_eval_20261123_4090.sh all --preflight-only
```

两模型整体提交时每个checkpoint只预检一次。某目标预检失败则其四项不申请GPU，另一个目标继续；
vc某次提交失败也继续其余任务，整体返回非零，详细错误保存在提交报告。
不要在同一任务仍排队或运行时重复执行提交命令。没有增加额外GPU smoke作业。

## 日志与结果

```bash
ls -lt code/RSmol/log/ced_eval_20261123/
```

- `<时间>_<pid>.submission.json`：预检结果、checkpoint/输出绑定、完整vc参数、stdout/stderr和返回码。
- `ced23-<target>-<task缩写>-<hash>.JOB.log`：GPU stdout/stderr，对实际日志文件用 `tail -f`。
- 各新输出目录 `evaluation_report.json`：最终状态；`model_audit.json`：实际源码、训练架构及forward检查；
  `raw_generations.jsonl`：逐题原始生成；另有各benchmark评分产物和 `run_config.json`。

中断后确认原作业退出，再执行同一条单项命令。源码、数据、scorer、checkpoint及协议身份相同才续跑；
不完整raw末尾按原存储逻辑恢复，完整坏行拒绝；完整输出可复用raw重评分。
同一benchmark两份完整结果可用本包compare作只读数据/协议/原始记录检查和CSV汇总，不跨benchmark混用指标。

## 同步清单与本地验证

本次新增本包、`code/RSmol/run_ced_eval_20261123_4090.sh`、`tests/*ced_eval_20261123*.py`、验收报告与RM记录；
同时修改两个旧runtime中的read2调用，不修改训练目录、共享模型、数据/scorer或上一批targets清单。
本包复用当前 `mizar_s1_ced_reasonaqa5000_eval`、`mizar_s1_ced_mmsu_eval`、`text71427_ced_eval/observers.py`及既有scripts，
这些依赖和此前紧凑小写MMSU改动必须已同步；不能只拷贝本页Shell。

`python -B tests/validate_ced_eval_20261123.py` 保存 `audit_reports/ced_eval_20261123_validation.json`。
除实际Git快照、合同、Bash参数链、模拟vc、八项完整尺寸工作流/续跑与题面/评分检查外，
本次直接执行真实 `audit_model`、固定源码中实际router/LoRA/参数审计函数，覆盖原报错与两个旧入口。
无Torch时这些结构测试使用显式张量/模块元数据夹具，不以模拟审计返回值替代真实函数。
另设可选真实CPU小模型forward/重载/非初始参数保留测试，缺依赖明确跳过；bulk fsync在全尺寸合成工作流中模拟。
本地验证不冒充远程生产权重加载、4090显存或GPU数值PASS；本工具不自动commit/push或提交真实作业。

本轮最终联合39项：37通过，2项真实CPU张量测试因缺Torch/Transformers/Safetensors跳过。
首轮新包30项全通过；随后联合验收的一项历史整文件SHA守卫拒绝了本轮授权的read2修复，
已用 `compatibility_reference.json` 记录精确前后SHA（其余函数AST相同），保留历史报告后完整复验通过。
2927份其他已有生产源码摘要保持，两个旧runtime各新增5行；Python3.10、两份Bash入口、文档5块Bash命令及whitespace检查通过。
