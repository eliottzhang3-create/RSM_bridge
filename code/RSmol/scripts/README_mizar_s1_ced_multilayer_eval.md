# 二十四条 CED 交互/MeSH/官方循环路线的 MMAU test-mini / MMAR 评测

两个既有提交入口 `run_mizar_s1_ced_small_mmau_eval_5090.sh`、
`run_mizar_s1_ced_small_mmar_eval_5090.sh` 支持以下原十七个及文末新增七个
`RSMOL_MIZAR_CED_EVAL_VARIANT`。新增实现位于训练目录外：

```text
scripts/evaluate_mizar_s1_ced_multilayer_benchmark.py
scripts/mizar_s1_ced_multilayer_eval_runtime.py
scripts/evaluate_mizar_s1_ced_multilayer_benchmark.sh
```

训练源码、共享递归实现和原有普通 CED / MeSH / 四条非 MeSH evaluator 保留。
默认不指定 variant 时仍为 `base`。每条新路线直接导入自己的实际训练模型，
完整严格加载音频训练 checkpoint；不重新实现模型交互，也不修改训练 forward。

## 路线与默认产物

| variant | contract 版本 | 新增分支参数 | 模型上下文与行为 |
|---|---|---:|---|
| `recursive_5_10_5_layer36912_crossattn` | v2 | 4,892,741 | 四层原始 CED grids + 固定 prompt mask；逻辑索引5、15执行文本 cross attention，保留固定γ=0.3 |
| `recursive_5_10_5_layer36912_query32_crossattn` | v1 | 6,241,733 | 四层 grids 各压缩为32个摘要后交互；共享 learned queries / compression attention，保留固定γ=0.3 |
| `recursive_mesh_layer4_8_inject` | v1 | 2,444,930 | 原顺序 `(A4,A8)`；在原 memory read₀、read₁之后分别注入音频位置 |
| `recursive_mesh_layer4_8_listen` | v1 | 2,450,700 | 原顺序 `(A4,A8)`；前两次边界执行原 write → 新 listen → 原 read，保留两个新增 routers |
| `recursive_mesh_layer36912_listen` | v2 | 3,552,788 | 四层 grids + prompt mask；四路 sigmoid 聚合、逐音频 token tanh 幅度 gate、两次 memory listen 写入，无固定γ或/4缩放 |
| `recursive_mesh_layer4_8_audio7slot_split3` | v1 | 2,468,010 | `(A4,A8)` 初始化音频七槽的槽5/6；文本五槽；12个独立 routers，三次普通读写，无直接残差注入 |
| `recursive_mesh_layer4_8_audio7slot_split3_inject` | v1 | 2,469,164 | 相同七槽结构；复用 A4/A8，在前两次 read 后分别执行门控残差注入 |
| `recursive_mesh_layer8_4_inject` | v1 | 2,444,930 | 接口仍为 `(A4,A8)`；模型内部在逻辑边界5、15的 read 后先注入 A8，再注入 A4 |
| `recursive_mesh_layer4_8_audio7slot_split3_prompt_mix` | v1 | 2,802,093 | 有效 prompt 输入 embedding 池化并 softmax 得到三权重；E_mix 作音频 prefix，三份加权 E/A4/A8 作为七槽 memory 种子 |
| `recursive_5_10_5_layer8_4_inject_audio_reinject` | v1 | 2,777,859 | 非 MeSH；每次 LM forward 一次共享 input 投影，边界5/15先 input 残差，再分别 A8/A4 残差 |
| `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8` | v1 | 4,406,019 | 同样的非 MeSH 串行注入；中间物理层5..14的七种投影分别使用两轮独立 LoRA，基础 W 共享；rank8/alpha16/scale2/dropout0 |
| `recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8` | v1 | 4,406,019 | 非 MeSH；边界5/15先 input 残差，再分别 A4/A8；中间 gate 读取 post-input hidden；双循环独立 LoRA |
| `recursive_mesh_layer4_8_inject_loop_lora_r8` | v1 | 4,073,090 | 原五槽/六 routers，三次 write/read；前两次 read 后注入 A4/A8；双循环独立 LoRA，无额外 input 残差 |
| `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r16` | v1 | 6,034,179 | 非 MeSH 串行 input→A8 / input→A4；两轮独立 rank16/alpha32/scale2/dropout0 LoRA |
| `recursive_textmesh_layer8_4_inject_audio_reinject_loop_lora_r16` | v1 | 6,034,179 | 相同串行音频注入及双 rank16 LoRA；仅位置126起的 SEP/文本参与五槽 memory/六 routers |
| `recursive_textmesh_layer8_4_inject_audio_reinject_loop_modality_lora_r8` | v1 | 6,034,179 | 相同文本 memory/音频串行结构；两轮×音频/文本四套独立 rank8/alpha16 LoRA，按固定位置选择 |
| `recursive_mesh_5_10x4_5` | v1 | 0（无额外音频分支） | 20物理层/50次执行、四轮中段循环、全序列七槽和10个独立 routers，GBS128 |

query32 路线 README 的母本遗留描述不作为架构身份：实际 `contract.py` 是
`ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer36912_query32_crossattn_v1`。
其32个摘要只用于交互分支，输入 prefix 保持383。

十七条训练目录分别是训练根
`mellow_official_training_c8204d8_adamw_cosine/` 下的：

```text
mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_crossattn
mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_query32_crossattn
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_listen
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer36912_listen
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_audio7slot_split3
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_audio7slot_split3_inject
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer8_4_inject
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_audio7slot_split3_prompt_mix
mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject
mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8
mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8
mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16
mizar_stage1_fixed20_gbs256_3epochs_recursive_textmesh_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16
mizar_stage1_fixed20_gbs256_3epochs_recursive_textmesh_ced_small_layer8_4_inject_audio_reinject_loop_modality_lora_r8
mizar_stage1_fixed20_gbs128_3epochs_recursive_mesh_5_10x4_5_ced_small
```

所有远程 run 均位于
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/` 下。
默认 checkpoint 为各 run 的 `formal/checkpoints/model--step-17313.ckpt`，
四循环 MeSH 和 `official_final_s1_5_20x2_5` 则为 `formal/checkpoints/model--step-34626.ckpt`。
runtime 均为同 run 的 `formal/runtime.json`：

| variant | 相对输出根的默认 run |
|---|---|
| `recursive_5_10_5_layer36912_crossattn` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer36912_crossattn_5090/formal_20261030_140459065784533` |
| `recursive_5_10_5_layer36912_query32_crossattn` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer36912_query32_crossattn_5090/formal_20261030_162028139130855` |
| `recursive_mesh_layer4_8_listen` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_listen_5090/formal_20261030_172450546025072` |
| `recursive_mesh_layer4_8_inject` | 无内置默认值；可显式指定已提供的 `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_5090/formal_20261030_172359752928500` |
| `recursive_mesh_layer36912_listen` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer36912_listen_5090/formal_20261101_113133327944502` |
| `recursive_mesh_layer4_8_audio7slot_split3` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_5090/formal_20261101_170211458358358` |
| `recursive_mesh_layer4_8_audio7slot_split3_inject` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_inject_5090/formal_20261102_002211827036990` |
| `recursive_mesh_layer8_4_inject` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer8_4_inject_5090/formal_20261102_112040402999384` |
| `recursive_mesh_layer4_8_audio7slot_split3_prompt_mix` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_prompt_mix_5090/formal_20261102_132232677352730` |
| `recursive_5_10_5_layer8_4_inject_audio_reinject` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_5090/formal_20261102_112121702891801` |
| `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8_5090/formal_20261103_165410438279511` |
| `recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8_5090/formal_20261104_131843931946322` |
| `recursive_mesh_layer4_8_inject_loop_lora_r8` | `mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_loop_lora_r8_5090/formal_20261104_161727489909082` |
| `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r16` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r16_5090/formal_20261105_173212933623545` |
| `recursive_textmesh_layer8_4_inject_audio_reinject_loop_lora_r16` | `mizar_s1_ced_small_gbs256_recursive_textmesh_layer8_4_inject_audio_reinject_loop_lora_r16_5090/formal_20261105_154241055340068` |
| `recursive_textmesh_layer8_4_inject_audio_reinject_loop_modality_lora_r8` | `mizar_s1_ced_small_gbs256_recursive_textmesh_layer8_4_inject_audio_reinject_loop_modality_lora_r8_5090/formal_20261105_173710754949913` |
| `recursive_mesh_5_10x4_5` | `mizar_s1_ced_small_gbs128_recursive_mesh_5_10x4_5_5090/formal_20261103_093224843011222` |

没有默认值的路线不猜测训练时间戳，也不自动选择 latest。未指定 run 或完整路径时，
提交脚本在调用 `vc submit` 前报错。
checkpoint 精确选择指定 step 文件并核对其内部 step；同目录的中间 checkpoint 不参与选择。

## 与原 CED 相同的推理和评分协议

| 项目 | 十七条路线的统一设置 |
|---|---|
| 数据 | MMAU test-mini / MMAR 各完整1000条；原 metadata、音频查找逻辑、原数据和选项顺序 |
| 音频 | mono、32 kHz；fixed20 中心裁剪/右补零至20秒，再分成两个10秒半段；重拼20秒后才进入 CED |
| Prompt | 原固定选项模板、整体小写、训练 tokenize/EOS 后缀；截断/右补 `!` 到256 tokens |
| Prefix | `126 audio + 1 SEP + 256 prompt = 383`，第一个答案 logit 在382 |
| 数值 | 单GPU、batch1、FP32、全部 eval、inference mode；seed1234、TF32关闭及原 cuDNN 设置 |
| 解码 | `attention_mask=None`、`labels=None`、`use_cache=False`，逐 token 完整重算，纯 argmax |
| 停止 | 两个 benchmark 都最多300个新 token，首次生成 EOS 后停止 |
| 文本 | `skip_special_tokens=False`、`clean_up_tokenization_spaces=False`；EOS、大小写、空白原样保存，不 strip 或提取选项字母 |
| 评分 | 原官方 scorer 与预测字段；必须1000条 generated、零 skip，且 scorer 成功确认1000条分母 |
| 资源 | `pdgpu-5090 / 8 CPU / 32G / 1 GPU / 1 node`，原容器及 `mellow_c8204d8` |

协议字典直接复制普通 CED 的 `PROTOCOL`。runner 的15个公共函数与原评测保持 AST 一致，
差别集中在参数解析、模型上下文传递及输出身份。
prompt mask 只用于新增交互分支的有效 prompt 池化，不传给 LM 的 causal attention，
也不屏蔽原来可见的 `!` padding。解码期间 mask 固定为256，不包含新增答案 token。

每个样本只进行一次 CED 编码；原始特征作为局部上下文显式传给每一步 LM。
LM 内的投影、query32 压缩和交互仍按训练实现每次完整 forward 执行。
MeSH memory 由原训练 forward 每次重新创建，不跨 token 完整 forward 或样本持久化。
四层 MeSH listen 的 prefix API 返回 `(prefix, output, None)`；评测侧从
`output["layer_grids"]` 和本次 tokenize 的 `attention_mask` 组装上下文。

普通4/8接口均使用原顺序 `(A4,A8)`，不在 adapter 中交换特征。
split3/split3_inject 的两套 MLP 每样本只计算一次，A4/A8 在所有解码 forward 中复用；
音频 memory 每次从 `[E,0,0,0,0,A4,A8]` 初始化，SEP/文本仍为五槽。
七个槽都参与三次普通 write/read，循环 query 取循环输入、write value 取循环输出。
纯 split3 无旧 listen gate 或边界处直接注入；split3_inject 额外在两次 read 后注入，
gate 读取 post-read 音频 hidden，复用同一对种子特征，无第三次注入或额外直接 memory write。
七槽不增加序列 token，prefix 仍为383。

split3 仅控制初始化：原 slot0 的 weight 复制到槽0/5/6，bias 减 log(3)。
评测保留训练后的独立 router 行和 gate，不重新三等分、不清零、不限制概率、不乘3补偿。

prompt_mix 直接使用训练 `Mellow.generate_prefix_inference` 返回的三份加权种子，
每步通过 `audio_memory_seeds=(αE·E,α4·A4,α8·A8)` 传入训练 LM。
prompt MLP 在每样本构建 prefix 时仅计算一次，使用 tokenizer mask 对实际256个 LM 输入
prompt embeddings 求有效均值，保留有效的 `!` 和 EOS，不含答案/SEP/contextual hidden。
音频 prefix 为三份种子之和，memory 每次初始化为
`[αE·E,0,0,0,0,α4·A4,α8·A8]`，文本仍五槽；不重复加权或做直接残差注入。

串行8→4路线每次 LM forward 从原 mapper 音频 prefix 一次计算共享 A_input，
在边界5先 input gate 残差，再用更新的 hidden 计算 gate8；边界15同样先 input 再 gate4。
真实轨迹为 `[[5,"input"],[5,"layer8"],[15,"input"],[15,"layer4"]]`；两层特征仍显式传 `(A4,A8)`。

四循环 MeSH 使用训练路线的严格 loader 和共享 `recursive_model_5_10x4_5_mesh.py`，
执行 `0..4,(5..14)×4,15..19` 共50次。全序列 memory 为 `[原始输入 embeddings,0,0,0,0,0,0]`，
并非音频七槽/文本五槽分流。边界5/15/25/35/45执行五次原 write/read，10个 routers 各用一次；
无额外音频上下文，prefix API 的 None 规范化为 `{}`，禁止两循环 ablation。

## 严格加载与执行检查

加载前检查 formal checkpoint 的 schema、route、合同/SHA、同 run 的 runtime、
训练源码 SHA、递归模型源码 SHA、资产、指定 step、3 epochs、loss 标识和完整分支合同。
四循环默认检查 step34626、训练 geometry `8,4,4`；其余默认 step17313、`8,8,4`。
prompt_mix 同时检查 audio_listen/prompt_mix 合同；串行路线同时检查 layer_injection、
audio_reinjection、serial_audio_injection 三部分；textmesh 使用其训练合同的 serial_injection 字段，
并额外检查 text_memory。已完成 checkpoint 可用于评测，
不套用会拒绝完成状态的训练 resume 校验，也不恢复 optimizer/scheduler。

用对应路线构建真实训练模型，转 FP32 后 `strict=True` 加载完整 `state_dict`，
包括 CED、原 mapper、LM 和所有 learned query/投影/MLP/gate/router。
仅统一规范化全体键共有的 DDP `module.` 前缀；混合前缀、缺键和多余键都拒绝。
加载后调用实际训练结构 validator，再冻结所有参数并检查全部 eval/FP32。
七槽路线完整加载后显式使用 `validate_audio_listen(lm, fresh=False)`，
允许已训练 router/gate 偏离初始化。底座 `memory_slots=5/router_count=6`
仍描述原文本模型；另行记录并验证实际音频7槽、文本5槽和12个独立 routers。
tokenizer 沿用原训练检查，不增长词表、不 resize embedding。

每次 LM forward 的只读 hooks 检查：

- 四循环以外的路线：20个独立物理层按 `0..14,5..14,15..19` 执行30次，无 KV cache。
- 有直接交互的分支位于逻辑索引5、15；4→8与8→4分别检查；纯七槽 split3 无直接分支事件。
- MeSH 原 routers 和额外 routers 独立且按各训练模型顺序调用。
- 七槽路线在5、15、25各执行音频write/read router、文本write/read router，共12次调用。
- query32 每次 LM forward 一次压缩，输入 query `[4,32,576]`、音频 bank `[4,126,576]`。
- 串行8→4与4→8按各自路线检查共享 input 投影一次及四次 gate 调用顺序，三条内部 trace 与实际 hook 一致。
- 五槽 MeSH 注入路线合并观察 router/gate 调用顺序：边界5/15先 write/read routers 再 A4/A8（或A8/A4）gate，边界25只有 write/read routers。实际 memory 操作另外由真实 tensor 测试核对。
- 双 LoRA 在严格完整加载后、推理冻结前验证70个真实 wrapper、280份独立 A/B矩阵；rank8为1,628,160参数，rank16为3,256,320参数。每次 forward 观察140次全序列投影，分别使用 loop0/1各70次，检查调用后索引复位及禁止 gradient checkpointing。串行音频分支2,777,859参数继续单独保留，不合并或重新初始化 LoRA。
- textmesh 仅文本 memory 为 `[1,5,L-126,576]`；六个 router 输入均为 `[1,L-126,576]`。边界5/15检查 write/read routers→input gate→CED8/4 gate，边界25仅 write/read routers。
- 四套模态 LoRA 为560份独立 A/B、3,256,320参数；140次共享基础投影、280次低秩分支，四套各70次。每步检查音频126、文本L-126及独立轮次/调用次数/索引复位，文本包含SEP、可见padding和所有已生成token。
- 四循环按50次真实层调用及五次边界检查10个 routers，每次 forward 禁止两循环 ablation。

五槽 listen 的 router hooks 观察的是**权重计算调用顺序**：原 write router → 原 read router → 新 listen router。
这与 memory 操作的原 write → 新 listen → 原 read 一致，read 权重在实际 read 之前已经算好。
真实张量测试另外观察实际 `_write` / `_read` 和分支执行顺序。
hooks 不返回替代输出，不修改隐藏状态或 logits；训练数值诊断模式保持关闭。
源码 SHA 不符时应恢复匹配训练时的源码，不能关闭校验加载。
模型构建仍需要原预训练 CED 和文本资产位于合同指定的远程位置。

## 先前已支持的三份默认 checkpoint

通过 GitHub 同步代码后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer36912_crossattn bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer36912_crossattn bash run_mizar_s1_ced_small_mmar_eval_5090.sh

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer36912_query32_crossattn bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer36912_query32_crossattn bash run_mizar_s1_ced_small_mmar_eval_5090.sh

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer4_8_listen bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer4_8_listen bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

## 已支持的四层 listen 与纯 split3

同步代码后，在同一个远程 `code/RSmol` 目录执行：

```bash
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer36912_listen bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer36912_listen bash run_mizar_s1_ced_small_mmar_eval_5090.sh

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer4_8_audio7slot_split3 bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer4_8_audio7slot_split3 bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

每条命令提交一个单卡 full 评测作业，无 smoke 或 tensor 测试自动前置任务。
compute wrapper 激活 `mellow_c8204d8` 并启用 HF 离线模式，不安装或升级依赖。
提交时打印 variant、checkpoint、runtime 和输出路径。

## 2026-11-03：本次四份产物的八项 full 评测

本地修改经 GitHub 同步后，在远程登录节点执行以下完整命令：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer4_8_audio7slot_split3_inject bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer4_8_audio7slot_split3_inject bash run_mizar_s1_ced_small_mmar_eval_5090.sh

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer4_8_audio7slot_split3_prompt_mix bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer4_8_audio7slot_split3_prompt_mix bash run_mizar_s1_ced_small_mmar_eval_5090.sh

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer8_4_inject_audio_reinject bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer8_4_inject_audio_reinject bash run_mizar_s1_ced_small_mmar_eval_5090.sh

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_5_10x4_5 bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_5_10x4_5 bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

四个默认 run/step 已在上表精确绑定，无需传入路径或 step/geometry。
已有 shell 会话若设置过 `RSMOL_MIZAR_CED_*` 路径或 step/geometry 覆盖变量，
需先清除这些旧覆盖值再使用默认命令。每条命令直接提交一个 full 作业，不运行自动前置测试。

## 补充产物：MeSH 8→4 注入的两项评测

已绑定 `formal_20261102_112040402999384` 的 step17313 和同 run 的 runtime，
直接使用已有 MeSH 8→4 adapter。保持 `(A4,A8)` 输入、read 后 A8→A4 注入、
五槽 memory、六个原 routers 和原推理/评分协议。同步代码后，在远程 `code/RSmol` 执行：

```bash
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer8_4_inject bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer8_4_inject bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

本次默认产物绑定经15项合同/上下文与9项Bash假提交检查验证通过；
未读取远程权重或提交真实作业。

## 2026-11-04：非 MeSH 串行8→4 + 双循环 LoRA

新增 variant `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8`。
对应的训练入口为 `run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8_5090.sh`，
训练目录、route 与完整 checkpoint 均为非 MeSH 路线。
默认精确读取 `formal_20261103_165410438279511` 的 step17313 和同 run 的 runtime，
其余中间 checkpoint 不干扰。训练 geometry 为8×8×4，评测仍为单GPU/batch1。

严格加载完整已训练 mapper、A4/A8 MLP、input projection/三个 gates、共享基础 LM
和两轮 A/B 参数。LoRA 覆盖中间十个物理层的 q/k/v/o 与 gate/up/down，
两轮独立选择、全序列生效；音频直接残差仍只修改前126个位置。
20物理层/30次执行、边界5的 input→A8、边界15的 input→A4、每次完整 forward
一次原 mapper prefix 投影，以及原推理/评分协议均保持。

本地修改通过 GitHub 同步后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8 bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8 bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

沿用原队列/资源/环境；每条命令提交一个 full 作业，无自动前置测试。
默认结果分别写入该 run 的 `eval/mmau_test_mini_step17313/`、`eval/mmar_step17313/`。
若当前 shell 留有旧 `RSMOL_MIZAR_CED_*` 路径或 step/geometry 覆盖变量，先清除覆盖值。

统一验证192项实际检查通过，26项真实 tensor 检查因缺 Torch/Transformers 跳过
（当前适配17项、旧回归9项）；Python/Bash语法通过。
覆盖十二条路线/24种完整1000条模拟流程与提交组合、实际 LoRA 训练源码的 NumPy 行为、
非零两轮适配器及串行注入数值、漏调用/错轮/重复调用/索引残留拒绝和原路线回归。
真实 tensor 用例另行覆盖非零 LoRA 的 adapter/direct logits、strict restore 的共享 W 与
两轮 A/B 缺键拒绝、禁用/交换适配器的数值差异、推理冻结后 hooks 及异常后复位。
未读取远程 checkpoint/runtime，也未提交真实评测作业。

## 2026-11-05：非 MeSH 串行4→8与 MeSH 4→8 的双 LoRA 评测

新增两个 variant，均使用各自实际训练目录中的模型、严格完整 checkpoint 加载和原评测协议：

| variant | 默认 run | 训练 geometry / checkpoint step |
|---|---|---|
| `recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8` | `formal_20261104_131843931946322` | 8×8×4 / 17313 |
| `recursive_mesh_layer4_8_inject_loop_lora_r8` | `formal_20261104_161727489909082` | 8×8×4 / 17313 |

非 MeSH 保留 `[[5,"input"],[5,"layer4"],[15,"input"],[15,"layer8"]]`；
共享 input projection 每次完整 forward 一次，两处 CED gate 读取 post-input hidden。
MeSH 每次完整 forward 以原 input embeddings 初始化五槽 memory 的槽0，其余四槽为零，
保留六个独立 routers 和三次 write/read。首组 query/value 为 prefix output；
后两组 query 为对应循环输入，write value 为循环输出。只在前两次 read 后分别注入 A4/A8。

两条路线的七种中间层投影均使用两轮独立 LoRA，rank8/alpha16/scale2/dropout0；
保留共享基础 W/bias、两套已训练 A/B 与原 mapper/MLP/gates/routers，不合并、不重新初始化。
每次完整 forward 检查30次层调用、140次全序列 LoRA 投影及两轮各70次调用，检查索引复位。
每样本编码一次 CED，显式复用同一对 `(A4,A8)`；memory 不跨完整 forward 或样本复用。
新增参数分别为2,777,859+1,628,160=4,406,019和2,444,930+1,628,160=4,073,090。

本地修改提交并推送至 GitHub 后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol

# 非 MeSH 串行4→8 + 双 LoRA：MMAU test-mini、MMAR
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8 bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8 bash run_mizar_s1_ced_small_mmar_eval_5090.sh

# MeSH 4→8 + 双 LoRA：MMAU test-mini、MMAR
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer4_8_inject_loop_lora_r8 bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh_layer4_8_inject_loop_lora_r8 bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

每条命令提交一个完整1000条评测作业，无自动前置任务；沿用原队列/资源/环境。
精确读取指定 run 的 step17313 和 `formal/runtime.json`，其他中间 checkpoint 不参与选择。
两个 run 各自产出 `eval/mmau_test_mini_step17313/`、`eval/mmar_step17313/`，共四项结果。
使用默认命令前应清除当前 shell 中遗留的 `RSMOL_MIZAR_CED_*` 路径或 step/geometry 覆盖变量。
本轮未提交真实作业或读取远程 checkpoint/runtime。

统一验证223项实际检查通过，28项真实 tensor 检查因缺 Torch/Transformers 跳过
（当前适配19项、旧回归9项）；Python/Bash语法通过。
覆盖十四条路线/28种完整1000条模拟流程与假提交、两个非 MeSH 顺序及 MeSH 双 LoRA
实际训练源码的 NumPy 行为、strict 加载接口、错轮/漏调用/重复调用/残留索引拒绝，
以及 gate 先于 write/read routers 调用的拒绝检查。原评测与官方 Mizar 提交分支回归通过。
真实 tensor 测试扩展到三个双 LoRA 版本，检查非零 LoRA 的 adapter/direct logits/token、
首答案对齐、共享 W 和两轮 A/B strict restore、禁用/交换适配器数值差异、异常后复位，
另核对非 MeSH 4→8 的 post-input gate，以及 MeSH 五槽 memory、三次写入、循环输入 query
和两次 post-read gate 的数值与跨样本隔离。

## 2026-11-06：非 MeSH rank16、textmesh rank16 与 textmesh 四套模态 rank8

本次三个默认 run 绑定见上表，均精确读取 step17313 和同 run 的 `formal/runtime.json`。
第二个用户提供的相对 `outputs/RSmol/...` 路径按训练启动器补全到
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/...`。
训练 geometry 仍为8×8×4、三 epochs，评测仍为单GPU/batch1，原数据/推理/评分协议保持。

三条路线均保留每次完整 LM forward 一次原 mapper 音频投影、边界5的 input→CED8、
边界15的 input→CED4及 post-input CED gate，输入接口始终是 `(A4,A8)`。
非 MeSH rank16不使用 memory/router。
两条 textmesh 的音频位置0..125绕过 memory/router，文本从位置126开始，包含SEP、
可见的prompt padding及已生成答案；五槽 memory 为 `[B,5,L-126,576]`，首步为 `[1,5,257,576]`。
slot0取原始文本 input embeddings，其余四槽为零，每次完整 forward 重建。
三组router保留prelude output-query/value、循环input-query/output-value；
末次read只更新文本，第二轮音频输出直接进入末五层。完整序列attention/RoPE保持原模型实现。

中间物理层5..14的七种投影保留共享基础W；两套 rank16/alpha32为280份A/B，
四套模态 rank8/alpha16为560份A/B，总量均为3,256,320，连同串行音频共6,034,179。
四套分别是audio.loop0、text.loop0、audio.loop1、text.loop1，选择依据固定位置边界126；
SEP、padding和新生成token均使用文本LoRA，不依赖token id或prompt mask。
每投影基础W对全序列计算一次，再拼接两种模态的低秩残差；scale2/dropout0保持。
rank16与模态rank8的权重键、矩阵形状和合同严格区分，禁止交叉加载或merge。

同步本地代码到GitHub后，在远程登录节点执行以下六项full评测，无自动前置作业：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol

# 清除以往手工设置的模型路径/step/输出覆盖，使用本次三个默认产物。
unset RSMOL_MIZAR_CED_RUN_ROOT RSMOL_MIZAR_CED_CHECKPOINT RSMOL_MIZAR_CED_RUNTIME_CONFIG
unset RSMOL_MIZAR_CED_ROUTE_ROOT RSMOL_MIZAR_CED_EXPECTED_STEP RSMOL_MIZAR_CED_EXPECTED_GEOMETRY
unset RSMOL_MIZAR_CED_MMAU_OUTPUT_DIR RSMOL_MIZAR_CED_MMAR_OUTPUT_DIR

# 非 MeSH 8→4串行注入，两套rank16
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r16 bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r16 bash run_mizar_s1_ced_small_mmar_eval_5090.sh

# textmesh 8→4串行注入，两套rank16
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_textmesh_layer8_4_inject_audio_reinject_loop_lora_r16 bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_textmesh_layer8_4_inject_audio_reinject_loop_lora_r16 bash run_mizar_s1_ced_small_mmar_eval_5090.sh

# textmesh 8→4串行注入，四套模态rank8
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_textmesh_layer8_4_inject_audio_reinject_loop_modality_lora_r8 bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_textmesh_layer8_4_inject_audio_reinject_loop_modality_lora_r8 bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

输出分别位于各自训练run的 `eval/mmau_test_mini_step17313/` 与 `eval/mmar_step17313/`。
输出身份继续包含模型/数据/评分器/评测代码SHA与协议，身份不同的已有结果禁止混用。
本轮真实远程checkpoint、runtime加载及六项GPU评测需在服务器实际执行。

本地统一验证282项检查通过，32项真实tensor检查因缺Torch/Transformers跳过
（当前适配23项、旧回归9项）；Python/Bash语法通过。
覆盖17路线/34种完整1000条模拟流程与Bash假提交、三条新路线实际训练源码的NumPy行为、
严格合同/源码/加载接口、错轮/漏调用/重复调用/残留索引/错误memory范围/错误模态长度拒绝，
以及全部原评测回归。真实tensor用例另覆盖逐步logits/token、首答案对齐、完整strict恢复、
textmesh memory/query/write/末次音频直通和生成长度增长、四套模态边界与跨rank/架构加载拒绝。
训练时期的两个textmesh快照包含本次获授权修改的两个评测入口、adapter和评测README；
统一验证对此四文件明确记录例外，其余185/117份快照文件仍逐字节验证，并完整执行原训练AST/配方对照。
历史快照文件、训练目录和生产checkpoint/source SHA校验均保持原样；此替代记录写入验证报告。

## 路径与 step 覆盖

替换产物时可用 `--run-root /path/to/actual_run` 等 CLI 参数。标准
`<run>/formal/checkpoints/<file>` checkpoint 路径可推导其 run；对十七条路线，
显式 run 优先，其次是显式标准 checkpoint 所属 run，最后才使用已知默认 run。
非标准 checkpoint 路径应同时提供 `--checkpoint`、`--runtime-config`、`--output-dir`。
新路径解析兼容 `--option value` 和 `--option=value`。

保留原有环境覆盖接口：`RSMOL_MIZAR_CED_RUN_ROOT`、`RSMOL_MIZAR_CED_CHECKPOINT`、
`RSMOL_MIZAR_CED_RUNTIME_CONFIG`、`RSMOL_MIZAR_CED_ROUTE_ROOT`、
`RSMOL_MIZAR_CED_EXPECTED_STEP`、`RSMOL_MIZAR_CED_EXPECTED_GEOMETRY`、
`RSMOL_MMAU_DATASET_DIR`、`RSMOL_MMAR_DATASET_DIR`、
`RSMOL_MIZAR_CED_MMAU_OUTPUT_DIR`、`RSMOL_MIZAR_CED_MMAR_OUTPUT_DIR`。
额外 CLI 参数继续传给 runner；默认 step/训练 geometry 按所选路线决定。
CLI `--expected-step`、`--expected-geometry` 优先于环境变量，路径推导前生效；
例如 `--expected-step 10000` 同时改变默认 checkpoint 文件名和输出目录名。
训练 geometry 只校验训练身份，评测始终 batch1/单GPU。

## 结果与续跑

结果默认写入各训练 run 自己的：

```text
eval/mmau_test_mini_step<17313或34626>/
eval/mmar_step<17313或34626>/
```

各项保存 `run_config.json`、`progress.jsonl`、`raw_generations.jsonl`、
`predictions_fixed_order.json`（MMAU）或 `predictions_official.json`（MMAR）、
`official_evaluation.txt`、`evaluation_report.json`。
JOB.log 位于远程仓库 `code/RSmol/log/`，按 variant 和 benchmark 区分。

`run_config.json` 绑定 checkpoint 完整 SHA、runtime、variant/route、数据、scorer、
评测实现源码和原协议。相同身份重新提交会从已落盘记录续跑，不重新推理完成的样本。
身份不同、源码变化或非空无身份目录会拒绝写入，应指定新的独立输出目录。
同一输出目录只运行一个作业。

成功需 `evaluation_report.json` 的 `status=PASS`、1000条 generated / 零 skip，
且官方 scorer 成功确认1000条分母。结构和执行检查在
`model_audit.recursive_audit`、`model_audit.recursive_forward_audit`。
完全复用1000条已有结果时没有新 forward，后者可以是 PENDING / 0。
本地代码验证报告的 PASS 不代表远程权重或 benchmark 评分已经通过。

## 验证与边界

在仓库根目录运行：

```bash
python -B tests/validate_mizar_s1_ced_multilayer_eval.py
```

汇总报告是 `audit_reports/mizar_s1_ced_multilayer_eval_validation.json`。
2026-11-05扩展覆盖十四条路线/28种完整1000条模拟流程与提交组合；
223项实际检查通过，28项真实 tensor 检查因缺 Torch/Transformers 跳过，详见上方本次扩展说明。
2026-11-03扩展覆盖十一条路线/22种完整1000条模拟流程与提交组合，
并执行 prompt_mix、串行8→4和四循环的训练源码行为检查及官方 Mizar 提交分支回归。
177项实际检查通过，23项真实 tensor 检查因缺 Torch/Transformers 跳过
（当前适配14项、旧回归9项），Python/Bash语法通过。
2026-11-02扩展后本地结果：125项实际检查通过，19项真实 CPU tensor 检查
因缺少 Torch/Transformers 跳过（当前适配10项、旧回归9项），Python/Bash语法通过。
八条路线均覆盖合同、上下文、解码、16种完整1000条模拟流程及16种提交组合；
额外运行两条七槽路线和8→4的实际训练源码 NumPy/stdlib 检查，
覆盖初始 memory、三次普通读写、split3 初始化、非零gate的post-read输入和注入顺序。
真实训练结构 validator 的 NumPy参数检查确认已训练router/gate可通过 `fresh=False`，
且不会被重置；它不等同于真实 PyTorch 权重加载。

2026-10-31本地结果：93项实际检查通过，18项真实 CPU tensor 检查因缺少
Torch/Transformers跳过（本次新增9项、旧回归9项），Python/Bash语法通过。
当时涵盖五路线真实合同识别、交叉路线拒绝、strict 加载接口、原协议/公共函数 AST 对照、
显式上下文、NumPy 解码检查、十种1000条模拟流程、续跑、官方 scorer 成败分母、
十种提交组合/路径覆盖及旧 evaluator 回归。

当前有23项真实 CPU PyTorch/Transformers 测试，使用实际训练模型、非零 gates、非零双/四套 LoRA 及偏离初始化的七槽 router 行，
检查 adapter/direct logits 与 tokens、训练683/评测383首答案对齐、strict restore、
跨样本隔离、query32 压缩、七槽种子/读写/循环query和post-read gate，以及 MeSH 实际操作顺序。多数使用缩小 LM 宽度/词表，
保留20物理层、30/50执行和383 prefix；结构检查另外使用生产 hidden576。
新增检查 prompt mask 的有效 EOS/!、加权种子仅计算一次并复用、prompt MLP strict restore、
非零 gate 的 input→CED8/4 串行数值与原始 prefix 投影，以及四循环全序列七槽、全部写入和跨样本重建。
这些测试不用远程权重、CUDA、下载或作业提交，不能替代真实 checkpoint 的 GPU 验证。

具备原有 tensor 依赖的环境可手动强制所有 tensor 测试实际执行：

```bash
python -B tests/validate_mizar_s1_ced_multilayer_eval.py --require-tensors
```

缺依赖或任一 tensor skip 会使该模式失败。本机缺 Torch/Transformers 时明确记录跳过，
不得把 stand-in/NumPy 检查写成真实 Torch 检查通过。
此前三份远程 checkpoint 的历史验证范围不代表本次新增路线；最新范围见下方2026-11-08说明。

## 2026-11-08：新增七条路线（共14项评测）

推理、音频、prompt、解码和官方评分保持原 `ced_small_mizar_s1_fixed20_raw_greedy_v1` 协议。
训练源码和共享模型实现均未修改；新适配器 `scripts/mizar_s1_ced_official_recursive_eval_runtime.py`
仅处理官方训练合同、初始化产物审计和严格加载。原始 `official_mizar` 入口仍保持原用途。

| variant | 默认 run（相对远程 outputs/RSmol） | step |
|---|---|---:|
| `recursive_mesh_4slot` | `mizar_s1_ced_small_gbs256_recursive_mesh_4slot_3090/formal_20261107_203129899582015` | 17313 |
| `recursive_mesh_6slot` | `mizar_s1_ced_small_gbs256_recursive_mesh_6slot_3090/formal_20261106_092419698981503` | 17313 |
| `recursive_mesh_7slot_v2` | `mizar_s1_ced_small_gbs256_recursive_mesh_7slot_v2_3090/formal_20261107_215615398982617` | 17313 |
| `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r4` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r4_5090/formal_20261107_215916042121396` | 17313 |
| `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_modality_lora_r8` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_modality_lora_r8_5090/formal_20261107_215838489409042` | 17313 |
| `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_shared_modality_lora_r8` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_shared_modality_lora_r8_5090/formal_20261107_215949376143529` | 17313 |
| `official_final_s1_5_20x2_5` | `mizar_official_final_s1_gbs128_5_20x2_5_5090/formal_20261107_003434623913704` | 34626 |

前三个 MeSH 是全序列4/6/7槽，三组 write/read、六个独立router；不是音频七槽/文本五槽实验。
七槽v2保留本路线文本checkpoint-009244及BF16文本训练来源校验，音频训练和评测均为FP32。
三条非MeSH串行路线保持逻辑5/15的 input→CED8、input→CED4，显式 `(A4,A8)` 每步传入。
rank4/alpha8共280份A/B、814,080 LoRA参数；Attention共享/MLP模态独立为400份、2,641,920参数。
MLP共享+模态独立为520份、3,655,680参数；Attention scale2，MLP shared及private各scale1，
从相同输入并行相加，基础Linear仅一次；不能按统一scale2或串行分支加载。

官方路线是30物理层/50次执行 `0..24,5..24,25..29`，逻辑5/25串行8→4，
中间20物理层使用两轮r8/alpha16/scale2 LoRA，共560份A/B、3,256,320参数。
审计 `model_dir` 完整初始化manifest，调用GPU-safe训练loader后，严格覆盖训练checkpoint全部1038份state。
不调用CPU转换依赖入口、不以初始化权重替代评测权重，不调用会拒绝已完成游标的训练resume校验。

新增七条训练目录如下，其中两个Attention共享路线使用短名称：

```text
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_4slot_ced_small
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_6slot_ced_small
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_7slot_v2_ced_small
mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r4
mizar_stage1_fixed20_gbs256_recursive_5_10_5_ced_small_in84_loop_attn_shared_mlp_modality_lora_r8
mizar_stage1_fixed20_gbs256_recursive_5_10_5_ced_small_in84_loop_attn_shared_mlp_both_lora_r8
mizar_official_final_s1_gbs128_5_20x2_5
```

本地提交并推送GitHub后，在远程登录节点执行以下命令，提交7×2个完整1000条评测：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol

variants=(
  recursive_mesh_4slot
  recursive_mesh_6slot
  recursive_mesh_7slot_v2
  recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r4
  recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_modality_lora_r8
  recursive_5_10_5_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_shared_modality_lora_r8
  official_final_s1_5_20x2_5
)
for variant in "${variants[@]}"; do
  RSMOL_MIZAR_CED_EVAL_VARIANT="$variant" bash run_mizar_s1_ced_small_mmau_eval_5090.sh
  RSMOL_MIZAR_CED_EVAL_VARIANT="$variant" bash run_mizar_s1_ced_small_mmar_eval_5090.sh
done
```

每条默认只读取表中run的精确step文件和同run的 `formal/runtime.json`，不搜索latest或其他中间checkpoint。
结果位于各run的 `eval/mmau_test_mini_step<step>/` 和 `eval/mmar_step<step>/`。
沿用pdgpu-5090、8CPU、32G、1GPU、1node、mellow_c8204d8；训练在3090不会更改评测资源。
可继续通过既有 `--checkpoint`、`--run-root` 等参数覆盖；覆盖checkpoint会推导其对应runtime和输出目录。

校验入口仍为 `python -B tests/validate_mizar_s1_ced_multilayer_eval.py`，新增合同、加载顺序、
逐步执行、14项提交及覆盖路径检查，既有1000条流程覆盖全部24个variant。
`--require-tensors` 要求所有真实张量用例执行，缺Torch/Transformers时失败，不把跳过记为张量PASS。
本地检查不代表远程生产权重或GPU评测通过；以远程evaluation_report.json和official_evaluation.txt为准。
