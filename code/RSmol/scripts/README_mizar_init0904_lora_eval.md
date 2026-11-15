# init0904 串行 CED8→4、双循环 LoRA r8 评测

本地 Windows 维护代码，经 GitHub 同步到远程 Linux。权重、数据与 GPU 作业均在远程；本次只扩展评测注册与提交映射，未修改训练实现或训练合同，也未代为提交作业。

## 唯一目标

- 训练启动器：`code/RSmol/run_mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_5090.sh`
- 训练目录：`code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8`
- 评测 variant：`recursive_5_10_5_init0904_layer8_4_audio_reinject_loop_lora_r8`
- 训练 route：`ced_small_mizar_s1_fixed20_gbs256_init0904_layer8_4_audio_reinject_loop_lora_r8_v1`
- run：`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_gbs256_5090/formal_20261113_221451337688842`
- checkpoint：该 run 的 `formal/checkpoints/model--step-17313.ckpt`
- runtime：同 run 的 `formal/runtime.json`；训练 geometry 必须为 `8,8,4`。

MMAU/MMAR 新增独立的默认 run 映射。MMSU 原入口从共享注册表动态识别新路线，继续要求显式 run、step 与 geometry。三个入口均按精确文件名选择 step17313，不扫描或自动挑选 latest，因此其他中间 checkpoint 不参与选择。已有环境变量/CLI 覆盖机制仍然有效；以下命令假定未设置其他目标的 `RSMOL_MIZAR_CED_*` 覆盖变量。

## 架构与完整加载

复用现有非 MeSH 串行8→4 r8路线的全部前向设置。只改变路线身份、实际训练源码目录与 run；模型及生成函数保持原样。

CED 每样本编码一次，取得最终 E126 和完整 block4、block8 在最终 norm 前的输出；独立 MLP 产生 A4/A8。接口始终传入 `(A4,A8)`，实际 LM 在逻辑索引5执行 input residual→CED8 residual，在15执行 input residual→CED4 residual。CED gate 读取 input residual 之后的 hidden；残差仅作用于前126个音频位置。原始 E 的共享投影每次完整 forward 计算一次。

20个物理层执行30次：`0..14,5..14,15..19`。中间物理5..14的七种投影使用两个独立的 LoRA A/B集合，r8、alpha16、scale2、dropout0；280个矩阵、1,628,160个 LoRA 参数；连同串行音频分支总共4,406,019参数。执行审计继续检查每 forward 140次 LoRA投影调用、每轮70次、轮次清理、注入顺序及禁用缓存。

模型由该训练目录构建，严格核对 checkpoint/runtime合同、代码/模型 SHA、训练几何和分支合同，随后以 `strict=True` 加载完整音频训练 `state_dict`（支持统一 DDP 前缀）。训练最终值覆盖所有初始化值；不重置 gates、MLP、LoRA 或 mapper。文本初始化来源为 `stage4_5_10_5/formal-maxlr8e-4-resume4500-20260904_154711/checkpoint-step-009244`，完整音频 checkpoint 才是评测权重来源。

## 保持既有协议

三项评测共同保留 mono32kHz、fixed20中心裁剪/右补零、两个10秒半段重新拼接送入 CED；256 prompt、可见 `!` padding、训练 EOS规则；383 prefix、首答案 logit382；单 GPU/batch1/FP32/eval/inference、seed1234、关闭 TF32；`attention_mask=None`、`labels=None`、`use_cache=False`、每 token完整重算、argmax、最多300新 token、首次 EOS 停止。原始大小写、空白、EOS 原样保存。

- MMAU test-mini：完整1000条，原顺序和固定选项，沿用原小写题面及官方 scorer。
- MMAR：完整1000条，沿用原小写题面及官方 scorer。
- MMSU：完整5000条，原三 shard 顺序及空选项位置，保留官方题面大小写和 A/B/C/D 指令；GT和类别不进入模型。继续使用现有 `mmsu_answer_text_contains_v2`，仅在评分副本处理终止 EOS、首选项标签、Unicode casefold和空白，完整 GT答案文本按词边界匹配。生成字母不决定命中；仅字母且无答案文本仍算错。命中映射到原正确选项字母、未命中映射为空，由原固定 SHA官方 scorer 计入完整5000分母。原始预测、解析审计、normalized评分副本及报告分别保存。

MMSU的既有协议与 MMAU/MMAR 保持各自原样，未为本次路线统一改写题面或评分。

## 远程提交

先在本地将本次代码提交并推送到你使用的 GitHub 分支，再在远程拉取。以下三次提交分别申请 `pdgpu-5090 / 8 CPU / 32G / 1 GPU / 1 node`，沿用原容器和 `mellow_c8204d8` 环境。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_init0904_layer8_4_audio_reinject_loop_lora_r8 \
  bash run_mizar_s1_ced_small_mmau_eval_5090.sh

RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_5_10_5_init0904_layer8_4_audio_reinject_loop_lora_r8 \
  bash run_mizar_s1_ced_small_mmar_eval_5090.sh

bash run_mizar_s1_ced_small_mmsu_eval_5090.sh \
  --eval-variant recursive_5_10_5_init0904_layer8_4_audio_reinject_loop_lora_r8 \
  --run-root /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_gbs256_5090/formal_20261113_221451337688842 \
  --expected-step 17313 --expected-geometry 8,8,4
```

结果在该 run 的 `eval/mmau_test_mini_step17313/`、`eval/mmar_step17313/`、`eval/mmsu_step17313/`。沿用既有身份校验、独立目录、进程锁、原始生成保存与断点续跑机制。

## 本地验证与源码保护

```powershell
python -B tests/validate_mizar_s1_ced_mmsu_eval.py --report audit_reports/mizar_init0904_lora_eval_validation.json
```

该验证入口已加入本路线测试和现有 MMSU v2评分回归。新增12项无生产权重回归覆盖实际合同/错误身份拒绝、严格加载顺序、串行注入/双 LoRA执行 hooks、MMSU每样本特征对传递、EOS/300token协议、1000+1000+5000合成流程与续跑，以及三个实际 Bash入口的模拟提交。另有5项真实小模型 tensor检查，覆盖逐步logits/token一致、首答案382对齐、hooks不改结果、样本隔离、完整最终权重加载不重置与缺 LoRA键拒绝；缺少 Torch/Transformers 时明确跳过。测试结果详见上述 JSON报告；不代表远程真实 checkpoint 或完整 benchmark 已执行。

历史 `reference_audit.json` 保持原样。`mizar_selected4_eval/init0904_lora_reference.json` 单独记录三个注册文件的扩展前后 SHA、精确新增块及31份本路线训练源码/启动器 SHA；审计在内存剥离精确新增块后继续验证此前的全部函数/注册/提交分支，并检查训练文件未改动。生产合同和权重加载检查未放宽。
