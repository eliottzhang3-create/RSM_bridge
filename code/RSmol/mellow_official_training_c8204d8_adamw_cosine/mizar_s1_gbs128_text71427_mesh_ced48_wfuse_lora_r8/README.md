# 7–14×2–7 MeSH：CED4/8 write-value 拼接融合 + 两轮独立 LoRA r8

2026-11-20，从 `mizar_s1_gbs128_text71427_mesh_ced48_listen_lora_r8` 完整复制隔离。
本地Windows维护代码、GitHub中转；生产权重、数据、输出及GPU执行在远程Linux。
母本及原入口/共享模型不改。`parent_snapshot.json` 与 `protected_sources.sha256`
保护142份已纳入Git的母本/参考/共享文件。
`parent_reference_audit.json` 仅为母本历史，当前验证见 `reference_audit.json`。

## 正式提交

本地commit/push之后，在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_wfuse_lora_r8_3090.sh
```

登录节点只做原CPU资产检查/index准备，再通过vc submit申请
pdgpu-3090 / 单节点8GPU / 64CPU / 256G，原镜像及mellow_c8204d8环境。
直接完整formal，无新增自动smoke/审查/人工放行条件，完整job名含后缀小于60字符。
默认文本来源（只继承权重，不读training_state.pt）：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/formal_reinit1000_lr6e4_20261118_112042_3963870/checkpoint-009244
```

CED仍为 /hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small，无需重转文本。
显式严格加载全部FP32 safetensors/六个原routers/绑定embedding-head，核对完成标记/config/源层映射，
再注册融合矩阵和LoRA；再次核对继承tensor值与Parameter对象不变。音频optimizer等从step0新建。

## 精确模型合同

28物理层/42次执行：0..6、7..20、7..20、21..27。
每完整forward重建五槽memory [embedding,0,0,0,0]，覆盖全部多模态位置。
只有原六个write/read router，不保留listen router、音频gate或额外memory写入。

CED冻结/eval，一次编码提取final和block4/8完整输出（final norm前）。
[B,504,384] -> [B,2,4,63,384] -> [B,126,4,384] 保持时频对应。
主mapper仍为官方freq_merge_126；两个独立MLP仍为
LN1536 -> Linear1536,576 -> GELU -> Dropout0 -> Linear576,576 -> LN576。
主mapper及MLP在CED的no_grad范围外训练。

新增两套独立Linear(1152,576,bias=False)：

```text
V_audio[t] = W_f @ concat(H_audio[t], A_layer[t])
W_f initially = [I_576, 0_576]
V = concat_positions(V_audio, H[:,126:])
```

hidden在前、CED MLP输出在后；整个矩阵训练。融合层不加LN/非线性/dropout/额外残差。
只替换write value，不原地覆盖prefix/core或router query。

| 原交互 | write/read query（原规则） | 写入value |
|---|---|---|
| prefix后，边界7 | prefix output | prefix output前126位置与A4融合 |
| 第一轮后，边界21 | 第一轮input | 第一轮output前126位置与A8融合 |
| 第二轮后，边界35 | 第二轮input | 第二轮output原样 |

SEP、prompt、answer/PAD的value保持。槽概率仍由原router计算，M += slot_probability * V后执行原read。
后续query可经memory自然受音频影响，其来源和计算规则保持原样。

融合矩阵1,327,104参数；两个MLP2,443,776；两轮LoRA2,279,424；
相对普通71427 CED+MeSH新增合计6,050,304，比listen母本多1,320,180。
原六router仍12张量/17,310参数。

LoRA源码保持母本：仅物理7..20的q/k/v/o/gate/up/down，两轮独立A/B，
r8/alpha16/scale2/dropout0，392矩阵，A Kaiming/B零。基础W共享且全量训练，前后7层无LoRA。
新模块初始化隔离CPU RNG；逐调用finally清理轮次，拒绝KV cache/decoder gradient checkpointing。
每次full-prefix forward显式传(A4,A8)，不缓存跨样本音频状态。

## 初始化等价与梯度审计

初始[I,0]使value等于原hidden，零LoRA B使LoRA增量为零。
同一文本checkpoint、同一主mapper/输入的初始模型数学上等价于原五槽CED+MeSH。
新矩阵更新后不要求训练轨迹相同；不宣称已验证远程GPU逐位一致。

首步Encoder半矩阵可以获得任务梯度，但因其初值为零，两个MLP首步任务梯度应为零；
零LoRA B时A的任务梯度同样为零。全部分支仍实际执行，不按零权重跳过计算图。
first_gradient_rank<N>.json分别记录融合矩阵两半和原六router的梯度。
沿用母本32次真实更新的累计解锁检查，角色改为两MLP/两fusion/两轮LoRA A/B，状态随checkpoint恢复。
任务梯度和AdamW weight decay引起的参数变化分开解释。

metrics的fusion记录hidden/MLP/fused RMS、value变化RMS及比例、
Encoder半矩阵L2、hidden半矩阵偏离单位阵的L2；保留两轮LoRA trace。

## 数据、训练与资源

| 项目 | 合同 |
|---|---|
| 数据 | hybrid1,477,492行；原音频mapping、MusicCaps及全部SpeechCraft splits |
| 音频/文本 | fixed20中心裁剪；原S1 EOS/PAD/截断；126audio+1SEP+256prompt+300answer=683 |
| loss | answer-only shifted CE；rank-local有效token mean -> GA4均值 -> DDP均值 |
| batch/horizon | 8GPU × micro4 × GA4=GBS128；3epochs；11542更新/epoch；总34626；warmup1731 |
| maxLR | LM/原router3e-4；主mapper/MLP/fusion/LoRA6e-4 |
| optimizer | 原AdamW(.9,.999)/eps1e-8；矩阵WD.01、bias/低维WD0 |
| 调度/精度 | 原LambdaLR floor.1/GradNormTracker；FP32、原GradScaler、无autocast/TF32 |
| RNG | seed1234，rank1234+rank |

CPU index后共享音频立即训练，默认4workers后台复制，完整后optimizer边界8rank MIN共同切换。
tmpfs为 /dev/shm/mizar_s1_gbs256_text71427_ced48_wfuse_lora_r8_<token>；
gbs256为母本安全守卫前缀，实际GBS128。先stop/reap producer，再owner/token清理。
完整epoch更新循环仅分支诊断不同；CED/mapper、data/staging与LoRA源码逐字节保留。

## 输出和完整恢复

默认run root：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced48_wfuse_lora_r8_gbs128_text71427_mesh_3090/formal_<timestamp>
```

实际训练在其formal/下。保存5771/11542/17313/23084/28855/34626步，
formal/checkpoints/model--step-N.ckpt，最后另存model--epo-3.ckpt。
保存全部模型/融合矩阵/MLP/LoRA、AdamW/scheduler/Scaler/GradNorm、8rank Python/NumPy/CPU/CUDA RNG、
下一数据游标和累计梯度审计状态。恢复严格绑定新路线代码/资产/数据/配方，
校验具名optimizer库存/顺序/LR/WD、moments和模型张量shape/dtype/finite，不重置融合矩阵。
旧listen/其他路线checkpoint不接受full resume。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced48_wfuse_lora_r8_gbs128_text71427_mesh_3090/formal_REPLACE_WITH_ACTUAL_RUN
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_wfuse_lora_r8_3090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt"
```

参数沿用母本：--text-model-dir/--ced-model-dir/--resolved-paths/--index-cache-root/--stage-workers/--output-dir。
fresh输出须为空；完整恢复沿用原run root。既有evaluator尚未登记新合同，后续需单独适配。

## 验证与Git同步

```powershell
python -B tests/validate_mizar_s1_text71427_ced48_wfuse_lora_r8.py
```

报告audit_reports/mizar_s1_text71427_ced48_wfuse_lora_r8_validation.json。
本次便携验收103项执行通过，21项真实张量检查因缺依赖跳过；不代表生产GPU PASS。
覆盖实际源码NumPy/有限差分、原query/value边界、初始化/RNG、优化器/梯度审计、
Bash+模拟vc、CPU index/后台复制/失败清理及原路线回归。
真实CPU autograd/严格HF加载/683位置反传/完整checkpoint下一步对照另设测试；
缺Torch等依赖明确skip，不记作GPU PASS。
完整依赖环境可手动加--require-tensors禁止skip，不是formal前置。

本地同步示例，先检查并仅纳入本次文件；RM有并行修改时按块暂存：

```powershell
git add -- code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced48_wfuse_lora_r8 code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_wfuse_lora_r8_3090.sh tests/test_mizar_s1_text71427_ced48_wfuse_lora_r8_behavior.py tests/test_mizar_s1_text71427_ced48_wfuse_lora_r8_route.py tests/test_mizar_s1_text71427_ced48_wfuse_lora_r8_shared_start.py tests/test_mizar_s1_text71427_ced48_wfuse_lora_r8_tensors.py tests/validate_mizar_s1_text71427_ced48_wfuse_lora_r8.py audit_reports/mizar_s1_text71427_ced48_wfuse_lora_r8_validation.json
git add -p RM.md
git diff --cached --stat
git commit -m "Add isolated CED4/8 write-value fusion MeSH training"
git push
```

本地实现与合成验证不能证明生产权重加载、3090显存、8GPU训练或真实8rank恢复通过。

