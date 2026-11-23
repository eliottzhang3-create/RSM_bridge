# 7–14×3–7 / CED4,8 / audio8 pre-loop read3 / independent LoRA r8

独立正式训练路线，复制三循环 CED-small 母本，迁入双循环 audio7/read2 的记忆与 LoRA 机制。
本地 Windows 仅维护源码，经 GitHub 同步；权重、数据、正式执行和产物在远程 Linux。
原双循环、普通三循环、共享文本模型和既有评测器均不改动。

## 正式提交

代码同步后在远程仓库根执行；不需重新转换文本，不增加 GPU smoke 前置：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced48_audio8_read3_lora_r8_3090.sh
```

登录 CPU 先检查文本完成合同/CED资产并准备或复用 index，成功后才 vc submit。
资源为 pdgpu-3090、1节点8GPU、64CPU、256G；原镜像、
mellow_c8204d8 环境。计算节点 torchrun 8rank，禁止在登录节点直接 CUDA 训练。

默认且严格要求完整 step9244 的文本初始化：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x3_7_mesh/formal_reinit1200_seed1_20261119_084545_3432609/checkpoint-009244
```

加载全部28物理层、四组原write/read、16个已训练router张量及tokenizer。
检查全部来源/schedule别名、完成标记、FP32/finite/完整safetensors库存和实际embedding/head绑定，
只允许省略一个tied别名。先完整严格加载，再注册音频read和LoRA，逐张量及Parameter对象复核继承不变。
不读文本 training_state.pt；原双循环音频checkpoint只作实现参考，不作初始化或full resume。
音频optimizer/scheduler/Scaler/GradNorm/RNG/数据游标从step0新建。

## 模型与精确初始化

28物理层、56次执行：0..6、(7..20)×3、21..27。suffix逻辑起点49。
每完整forward在prefix前建立 audio8 [B,8,126,576] 和 text6 [B,6,L-126,576]：

```text
audio: [E,0,0,0,0,0,A4,A8]
text:  [E,0,0,0,0,0]
```

| 边界（累计执行层数） | write | 音频read | SEP/text read | 随后执行 |
|---|---|---|---|---|
| 7 | 原write0，六槽 | 新read0，八槽 | 原read0，六槽 | loop0/LoRA0 |
| 21 | 原write1，六槽 | 新read1，八槽 | 原read1，六槽 | loop1/LoRA1 |
| 35 | 原write2，六槽 | 新read2，八槽 | 原read2，六槽 | loop2/LoRA2 |
| 49 | 原write3，六槽 | 原read3，六槽 | 原read3，六槽 | suffix |

四个write均用原参数处理完整序列，仅更新前六槽；A4/A8的槽6/7全程只读。
prefix query/value均取prefix输出；各循环query取本轮输入，write value取本轮输出。
无gate、listen write、额外残差注入或跨forward记忆。每次LM调用显式传入本次(A4,A8)，拒绝缺失、
错shape/dtype/device、KV cache与不完整音频prefix。

三个新read分别从已训练read0/1/2独立复制，CPU fork_rng保持调用者RNG：

```python
W_new[:6] = W_text
b_new[:6] = b_text - math.log(2)
W_new[6] = W_new[7] = W_text[0]
b_new[6] = b_new[7] = b_text[0]
```

同query下原六槽内部相对分布保持，新两槽初始等权；设S=sum(exp(z))，
alpha=S/(S+4*exp(z[0]))，新概率为[alpha*p0,...,alpha*p5,(1-alpha)/2,(1-alpha)/2]。
组概率随query变化，不固定为1/2、1/4、1/4。训练后不再约束行值相等。

CED-small来自原 models/CED_small，冻结且每个子模块eval。每批一次whole20编码，
hooks捕获完整block4/8输出（final norm前），成功或异常均移除hooks。
严格保留split/frequency/time到[126,4,384]的排列，原最终特征走freq_merge_126 mapper。
两个独立LN1536→Linear1536/576→GELU→Dropout0→Linear576/576→LN576生成A4/A8。
CED处于no_grad，主mapper和MLP在其外训练；MLP初始化也隔离CPU RNG。
A4/A8在三轮间复用，不新增第三个CED层或第三个MLP。

仅物理7..20的q/k/v/o、gate/up/down七投影挂三套独立LoRA：
r8/alpha16/scale2/dropout0，A Kaiming/B零，基础W共享且继续全量训练；前后各7层无LoRA。
逐调用选轮次，finally复位；禁止decoder gradient checkpointing和adapter merge。
初始零B等价性仅对同一audio8模型关闭LoRA delta比较，不能声称等价普通三循环六槽音频模型。

| 新增部分（不含原主mapper） | 参数 |
|---|---:|
| 两MLP | 2,443,776 |
| 三audio read | 13,848 |
| 三轮LoRA：588矩阵 | 3,419,136 |
| 总计 | 5,876,760 |

原8个router加3个音频read，共11模块/22张量/41,544参数。
每轮14层×7投影=98次，三轮共294次投影调用；三轮共有42个middle-layer调用，完整LM为56次。

## 数据、优化及staging

| 配置 | 合同 |
|---|---|
| 数据 | 原hybrid 1,477,492行，保留MusicCaps及全部SpeechCraft splits |
| 音频 | mono32k/fixed20中心裁剪或右补零，原真实双录音规则 |
| 序列 | audio126+SEP1+prompt256+answer300=683，prefix383 |
| 文本/loss | 原S1模板、分词前追加EOS、截断后不补回；PAD为已有!，EOS/SEP0 |
| 注意力 | attention_mask=None/labels=None/use_cache=False；外部shifted answer-only CE |
| 归约 | rank-local有效token mean→GA4平均→DDP平均 |
| Batch/时长 | 8GPU×micro4×GA4=GBS128；3epochs、11,542更新/epoch、共34,626 |
| warmup/保存 | round(5%)=1731；5771/11542/17313/23084/28855/34626及epoch3 |
| LR | 基础LM/原8router最大3e-4；主mapper/两MLP/新3read/三套LoRA最大6e-4 |
| 优化/数值 | 原AdamW(.9,.999)/eps1e-8、矩阵WD.01/低维及bias无WD、LambdaLR floor.1、GradNormTracker |
| 精度/随机 | FP32前向和参数、保留GradScaler、无autocast/TF32；seed1234及rank1234+rank |

原数据manifest为 /hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json。
原resolved_paths为 outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv。
CPU index后共享原始音频立即训练，默认4workers后台复制raw bytes；全部文件/字节核对通过，
optimizer边界取四个microbatch前八rank MIN共同切换。退出先停止/reap producer，再按owner/token清理。
tmpfs为 /dev/shm/mizar_s1_gbs256_text71437_ced48_audio8_read3_lora_r8_<token>；
gbs256前缀仅满足原路径守卫，实际GBS128。

## 输出与完整恢复

默认run-root：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced48_audio8_read3_lora_r8_gbs128_text71437_3090/formal_<timestamp>/
```

其下formal/包含runtime.json、config_audit.json、metrics、各rank前向/梯度审计、
staging_report、training_report及checkpoints/。最终为formal/checkpoints/model--step-34626.ckpt
和model--epo-3.ckpt。完整job名含后缀少于60字符。

可选指定run-root：

```bash
RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced48_audio8_read3_lora_r8_gbs128_text71437_3090/formal_$(date +%Y%m%d_%H%M%S%N)"
bash code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced48_audio8_read3_lora_r8_3090.sh \
  --output-dir "$RUN_ROOT" --stage-workers 4
```

本路线中断后，沿用原外层run-root及相同代码/资产/数据/文本来源/配方恢复：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN_ROOT=/absolute/original/formal_timestamp
bash code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced48_audio8_read3_lora_r8_3090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

完整保存全部模型/分支、具名optimizer库存和顺序、moments、scheduler、Scaler、GradNorm、
8rank Python/NumPy/Torch/CUDA RNG、数据和梯度审计游标。先构造分支，再strict覆盖保存值，
不重置read或LoRA；继续至本轮总计34626步。已完成checkpoint可重载审计但不能作为训练resume。
每次重建本节点staging，从shared开始；旧节点路径不作恢复身份。
默认文本与CED资产仍须保留，恢复会核对来源。其他路线checkpoint不能full resume。

## 验收与精确Git同步

```bash
python -B tests/validate_mizar_s1_text71437_ced48_audio8_read3.py
# 有真实ML依赖时可选；合成CPU模型，不访问生产资产，不作为formal前置。
python -B tests/validate_mizar_s1_text71437_ced48_audio8_read3.py --require-tensors
```

报告：audit_reports/mizar_s1_text71437_ced48_audio8_read3_validation.json。
本地最终联合96项：78项执行通过，18项真实张量因缺ML依赖明确跳过（新路线9项、原路线回归9项）。
真实Torch/autograd、生产checkpoint/CED资产加载、3090显存和八卡训练未执行；不登记GPU PASS。
首轮77项通过后补充第三轮恢复控制流测试，原结果保存在报告previous_runs中。
验证执行实际生产routing、LoRA、stack、optimizer、checkpoint writer与恢复守卫，
NumPy独立递推/有限差分，真实Bash+模拟vc/torchrun、CPU index/复制/切换/失败清理及路径守卫；
另含可选真实小模型autograd、strict loader、保存重载和恢复后下一更新对照。
NumPy层与模拟CUDA/collective/作业提交不算真实autograd或GPU验证；缺依赖必须明确SKIP。
3609份本任务开始时已有tracked Python/Shell/JSON受摘要保护，原始库存不按本轮结果重写。
15份数据/staging运行源码逐字节继承；整个epoch更新循环与read2母本AST保持，仅修正三轮错误提示。
本地验收、生产权重/3090显存/八卡训练分开记录。当前未访问远程资产、提交真实作业或commit/push。
既有评测器未注册新合同，后续评测须单独适配。

本地仓库根精确暂存（只包含本次新路线及交接记录）：

```powershell
$route = 'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71437_mesh_ced48_audio8_read3_lora_r8'
git add -- $route code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced48_audio8_read3_lora_r8_3090.sh tests/test_mizar_s1_text71437_ced48_audio8_read3.py tests/test_mizar_s1_text71437_ced48_audio8_read3_route.py tests/test_mizar_s1_text71437_ced48_audio8_read3_shared_start.py tests/test_mizar_s1_text71437_ced48_audio8_read3_staging_roots.py tests/test_mizar_s1_text71437_ced48_audio8_read3_audio.py tests/test_mizar_s1_text71437_ced48_audio8_read3_loader.py tests/test_mizar_s1_text71437_ced48_audio8_read3_behavior.py tests/validate_mizar_s1_text71437_ced48_audio8_read3.py audit_reports/mizar_s1_text71437_ced48_audio8_read3_validation.json RM.md
git diff --cached --stat
git commit -m "Add isolated three-loop CED audio8 read3 LoRA training"
git push
```
