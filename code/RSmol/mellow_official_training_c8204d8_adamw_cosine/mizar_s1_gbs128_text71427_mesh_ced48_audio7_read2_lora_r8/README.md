# 7–14×2–7 / CED4,8 / audio7 pre-loop read2 / independent loop LoRA r8

2026-11-20，独立正式音频路线。本地Windows只有源码；数据、权重、环境和输出位于远程Linux，GitHub用于同步代码。
母本为`mizar_s1_gbs128_text71427_mesh_ced_small`，CED/LoRA来自`mizar_s1_gbs128_text71427_mesh_ced48_loop_lora_r8`，
混合记忆参考audio7slot_split3，但本路线只扩展两个read，并采用下述新初始化。原文件normalized-LF摘要见`protected_sources.sha256`。

## 正式提交

代码同步到远程仓库后执行（不添加smoke前置）：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_lora_r8_3090.sh
```

登录节点只做CPU资产检查和索引准备；随后vc submit申请pdgpu-3090、1节点8GPU、64CPU、256G。
使用母本镜像和`mellow_c8204d8`环境，计算节点torchrun启动8rank。
入口支持`--output-dir`（run root）、`--resume-checkpoint`、`--text-model-dir`、`--ced-model-dir`、
`--resolved-paths`、`--index-cache-root`和`--stage-workers`，默认复制线程4。

唯一默认文本来源（注意job目录3963870）：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/formal_reinit1000_lr6e4_20261118_112042_3963870/checkpoint-009244
```

9244仅为文本权重步数；音频训练optimizer/scheduler/Scaler/GradNorm/RNG/数据游标从0开始。
先校验完成标记、配置、tokenizer及全部FP32 safetensors（只允许省略一个tied embedding/head别名），
严格加载本包模型，然后注册新read/LoRA，再逐张量验证继承不变。不读取文本training_state.pt。

## 模型和精确初始化

hidden576，28个物理层、42次执行，物理索引均为零基：

| 边界 | Transformer执行 | write | audio read | SEP/text read |
|---|---|---|---|---|
| 第一轮前 | prefix0..6 | 原write0五槽 | 新read0七槽 | 原read0五槽 |
| 第二轮前 | loop1:7..20，LoRA0 | 原write1五槽 | 新read1七槽 | 原read1五槽 |
| suffix前 | loop2:7..20，LoRA1 | 原write2五槽 | 原read2五槽 | 原read2五槽 |
| 输出 | suffix21..27 | — | norm/head | norm/head |

每次完整forward在prefix执行前重建audio memory `[B,7,126,576] = [E,0,0,0,0,A4,A8]`；
SEP起其余557位置为text memory `[B,5,557,576] = [E,0,0,0,0]`。
三个write共享原五槽参数，处理完整序列，仅更新前五槽；A4/A8槽5/6在整个forward内只读。
前两个audio read读取七槽，最后一个audio read仅读前五槽。所有文本read仍使用原五槽。
prefix交互query/value均为prefix输出；循环交互query为该轮输入，value为该轮输出。
无额外gate、listen write、残差注入、重复初始化或额外MeSH交互；记忆不跨forward保存。

两个新audio read分别复制原read0/read1，严格执行用户最终确认的公式：

```python
W_new[:5] = W_text
b_new[:5] = b_text - math.log(2)
W_new[5] = W_text[0]
W_new[6] = W_text[0]
b_new[5] = b_text[0]
b_new[6] = b_text[0]
```

同query下，设原logits为z、S=sum(exp(z))，则alpha=S/(S+4*exp(z[0]))，
新概率为`[alpha*p0,...,alpha*p4,(1-alpha)/2,(1-alpha)/2]`。
原五槽内部相对分布不变，新增两槽初始等权；组概率随query变化，不强制1/2、1/4、1/4。
全部新参数独立存储，CPU fork_rng保持调用者RNG，训练后无行间约束。

CED-small严格离线，默认`/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small`，冻结/eval。
每批只编码一次20s音频，hooks取得完整第4/8 block输出（最终norm前），finally移除hooks。
整段mel/BN后拆段；`[B,504,384] → [B,2,4,63,384] → [B,126,4,384]`保持时频顺序。
最终特征走母本官方freq_merge_126 mapper（dropout0.05、resample=False、外部Identity）。
两独立MLP为`LN1536 → Linear1536,576 → GELU → Dropout0 → Linear576,576 → LN576`；
CED在no_grad内，主mapper及两个MLP在其外训练。每次LM调用显式传(A4,A8)，不缓存跨样本特征。

仅物理层7..20的q/k/v/o、gate/up/down七投影加入两轮独立LoRA：r8、alpha16、scale2、dropout0。
A为Kaiming uniform，B零；主干共享并全量训练，原Parameter对象及名称不变。
前后各7层不挂LoRA。逐次调用选择loop并finally清除，禁止decoder gradient checkpointing、KV cache和merge。

| 新参数 | 数量 |
|---|---:|
| 两MLP | 2,443,776 |
| 两audio read | 8,078 |
| 两轮LoRA（392矩阵） | 2,279,424 |
| 合计（不含已有主mapper） | 4,731,278 |

原6router加新2router，共8模块/16张量/25,388参数。配置与真实参数清单在GPU preflight核验。

## 数据、优化与staging

hybrid共1,477,492行，原manifest/resolved_paths和单/双原录音语义不变；fixed20中心裁剪/右补零。
原manifest：`/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json`。
resolved：`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv`。
8GPU×micro4×GA4=GBS128，3epochs，每轮11,542次更新（丢116行），总34,626，warmup1,731。
基础LM/原六router最大LR3e-4；主mapper/两MLP/两新read/两轮LoRA最大LR6e-4。
原AdamW(.9,.999)/eps1e-8/WD.01，bias与ndim≤1无WD；LambdaLR warmup/cosine floor.1及GradNormTracker保持。
FP32参数/forward，原GradScaler，无autocast/TF32；seed1234、各rank1234+rank。
布局126audio+1SEP+256prompt+300answer=683；PAD为既有`!` id17，SEP/EOS id0，无词表扩展。
prompt/answer先追加EOS再截断，不补回截断EOS。LM attention_mask/labels为None，外部CE使用logits382..681预测300答案。
rank-local microbatch有效token mean→GA4平均→DDP平均，保留母本语义。
CPU index后共享音频立即开训，4线程后台复制，全部完成后optimizer边界8rank MIN集体切换。
tmpfs前缀保留安全守卫要求的`/dev/shm/mizar_s1_gbs256_`，后缀独立；实际GBS128。
退出先停止/reap复制producer，再按owner/token清理。

## 保存和完整恢复

每5771步及最终保存：5771/11542/17313/23084/28855/34626，另存epoch3最终文件。
默认run root为`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced48_audio7_read2_lora_r8_gbs128_text71427_3090/formal_<timestamp>`，
实际训练在其`formal/`下，checkpoint位于`formal/checkpoints/`。
恢复示例（替换两个实际路径）：

```bash
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_lora_r8_3090.sh \
  --output-dir /absolute/path/to/existing_run_root \
  --resume-checkpoint /absolute/path/to/existing_run_root/formal/checkpoints/model--step-5771.ckpt
```

完整保存模型、新分支、AdamW、scheduler、Scaler、GradNorm、8rank RNG、epoch/数据游标和梯度审计游标。
校验本路线/代码/资产/数据/配方、参数名字及顺序、LR/WD/Adam步数和数据游标一致。
先构造分支再strict加载保存值；恢复后不重置read初始化或LoRA。拒绝其他路线、仅权重、缺分支及已完成run。
前两次更新及每25次更新审计missing/finite及正确LoRA轨迹，零梯度作诊断；B零时A任务梯度应为零。
没有gate unlock期限或概率占比硬门槛。rank报告记录参数变化及CED冻结状态。

## 本地验证

```bash
python tests/validate_mizar_s1_text71427_ced48_audio7_read2.py
```

执行源码保护/继承检查、实际源码NumPy行为、shell与Python3.10语法、CPU index/staging/清理回归及可选真实PyTorch检查。
NumPy检查不等于autograd；缺torch/transformers/safetensors时明确SKIP真实张量检查，不能登记GPU PASS。
验证报告在仓库`audit_reports/mizar_s1_text71427_ced48_audio7_read2_validation.json`。
