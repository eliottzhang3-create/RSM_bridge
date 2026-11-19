# 7–14×2–7 MeSH + CED4/8 listen + 两轮独立 LoRA r8

2026-11-19新增的隔离正式音频路线。母本为`mizar_s1_gbs128_text71427_mesh_ced_small`；
CED/listen参考`mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_listen`；
LoRA参考`mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8`。
103份母本、参考、入口及共享模型文件的normalized-LF SHA保存在`parent_snapshot.json`。
本地Windows只有代码，权重/数据/环境/正式输出在远程Linux，通过GitHub同步代码。

## 正式提交

代码commit/push后，在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_listen_lora_r8_3090.sh
```

默认文本来源：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/formal_reinit1000_lr6e4_20261118_112042_3963870/checkpoint-009244
```

登录节点仅做CPU资产/完成标记检查和index准备，随后`vc submit`申请
`pdgpu-3090 / 单节点8GPU / CPU64 / 256G`，使用原镜像及`mellow_c8204d8`，计算节点torchrun启动8rank。
正式入口直接formal，无独立smoke/审查报告/人工放行条件。完整job名含时间戳小于60字符。
支持`--output-dir`、`--text-model-dir`、`--ced-model-dir`、`--resolved-paths`、`--index-cache-root`、
`--stage-workers`、`--resume-checkpoint`。新训练output-dir须为空，脚本在run root下建立`formal/`。
默认run root：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced48_listen_lora_r8_gbs128_text71427_mesh_3090/formal_<timestamp>
```

## 严格继承与前向结构

9244是文本来源步数，音频optimizer从step0训练。检查完整标记/config/来源层映射/schedule，
显式加载本包`models/recursive_mesh_listen_lora.py`，不注册全局AutoModel替换。
HF missing/unexpected/mismatched/errors必须为空，全部safetensors逐张量FP32精确一致；
六个原write/read router的12张量（17,310参数）继承并记录SHA，embedding/head绑定保持。
只允许省略一个tied别名。严格检查后才注册listen/LoRA，再验证继承值及Parameter对象保持一致。
不读取文本`training_state.pt`，不恢复其optimizer/scheduler/Scaler/RNG/数据游标。

28物理层、42次执行，hidden576、vocab49152；下表物理层及逻辑索引为零基：

| 阶段 | 物理层 | 逻辑执行 | 音频与LoRA |
|---|---|---|---|
| 前置7层 | 0..6 | 0..6 | 原write0 → listen CED4 → 原read0 |
| 第一轮14层 | 7..20 | 7..20 | LoRA0；原write1 → listen CED8 → 原read1 |
| 第二轮14层 | 7..20 | 21..34 | LoRA1；原write2 → 原read2 |
| 后置7层 | 21..27 | 35..41 | 原后段与norm/head |

每forward重建五槽`[B,5,683,576]` memory，slot0为原始embedding，其余零。
原前段router查询前段输出，原循环router查询该轮输入、write value为该轮输出。
listen在边界7/21，原write后、原read前，gate/router查询分别为前段输出/第一轮输出的前126音频位置：

```text
g_t = tanh(Linear_576_1(h_t))
p_t = softmax(Linear_576_5(h_t), slots)
memory[:, slot, :126] += p_t[slot] * g_t * A_t
```

实际为非原地更新，SEP/文本memory保持原write后的值。原六个router加两个独立listen router，仍五槽。
gate weight/bias全零；listen weight为正态`std=sqrt(2/(5*576))`采样后clamp到±3std、bias零，
即`normal_`后`clamp_`，不是`trunc_normal_`重采样。CPU fork_rng保持外部RNG。无第三次listen、额外gamma或除以4。

## CED与两轮LoRA

CED-small路径`/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small`，离线/冻结/eval。
一次encoder forward通过hooks取得第4/8完整block输出（1-based，final norm前），finally移除hooks。
整段20秒mel/BN后拆两段；`[B,504,384] → [B,2,4,63,384] → [B,126,4,384]`保持时间/频率对应。
最终特征用原官方freq_merge_126主mapper（dropout0.05、resample=False、外置Identity）。
CED4/8各自独立projector：`LN1536 → Linear1536,576 → GELU → Dropout0 → Linear576,576 → LN576`。
CED/特征提取在no_grad内，主mapper与两个projector在其外训练。

仅物理层7..20的q/k/v/o及gate/up/down七投影加入LoRA，作用于完整序列。
基础W两轮共享并训练，每层/投影/轮次独立A/B，保留原weight/bias对象及state_dict名字：

```text
y_loop = linear(x, W, bias) + 2 * linear(linear(x, A_loop), B_loop)
```

rank8/alpha16/scale2/dropout0，A为Kaiming uniform、B零，CPU fork_rng初始化；前后各7层保持普通投影。
LoRA共392矩阵、2,279,424参数；两个projector共2,443,776，gate共1,154，listen router共5,770，
相对普通71427 CED路线新增总计4,730,124参数（不含已有主mapper），加载后核验实际尺寸。
每次middle调用显式选择loop并在finally清除；拒绝decoder gradient checkpointing，避免重放丢失loop作用域。
每次LM完整prefix调用显式传(A4,A8)、use_cache=False；后续评测也需加载本路线模型并传递该特征对。

## 数据与训练合同

原hybrid 1,477,492行及source counts保持：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
```

S1 fixed20中心裁剪、prompt/answer先追加EOS再截断、截断后不补EOS、已有PAD `!` id17、SEP/EOS id0保持。
历史filepath2不视为第二条原录音，保留原raw resolution、MusicCaps和全部SpeechCraft splits。
布局`126audio+1SEP+256prompt+300answer=683`，attention_mask/labels为None，原prompt/answer PAD可见。
外部answer-only CE：logits382..681对应300目标，排除PAD；rank/microbatch有效token mean，再GA4及DDP平均。

| 项目 | 值 |
|---|---|
| Batch/epochs | 8GPU × micro4 × GA4 = GBS128；3epochs |
| 更新数 | 每epoch11,542，总34,626；每epoch丢弃116行 |
| 最大LR | 基础LM/原router3e-4；主mapper/projector/gate/listen/LoRA6e-4 |
| Scheduler | 原LambdaLR，warmup1,731、cosine floor0.1，step0为0.1×max LR |
| AdamW | betas(0.9,0.999)，eps1e-8；矩阵WD0.01，bias/ndim≤1无WD |
| 精度 | FP32参数/forward，原GradScaler，无autocast/TF32 |
| 梯度/RNG | 原GradNormTracker(1,10,2.5,0.995)及异常终止；seed1234，rank1234+rank |

CPU index后共享音频立即训练，默认4线程后台复制原始字节，全部完成后optimizer边界8rank MIN集体切换。
tmpfs前缀`/dev/shm/mizar_s1_gbs256_text71427_ced48_listen_lora_r8_`中的gbs256为母本安全守卫前缀，实际GBS128。
先停止/reap producer再按owner/token清理；暂存路径不进入恢复身份。

## 保存与完整恢复

保存5771/11542/17313/23084/28855/34626步，位置`<run-root>/formal/checkpoints/model--step-N.ckpt`，
最后另存`model--epo-3.ckpt`。保留全部模型/新增分支、optimizer/scheduler/Scaler/GradNorm、8rank RNG、
下一个数据游标及listen/LoRA累计梯度解锁状态。使用同一run root恢复：

```bash
RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced48_listen_lora_r8_gbs128_text71427_mesh_3090/formal_REPLACE_WITH_ACTUAL_RUN
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_listen_lora_r8_3090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt"
```

恢复匹配route/架构/代码/资产/数据/配方，检查优化器完整库存、LR、数据/scheduler游标和梯度解锁状态。
首次microbatch检查两轮边界梯度后立即释放保留张量；首步允许零gate造成projector/listen任务梯度为零、
零B造成LoRA A任务梯度为零。全部新角色须在32次真实更新内在所有rank累计观察到非零梯度，状态随checkpoint保存。
报告含config_audit、first_recursive_backward/first_gradient_rank<N>、listen_gradient_open_rank<N>、metrics及training_report。
metrics记录gate幅度/饱和度、槽概率、memory增量及两轮LoRA trace。

## 本地验证

```bash
python -B tests/validate_mizar_s1_text71427_ced48_listen_lora_r8.py
```

结果`audit_reports/mizar_s1_text71427_ced48_listen_lora_r8_validation.json`。
本次本地执行87项通过，19项缺真实ML依赖明确跳过，未执行远程GPU训练。
含Python3.10/Bash、真实Bash+模拟vc、index/后台复制/失败清理、源码隔离、生产listen/LoRA函数的NumPy数值/有限差分、
12角色梯度审计、完整恢复拒绝条件及原音频/文本回归。官方`reference_audit.json`记录1,344次文本/RNG、8组tokenize、
34,627个LR位置误差0、10,000次GradNorm标量和mapper AST一致。
本机缺Torch等依赖，真实张量测试明确跳过；完整ML环境可加`--require-tensors`，缺依赖/跳过即失败，此选项不接入formal。
真实测试覆盖CED hooks/梯度、严格继承、零分支等价、392独立矩阵、两轮反传、683/answer shift及完整恢复后下一步一致。
本地检查不代表远程权重加载、3090显存或正式训练完成，实际结果以远程formal报告为准。
