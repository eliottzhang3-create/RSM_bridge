# SmolLM2：20 物理层、50 次执行的 HRM H2L3 文本训练

本路线只实现文本转换和文本训练。Windows 本地维护代码，GitHub 中转；真实 SmolLM2 权重、85 片 Parquet、转换产物与训练输出均在远程 Linux。GPU 作业只通过本路线的 `vc submit` wrapper 提交。未修改旧 MeSH、Huginn、普通递归、音频模型或其训练实现。

## 模型与初始化

默认来源：`/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2` 的原始 30 层模型。

| 目标模块 | 目标零基层号 | 原始模型一基层号 | 完整 forward 调用次数 |
|---|---|---|---:|
| Prelude | 0..4 | 1,2,3,4,5 | 1 |
| L | 5..9 | 6,8,10,12,14 | 6 |
| H | 10..14 | 16,18,20,22,24 | 2 |
| Coda | 15..19 | 26,27,28,29,30 | 1 |

20 个物理 decoder 模块，50 次层执行；两套 L/H 参数独立、各自跨调用共享。hidden576、MLP1536、GQA 9 query/3 KV heads、head64、RoPE theta100000、词表49152及 embedding/head 绑定继承。生产 LM 独立参数应为 102,420,288（约102.4M），不是此前讨论的30物理层版本。

`E = RMS_entry(Prelude(Embedding(x)))`；`H0=E`，`L0=c`。每个外循环依次执行三次 `L=RMS_exit_L(Lstack(L+H))`，再执行一次 `H=RMS_exit_H(Hstack(H+L))`，共两轮；最终 H2 进入 Coda、原 final norm、原 LM head。第二轮不重置状态，不重复加入原 E。每次完整 forward 重建状态，无跨 batch carry，无 KV cache。所有调用使用相同 token position IDs。

MagicNorm：中段每层采用无参数 PreNorm，层内 eps 保留源值1e-5；每次 L/H 调用后执行无参数出口 RMSNorm，eps1e-6。另有入口无参数 RMSNorm，eps1e-6，属于本项目的 Prelude 适配。初始 L 向量按照官方 `normal_().fmod_(3).mul_(1.014762601732121)`，隔离 CPU seed0 生成、FP32 buffer 保存，各 rank 相同，不训练、不每次采样。

中段 attention 输出在 o_proj 之前乘 sigmoid 逐通道576维 gate。转换时将源 attention Norm 的 gamma 折叠到Q/K/V投影列，将 MLP Norm gamma 折叠到gate/up投影列；新 gate weight=0、o_proj weight=源值×2。保留源层子层的实数算术初值，浮点误差另验。整个 HRM 模型与原30层模型不具有全模型 logits 等价性。Prelude/Coda和原最终Norm原值保留。

训练后重载只严格读取已保存参数，不重新折叠Norm、清零gate或再次乘2。显式 `SmolHRMForCausalLM` 类加载，不注册全局 AutoModel；不能用普通20层Llama替代本模型。

## 固定训练合同

| 项目 | 配置 |
|---|---|
| Attention / 目标 | 全程 causal＋真实padding mask；全部有效 next-token |
| Batch | 8 GPU × micro2 × GA64 = GBS1024 |
| 更新数 / 名义样本额度 | 9244 / 9,465,856 |
| 数据 | 原85片Parquet；seed0分片打乱后连续分rank；片内buffer4096 |
| 文本处理 | 原tokenizer add_special_tokens=False；截断1024；动态右padding；不packing |
| Loss | shifted CE sum / 全8rank和64次累积的有效目标token总数；无额外GA除法 |
| 优化器 | 官方 AdamATan2，beta=(.9,.95)，所有唯一可训练参数WD.1，无epsilon配置 |
| LR | 更新s使用 `3e-4*min(s/3000,1)`；s=1..9244 |
| EMA | .9999；训练raw；raw和EMA单独保存 |
| 精度 | FP32主参数/状态；BF16前向autocast；FP32 Norm/CE；TF32关闭 |
| 梯度处理 | 不裁剪，不用GradNormTracker/GradScaler；非有限值报错，不静默跳过 |
| 保存 | 每400次及9244；最近3份，正常最终8800/9200/9244 |
| 资源 | pdgpu-3090 / 64CPU / 256G / 8GPU / 1node |
| 环境 | 原容器；rsmol，Transformers4.54.1 |

第1步LR=1e-7；第3000步达到3e-4；3001..9244恒定，无cosine或末尾衰减。LR在对应optimizer.step之前设置，完整resume使用下一更新号继续。

反传独立调度：`K(s)=3+min(2,(2*s)//1849)`。更新1..924：H1/L6/H2；925..1848：H1/L5/L6/H2；1849..9244：H1/L4/L5/L6/H2。Prelude/Coda正常求导，E不全局detach；首步已有经H1的Prelude任务梯度。第一轮L的调用不求导，但其共享参数从后续L调用获得梯度。K5仍不是全部8次中段调用反传。

使用PyTorch DDP和现有Llama/SDPA路径，不依赖官方FA3、FSDP2或torch.compile；禁止decoder gradient checkpointing。micro2是固定计划值，实际3090峰值须由远程K5/完整context/已建立optimizer和EMA状态验证。本地CPU通过不代表3090显存或8GPU训练通过。

原数据读取/分词/padding函数逐函数复制并进行AST保护。当前名义训练额度不承诺所有有效文档恰好遍历一次。数据身份为85片的路径/名称/大小/mtime_ns，不读取全24GB内容做哈希，不冒充完整数据内容审计。

## GitHub 同步

本地PowerShell，在仓库根执行。只提交本路线；`RM.md`有用户先前修改，请按块选择本次记录。

```powershell
git add -- code/RSmol/text_hrm_h2l3 code/RSmol/scripts/convert_text_hrm_h2l3.sh code/RSmol/scripts/train_text_hrm_h2l3_ddp.sh code/RSmol/run_text_hrm_h2l3_3090.sh tests/test_text_hrm_h2l3.py tests/validate_text_hrm_h2l3.py
git add -f -- audit_reports/text_hrm_h2l3_validation.json
git add -p -- RM.md
git diff --cached --stat
git diff --cached --check
git commit -m "Add isolated 20-layer HRM H2L3 text conversion and training"
git push
```

隔离本地测试环境 `.hrm_test_runtime/` 被其目录内 `.gitignore` 全部忽略，不提交依赖、权重或测试数据。`protected_sources.json` 是代码摘要清单，可正常Git同步，不是生产权重。

## 远程CPU转换

在Linux登录节点执行，不申请GPU，不读训练数据；默认CPU8线程、严格离线。已有目标拒绝覆盖。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/scripts/convert_text_hrm_h2l3.sh
```

默认输出：`/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-HRM-H2L3-P5-L5-H5-C5`。

可显式指定独立路径：

```bash
bash code/RSmol/scripts/convert_text_hrm_h2l3.sh \
  --source-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2 \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-HRM-H2L3-P5-L5-H5-C5 \
  --threads 8
```

转换验证完整源tensor库存/finite/绑定、20层映射、各层非平凡Norm折叠和gate补偿、50次轨迹、入口1/L出口6/H出口2、因果后缀扰动、保存重载全部tensor/logits/tokenizer。支持单文件/分片safetensors及bin；拒绝混合入口、坏分片、矛盾tied值、训练态来源和已有输出。原始source文件SHA在发布前再次验证；失败保留私有staging/failure报告。沿用CloudStor兼容的无覆盖原子发布实现。

## 正式训练提交

转换成功后，在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/run_text_hrm_h2l3_3090.sh --dry-run
bash code/RSmol/run_text_hrm_h2l3_3090.sh
```

`--dry-run`只打印参数，不读模型/数据、不初始化CUDA、不提交。正式入口直接提交一个formal，无额外CPU数据审计、GPU smoke或历史PASS report前置。提交成功只表示入队，不代表训练PASS。

默认输出：`outputs/RSmol/text_hrm_h2l3/formal_<timestamp>_<pid>/`，不再追加内层formal。上述相对路径以 `/hpc_stor03/sjtu_home/jinwei.zhang/` 为根。

固定输出示例（必须是新的不存在目录）：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/run_text_hrm_h2l3_3090.sh \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_h2l3/formal_first_v1
```

支持 `--model-dir`、`--data-dir`、`--output-dir`、`--resume-from`。环境前缀仅 `RSMOL_HRM_*`，不继承其他路线变量。完整job名包含时间/PID后缀小于60字符。日志：远程仓库 `code/RSmol/log/text_hrm_h2l3/formal_*.JOB.log`。

## 输出、恢复与EMA

`runtime.json`记录完整合同、代码、初始化、数据和依赖身份；`metrics.jsonl`每更新记录8rank全局token平均loss、实际LR/BP和显存/耗时；BP3/4/5首micro分别保存各rank的实际边界及物理模块任务梯度审计，并立即释放审计tensor引用。失败写`failure_rank<N>.json`。

最终：`checkpoint-step-009244/`下标准HF模型是raw；`ema/`是独立EMA HF权重和tokenizer，不是完整resume入口。EMA保留原始初值贡献：.9999^9244约39.7%，不预判其一定优于raw。音频阶段以后再显式选择，当前不实现音频训练/旧评测适配。

每个完整checkpoint含raw模型、EMA、官方optimizer动量和每参数计数、LR/BP游标、8rank Python/NumPy/CPU/当前CUDA RNG、8rank数据游标和合同。完成标记发布前计算全部模型/状态文件SHA，完整恢复先校验再反序列化。严格拒绝其他路线、改源码/依赖/数据/超参、缺字段、错tensor形状/dtype/非有限、错rank/步数的恢复。

中断后用本路线原代码/数据/依赖，输出到一个新的独立run目录，继续至总计9244步：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/run_text_hrm_h2l3_3090.sh \
  --resume-from /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_h2l3/formal_first_v1/checkpoint-step-000400 \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_h2l3/formal_resume400_v1
```

不会重新预热/折叠Norm/初始化gate或EMA。最后9244步可以模型加载/评测/权重初始化，但拒绝再次完整resume。

数据恢复从保存的当前shard起始RNG重读该shard并跳过已完成microbatch，从而重建bounded shuffle buffer；实际CPU Parquet回归检查跨buffer、shard边界和epoch后的下一批一致。该保证依赖数据/tokenizer未变，不宣称CUDA/DDP训练bitwise exact。恢复可因当前shard重放而有CPU/I/O启动开销，不会把重放计入新的训练样本数。

## 本地验证

```powershell
.\.hrm_test_runtime\Scripts\python.exe -B tests/validate_text_hrm_h2l3.py --require-tensors
```

也可在任意已具备依赖的环境运行：

```bash
python -B tests/validate_text_hrm_h2l3.py --require-tensors
```

CPU合成测试不读取生产权重/数据，不运行CUDA，不提交作业，不作为formal额外前置。缺ML依赖时不加`--require-tensors`会明确skip，不能当作tensor通过。报告：`audit_reports/text_hrm_h2l3_validation.json`，以实际版本/通过项/跳过项为准。Windows如沙箱阻止临时目录或Git Bash信号管道，需要获准在沙箱外执行同一离线测试。

2026-11-21本地结果：27/27通过，无跳过。包含真实两进程CPU/Gloo/DDP的生产累积函数，BP3/4/5梯度与独立全局token平均对照；真实微型Llama转换/重载/因果性、非平凡gamma折叠、Prelude梯度、Adam-atan2独立NumPy对照、完整optimizer/EMA/RNG恢复后下一更新一致、Parquet重放、85片manifest启动路径、checkpoint最近3份保留及损坏拒绝，以及真实Bash到模拟vc/torchrun的参数链。生产配置在meta device核对102,420,288参数，未分配生产权重。

验证环境为Python3.13.5、Torch2.15.0+cpu、Transformers4.54.1；Torch/Python与远程rsmol环境不同。CPU验证不能代替远程Torch2.11/CUDA、3090显存和八卡NCCL训练/恢复实测。3237份既有tracked Python/Shell normalized-LF摘要保持，原数据与Llama辅助函数AST一致；官方optimizer仅清除两处空行尾空格，算法AST保持一致。
