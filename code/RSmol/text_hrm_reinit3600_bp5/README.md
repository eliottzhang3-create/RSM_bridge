# HRM：从 raw step3600 重新训练9244步，seed1、固定BP5

本路线为隔离实现，复用原 `text_hrm_h2l3` 模型、数据流、真实DDP token平均累积和官方Adam-atan2，不修改原路线。Windows本地维护代码，通过GitHub同步；生产模型、85片Parquet、初始化快照与训练输出均在远程Linux。当前只实现文本训练。

默认初始化来源固定为用户指定：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_h2l3/formal_20261122_000344_4005340/checkpoint-step-003600
```

使用根目录raw权重，继承已训练gate、全部20层、Norm、embedding/head及固定zL_init。加载后严格核对tensor库存/形状/FP32/finite/绑定，不运行转换、不重新折叠gamma、不清零gate、不重复补偿o_proj。新优化器动量/每参数计数从零开始，EMA从加载的raw权重开始累计，RNG及数据游标重新建立。来源step3600只记为血缘，本轮计数1..9244。

## 固定配方

| 项目 | 配置 |
|---|---|
| 模型 | P5/L5/H5/C5、20物理层、50次执行、H2L3；MagicNorm和gate保持 |
| Attention / loss | causal＋真实padding mask；全部有效next-token；全8rank/64micro统一token平均 |
| 更新 / batch | 9244更新；8GPU×micro2×GA64=GBS1024 |
| 数据 | 原85片Parquet，data seed=1，buffer4096，1024截断、动态右padding、不packing |
| BP | 全程5；H1、L4/L5/L6、H2求导，Prelude/Coda正常求导 |
| 优化器 | 原官方Adam-atan2，β=(.9,.95)，全部唯一可训练参数WD=.1 |
| EMA | .9999，重新从step3600 raw权重开始累计 |
| 精度 | FP32主参数/状态/残差，BF16前向、FP32 Norm/CE、TF32关闭 |
| 梯度处理 | 不裁剪、无GradScaler/GradNormTracker、不跳过异常更新 |
| 保存 | 每400及最终9244，最近3份；正常最终8800/9200/9244 |
| 资源 | pdgpu-3090，64CPU、256G、8GPU、1node；原镜像/rsmol/Transformers4.54.1 |

学习率在每次optimizer.step之前按本轮一基更新号s设置。`round(9244*.05)=462`，前462步含首尾两个端点：

```text
s=1..462:    3e-4 + (6e-4 - 3e-4) * (s-1)/461
s=463..9244: 6e-5 + (6e-4 - 6e-5)/2 * (1 + cos(pi*(s-462)/8782))
```

第1/462/4853/9244步分别为3e-4/6e-4/3.3e-4/6e-5。不存在零LR预热、旧3000步预热或旧BP warmup。

全局shard打乱和rank分配使用seed1；rank内shuffle使用`1+1009*rank`。训练Python/NumPy/Torch的rank seed沿用原设置，但重新建立，不恢复来源RNG。固定zL_init从权重文件继承，data seed不影响它。本轮名义9465856样本额度，不承诺每篇有效文档恰好一次。数据身份仍为名称/大小/mtime_ns，无全24GB内容扫描。

## Prelude实际任务梯度

Prelude五层全部trainable并进入优化器。E没有detach，H1在有梯度上下文中使用E，保留loss经过H2/H1回到E/Prelude的路径；第一轮L1..L3的无梯度调用不会移除这条路径。

首次microbatch实际反传后，检查Prelude输出、入口E、H1/H2、L4/L5/L6及各模块任务梯度，并逐一记录前五个物理层梯度。首个完整64micro累积结束、optimizer更新与weight decay之前，再检查每层全部参数的grad均存在/有限、每层聚合L2>0。恢复后首个更新也重新审计。任一rank失败会集体报错，不静默冻结或继续训练。

文件为`gradient_bp5_rank<N>.json`、`prelude_full_update_rank<N>.json`；每更新继续检查所有可训练参数缺失/非有限梯度。参数更新可能包含weight decay，因此不会单靠参数变化宣称任务梯度有效。

## 初始化快照与正式提交

原run只保留最近3份checkpoint。正式wrapper在CPU上先校验并复制raw模型/tokenizer到独立快照，成功后才执行vc submit；不存在额外GPU smoke或CPU数据内容扫描前置。准备过程不反序列化旧training_state.pt，也不复制旧优化器/EMA。它会核对原完整checkpoint的文件SHA，因此会顺序读取其文件做校验。快照采用无覆盖原子发布。

默认快照：`/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-HRM-H2L3-reinit3600-raw`。同来源且SHA完整的既有快照可复用，原checkpoint随后被保留策略移除也不影响排队/训练。快照存在但身份不同或损坏时直接拒绝；源已丢失且没有快照时拒绝，不选择其他步数。

本地PowerShell在仓库根同步本次文件：

```powershell
git add -- code/RSmol/text_hrm_reinit3600_bp5 code/RSmol/run_text_hrm_reinit3600_bp5_3090.sh code/RSmol/scripts/prepare_text_hrm_reinit3600_bp5.sh code/RSmol/scripts/train_text_hrm_reinit3600_bp5_ddp.sh tests/test_text_hrm_reinit3600_bp5.py tests/validate_text_hrm_reinit3600_bp5.py
git add -f -- audit_reports/text_hrm_reinit3600_bp5_validation.json
git add -p -- RM.md
git diff --cached --stat
git diff --cached --check
git commit -m "Add HRM raw3600 reinitialization with fixed BP5 and cosine LR"
git push
```

远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_text_hrm_reinit3600_bp5_3090.sh --dry-run
bash code/RSmol/run_text_hrm_reinit3600_bp5_3090.sh
```

dry-run不读资产、不准备快照、不初始化CUDA、不提交。正式命令自动准备快照再提交一个新formal作业。无需运行旧30层转换脚本，也无需先把原训练跑到9244。

如需先单独保留step3600，可立即在远程CPU执行（不提交GPU）：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/scripts/prepare_text_hrm_reinit3600_bp5.sh
```

固定输出示例（目录必须不存在）：

```bash
bash code/RSmol/run_text_hrm_reinit3600_bp5_3090.sh \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_reinit3600_bp5/formal_first_v1
```

默认输出`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_reinit3600_bp5/formal_<timestamp>_<pid>/`，无额外内层formal。
日志在远程仓库`code/RSmol/log/text_hrm_reinit3600_bp5/`。支持`--source-checkpoint`（仍必须原HRM step3600）、`--init-snapshot`、`--data-dir`、`--output-dir`、`--resume-from`；仅使用`RSMOL_HRM3600_*`变量，不继承旧路线变量。源码包依赖原`text_hrm_h2l3`，二者均保留在Git仓库中。

## 本轮断点恢复与输出

初始化快照与本轮完整resume是两种明确区分的入口。原step3600不得传给本轮`--resume-from`。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/run_text_hrm_reinit3600_bp5_3090.sh \
  --resume-from /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_reinit3600_bp5/formal_first_v1/checkpoint-step-000400 \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_reinit3600_bp5/formal_resume400_v1
```

完整恢复要求本新路线、相同源码（包括复用的原HRM文件）、数据/seed、依赖、配方和初始化身份，输出到新的目录。恢复raw、优化器全部动量/计数、EMA、8rank RNG和数据游标，继续到总计9244；此时不会重置动量/EMA或重走升温。末步9244拒绝再次完整resume。

`runtime.json`记录本轮合同；`startup_rank<N>.json`记录新开始或完整恢复模式/起始步/LR；`metrics.jsonl`记录全局token loss、实际LR/BP与显存。最终`checkpoint-step-009244/`为raw，`checkpoint-step-009244/ema/`为独立EMA；二者不预设孰优。完整训练报告才表示训练完成，vc入队不表示PASS。

模型config继续标识原HRM架构，新的marker/binding单独标识本轮训练配方。旧训练恢复入口不能代替本轮入口；音频训练与旧benchmark适配本次未加入。

## 本地验证范围

```powershell
.\.hrm_test_runtime\Scripts\python.exe -B tests/validate_text_hrm_reinit3600_bp5.py --require-tensors
```

任意具备依赖的环境也可运行`python -B tests/validate_text_hrm_reinit3600_bp5.py --require-tensors`。报告`audit_reports/text_hrm_reinit3600_bp5_validation.json`记录实际通过/失败/跳过和依赖版本。

2026-11-22联合回归42/42通过、无跳过；其中新路线13项及原HRM29项。Python3.10语法、三份Bash入口/README命令语法和新增文件whitespace检查另行核对。

测试覆盖真实CPU微型模型raw3600快照、已训练gate逐值继承、旧training_state不反序列化、新optimizer/EMA清零与初始化、源被移除后快照可复用；9244位置LR/BP、seed1真实Parquet顺序与游标重放、学习率462/463边界完整恢复后下一更新一致、checkpoint保留/损坏/错路线拒绝。

真实两进程CPU/Gloo/DDP测试比较新路线累积与独立全局token平均梯度，同时审计每层Prelude。另有实际正式启动及首个64micro更新的CPU控制流测试：模型/数据/autograd/optimizer真实，CUDA和八rank通信明确模拟，完成首更新后主动中断，不冒充GPU通过。真实Bash测试模拟prepare/vc/torchrun，检查含空格参数、3090资源、准备失败不提交及resume不重读原来源。

联合运行原HRM的数值/运行测试；其历史全仓源码摘要早于后续已授权MMSU修改，本次使用任务开始时完整3280份tracked Python/Shell摘要替代这一历史库存断言，报告明确记录。其他原测试保持。首轮两个异常类型断言不符已修正，历史库存差异明确单列，不修改原测试或覆盖其历史快照。

测试环境Python3.13.5、Torch2.15.0+cpu、Transformers4.54.1，与远程Torch2.11/CUDA不同。CPU合成通过不等于3090显存/真实八卡NCCL训练或生产来源checkpoint已通过；本地不读取远程权重/数据、不提交作业。缺依赖跳过会明确记录，不能当作真实tensor验证通过。
