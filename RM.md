## 2026-11-27：原入口改为完整AudioMCQ / L9–12纯循环四次 / 3epochs / LR2e-5

用户确认计划后在原 run_mizar_l9_12x3_strong_reason_5090.sh 和其既有包内原地修改；文件名保留，当前配方已为x4/AudioMCQ-only。
本地代码经GitHub中转，正式音频/旧hybrid/权重/输出在远程Linux；未自动commit/push或提交GPU作业。

新CPU manifest从原完整训练 s1_stage1_hybrid_filtered.json 的1477492原行中保留全部720524 AudioMCQ，删除756968 ReasonAQA。
AudioCaps113844、Clotho4798、MusicCaps3221、CompA-R249161、LP-MusicCaps-MTT19632、SpeechCraft287727、TACOS42141。
strong/weak、直接答案/CoT等全部历史视图/原字段/答案/顺序保持，不抽样、不只限strong，不从未过滤S1引回历史缺失引用。
复用原hybrid audit TSV exists成功路径，旧已用文件现缺失/空或映射冲突即失败，Reason音频不stat/staging；不改共享映射脚本或规则。
默认新数据 data/Mizar/l9_12x4_audiomcq_only_v1，train/selection/resolved TSV/dataset_report/manifest seal；绑定旧hybrid/TSV大小内容身份与preparer。
CPU不解码/复制/前向模型，每10万行打印扫描/保留/排除数，末尾映射每2万引用打印，失败保留building诊断，成功原子no-replace发布、同身份幂等复用。
旧strong+Reason数据与训练输出不覆盖。生产720524行manifest因远程原数据仅由用户远程运行CPU命令生成，本地只实际生成/验证明确小fixture。
入口新增--prepare-only先生成/核验新manifest不申请GPU；普通入口自动ensure manifest→CPU index→release/admission→单次vc。--check-args仍纯打印。

完整官方Mizar final原454状态继承CED/已训练projector/30层文本基座：1..8→(9..12)x4→13..30，42次block，四轮总共四次。
每轮完整输出直接为下一轮输入，删除A8捕获/残差注入/额外Linear/tanh/两条gate状态；原Llama MLP gate_proj保留。
仅共享9..12四层全部含norm参数训练，14160384参数；原冻结145139456参数及buffer/eval保持，冻结后缀保留输入梯度，完整四轮BPTT。
无KV/decoder checkpointing/跨forward隐状态，final norm仅最后一次。完整454状态无新增key，strict官方loader继续验证finite/FP32/tied。

8GPU×micro4×GA4=GBS128，3epochs，每轮5630步=5629完整128更新+12条尾批，尾批前4rank各2条/后4rank各1条GA1，无drop/padding重复。
总16890步，全局一条scheduler，maxLR2e-5/min2e-6、round总步数3%warmup507，epoch不重启，最后真实update到min。
保留原AdamW betas/eps/WD与一维bias免decay、FP32/无autocast/TF32off、GradScaler/GradNormTracker、fixed20/在线CED/projector/683位置/原文本与answer-only loss口径。
epoch每轮seed1234+epoch随机分片，按真实epoch内游标恢复；每轮边界先检查精确分片覆盖，再发布完整checkpoint；最终核验三轮所有(epochs,row)覆盖。
每0.25epoch后首个完整optimizer step保存：第一轮1408/2815/4223/5630，第二轮7038/8445/9853/11260，第三轮12668/14075/15483/16890。
共12保存点、最多最新4份；最终保留12668/14075/15483/16890，最终model--step-16890.ckpt。
完整checkpoint保存model/optimizer/scheduler/scaler/tracker/每卡RNG和下一个epoch/epoch内step；epoch边界恢复从下轮0步开始，LR不重置。
新route/model/data/recipe身份拒绝旧三轮门控checkpoint续训；保持原runroot/最新完整点/日志备份回退/源码环境数据身份检查。

pdgpu-5090/8GPU/32CPU/256G/1node、原镜像/conda/覆盖变量不变；无额外GPU smoke。
一次后台producer仅复制当前AudioMCQ去重原文件，先shared后complete八rank MIN边界切换本地，三轮复用本地；恢复重新stage。
新tmpfs前缀 /dev/shm/mizar_l9_12x4_mcq_<token>，先stop/reap再owner cleanup保持。
新输出 outputs/RSmol/mizar_final_l9_12x4_audiomcq_gbs128_3epochs_5090/formal_<tag>/formal；log/mizar-audiomcq-x4.<tag>.<JOB>.log。
metrics新增epoch/epoch_update/epoch_fraction，epochN_rankR覆盖JSON，最终PASS要求3epochs/16890updates/完整覆盖与最终重读一致。

本地17项最终回归通过无skip：真实30层小Llama eager/SDPA native一致、42次调用/完整输出直传/四轮BPTT/冻结/因果；生产meta14160384参数。
实际小fixture CPU manifest CLI/全强弱及CoT/Reason0/原路径/幂等seal/index及tamper拒绝；720524三轮八rank分片/尾批/不同shuffle/quarter保存与507warmup。
实际trainer.train三轮完整小模型17行/6updates与epoch1边界恢复，最终权重bitwise相同；完整checkpoint游标与最多保留4份通过。
真实2进程CPU Gloo的GA4/GA1与独立参考更新一致；真实Bash prepare-only/CPU失败不提交/空格参数链/单vc和stop-reap-cleanup通过（flock/vc/torchrun明确模拟）。
已有官方18文件实际SHA/454严格load、原576维30层383prefix native bitwise对照、实际CED/projector合成静音683token四轮前向/反传/AdamW后冻结保持通过。
审计 audit_reports/mizar_l9_12x4_audiomcq_validation.json；旧报告保留历史配方。测试Torch2.15CPU/Transformers4.54.1，显式仅测试torchaudio未用native扩展适配沿用。
初轮两处测试夹具问题（误将Llama gate_proj判为新增gate、optimizer/model实例不配对）修复后最终回归通过，无生产参数因此改动。
未验证远程生产hybrid音频齐备、CUDA/NCCL/5090显存吞吐或正式收敛。README/sync_files/提交/恢复命令更新。
以下原RM全部字节完整保留。

## 2026-11-26：原始 Mizar baseline 与 L9–12/L14–17/L19–22 三循环 checkpoint 的四项4090评测

按用户确认计划新增隔离 code/RSmol/mizar_core3_eval/ 与 run_mizar_core3_eval_4090.sh。
四模型各 ReasonAQA5000/MMAU test-mini/MMAR/MMSU，共16项；先独立提交官方baseline，
三个训练产物由 --checkpoint 显式传入，不猜最终路径/步数、不扫描latest。
prepare_mizar_l9_12x3_strong_reason.sh 是训练数据准备入口，不是第四个训练模型，无需为评测重做prepare。

baseline严格复用官方final release loader，默认 models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt，
完整30层单遍及已训练CED/projector/LM；18文件release大小/SHA在远程CPU admission核验。
三训练目标直接使用各自未修改模型：L9–12保留A8 anchor/已训练共享gate，456状态；
L14–17与L19–22无额外注入，各454状态，30物理层/38次执行，全参数strict覆盖后冻结eval。
CPU核对同run runtime/route/contract/source、动态geometry及完整中间/最终游标；
GPU再查所有张量shape/dtype/finite/tied及全部冻结状态与官方来源逐值一致。
不套用训练resume已完成拒绝，不创建/恢复optimizer、scheduler或训练RNG。
实际L14–17合同为 mizar_final_l14_17x3_strong_reason_gbs128_v2_previous_training，
其 added_injection/gate/added_parameters 字段与L19–22不同，按各路线原semantic_contract核验，不能猜字段。

Reason复用原seed1234固定5000manifest及完整答案exact；MMAU/MMAR各原1000、原始生成直接交原官方scorer；
MMSU原5000、最新紧凑全小写四槽/null保持、原mmsu_answer_text_contains_v2及官方汇总。
FP32/eval/batch1/seed1234/无autocast/TF32off，mono32k/fixed20中心裁剪右补零/10+10/whole CED；
原EOS后缀、prompt256、PAD17/SEP-EOS0/prefix383/首logit382，原因果可见PAD、无KV、
逐token完整重算/argmax300/首EOS停止与raw decode保持；每题音频编码一次。
每完整forward按30/38真实轨迹审计，三循环状态及A8每次重建，不跨题/生成forward持久化。

source_manifest固定三路线全部Python/Shell及release manifest，CPU预检制作内容校验的不可变快照；
可显式 --source-revision 从匹配Git object读取，不切分支、不放宽源码；完整import图核对实际快照路径。
独立CPU预检每模型一次、失败四项均不申请GPU，单vc失败继续其余项并整体非零。
全部pdgpu-4090/1GPU/8CPU/32G/1node、原镜像/mellow_c8204d8及conda覆盖，无新增GPU smoke。
模型/任务/checkpoint输出隔离，同身份逐题持久化/剩余题续跑、坏完整行拒绝和完整覆盖后评分保持。
新compare核对共同数据/协议/共享代码/完整分母与原始预测，比较实际逐题输入指纹，记录30/38执行。

最终本地23/23 CPU回归通过，无skip。真实30层小Llama eager/SDPA三路线38次执行与独立递推、
训练后gate/state保持、383/683首/第二答案对齐与因果性、严格状态/冻结漂移拒绝通过。
实际加载控制流使用真实微型模型和明确生产资产/CUDA夹具，三路线strict覆盖后值保持；
原30层native baseline、真实旧baseline prefix接口及四模型四个实际生成适配器/每题一次编码通过。
真实SoundFile双声道16k解码/32k重采样/fixed20通过；四组5000/1000/1000/5000完整尺寸流程、
完整答案/原MMSU scorer/原始官方预测、13题中断后补齐及损坏/输入指纹差异拒绝为明确合成夹具。
真实Git HEAD源码读取/快照篡改拒绝、Bash入口/空格参数、16计划/4090资源/短job/预检失败不提交及vc失败继续通过。
初轮合同手写字段误配L14–17及漏审计元字段已修正为原合同生成器/明确属性；新增CUDA夹具种子接口遗漏已修正，
所有失败和定向复验保留validation_runs，最后完整23项复验通过；原训练/旧评测未为这些问题改动。
本地Torch2.9 CPU/Transformers4.54.1/torchaudio2.9，未用音频native扩展仅测试进程适配，生产无适配。
Windows沙箱临时目录/Bash signal pipe权限阻止后，获准沙箱外本地离线验收。
未读取远程生产权重/音频，未提交真实GPU作业或commit/push，CPU结果不代表CUDA/4090显存吞吐或实际分数PASS。

验收 audit_reports/mizar_core3_eval_validation.json；完整提交/单项/预览/CPU预检/恢复/日志/比较与精确Git命令见新包README。
28份交付见sync_files.txt，194项既有依赖见reference_dependencies.txt，需随GitHub代码同步已存在。
本次仅前置本条，以下RM既有全部字节完整保留。

## 2026-11-26：原始Mizar L19–22共享三轮，仅四层训练、无额外注入

按用户确认计划，从L9–12完整复制隔离code/RSmol/mizar_l19_22x3_strong_reason/，新增run_mizar_l19_22x3_strong_reason_5090.sh。
本地Windows改代码、GitHub中转，正式权重/音频/输出仍在远程Linux；未代为commit/push、连接远程资产或提交GPU作业。
任务开始4116份既有Python/Shell/JSON/Markdown的normalized-LF基线保持；RM原文完整保留。
工作期间另一会话前置L14–17记录，已在source_guard登记其精确前缀SHA并完整保留，未修改该路线。

完整严格继承官方Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt的454状态及已训练CED/projector/全部30层LM，18个release文件SHA/大小校验保持。
原始1-based轨迹1..18,(19..22)x3,23..30，共30物理层/38次block；同一组四层参数共享三轮，完整683位置。
仅19–22全部attention/MLP/层内norm可训练，每层3540096，合计14160384；其余145139456冻结eval，embedding/head保持绑定。
移除A8/anchor、共享audio_gate及循环间额外残差，不新增模型参数，state_dict恰为原454项。
前18层no_grad；三轮完整BPTT不detach；冻结后缀23–30、最终norm/head保留输入梯度，final norm仅第30层后一次。
无跨forward隐藏状态、KV或decoder gradient checkpointing；正常Transformer层内残差保留。

默认复用原data/Mizar/l9_12x3_strong_reason_v1封存数据与旧训练成员修复，数据格式/成员/顺序/原音频路径/CoT不改，不重抽题。
旧未使用才过滤、旧使用复用audit路径且失效拒绝；保留strong视图N配ReasonAQA N，总2N，实际N仍由远程prepare确认。
独立CPU index绑定本路线源码与原dataset seal；26份数据/加载/检查点等文件逐字节继承，完整train/optimizer_update与LR/分片数学AST保持。
原8GPU×micro4×GA4=最大GBS128、1epoch、ceil(2N/128)更新，动态尾批/空rank、无drop/补齐重复保持。
原AdamW max/min1e-4/1e-5、betas(.9,.999)/eps1e-8/WD.01与低维免decay、round5%warmup/cosine最后真实更新到min、GradScaler/GradNorm保持。
FP32/无autocast/TF32off；fixed20中心截20s/右补零/mono32k/10+10在线CED，126+1+256+300=683、PAD17/EOS0、原EOT截断和answer-only shifted CE保持。
每卡micro有效token mean→实际GA mean→活跃rank mean，空rank零任务损失仅通信；原seed及精确恢复分片保持。

pdgpu-5090/8GPU/32CPU/256G/1node，原镜像/mellow_c8204d8/Transformers4.54.1和conda覆盖，直接单次formal，无额外GPU smoke。
独立output/log/index/tmpfs及route身份；/dev/shm/mizar_l19_22x3_sr_<token>与真实路径守卫一致，先stop/reap producer再owner/token清理。
共享盘立即训练、后台4workers复制、完整文件/字节校验后optimizer边界8rank MIN共同切本地；恢复重新staging。
每500及真实最后步完整保存、保留3份；同本路线同源码/资产/数据/环境/配方在原run root完整恢复，旧L9–12及其他路线拒绝。
完整模型/具名AdamW moments/scheduler/scaler/tracker/8rank RNG/数据游标、日志备份回退、最终checkpoint重读及PASS完成判定保持。

最终本地28/28 CPU回归通过、无skip：真实小30层Llama eager/SDPA的native单遍、独立38次纯循环、三轮直接衔接、12次core输出任务梯度与冻结后缀输入梯度/因果性。
真实两进程Gloo GA4/尾GA1及空rank零任务梯度对照、完整AdamW恢复下一更新逐值一致、错旧route/额外gate库存/nonfinite/frozen漂移拒绝通过。
真实fixture manifest CLI/历史成员与等量/任意尾批/封存index幂等及篡改拒绝、实际CPU admission/旧路线拒绝、Bash单次提交/空格参数/生命周期与真实tmpfs赋值及owner清理守卫通过。
已有本地官方18文件SHA/454严格继承、原576维30层383prefix native逐值一致、真实CED/projector+683token静音CPU前反向/AdamW后冻结状态不变通过。
首轮一项新增断言误匹配标准MLP gate_proj已修正；次轮26项及官方模型通过，RM并发前置触发守卫，登记精确并发记录后最终完整28项及官方复验通过。
报告保留previous_runs及原因；生产代码未因这些测试/并发问题改变。Windows沙箱依赖/Bash权限获准沙箱外离线验收。
32份Python3.10语法、2Shell/4README Bash块、44份交付/34项既有准备依赖的Git可见性及新增文件whitespace通过。
测试Torch2.15 CPU/Transformers4.54.1，torchaudio2.9未用native扩展仅测试进程明确适配；生产代码无适配。
未验证远程真实音频齐备、CUDA/NCCL/5090显存吞吐或收敛，不将CPU结果写为GPU PASS。
报告audit_reports/mizar_l19_22x3_strong_reason_validation.json；完整Git同步、数据复用、正式提交/恢复和输出说明见新包README.md/sync_files.txt。
GitHub同步后，远程仓库根执行bash code/RSmol/run_mizar_l19_22x3_strong_reason_5090.sh；若原数据未成功prepare，先执行原prepare_mizar_l9_12x3_strong_reason.sh。
以下原RM全部字节完整保留。

<!-- mizar_l19_22x3_original_rm -->
## 2026-11-26：原始Mizar L14–17共享三轮，仅四层训练、无额外注入

按用户确认方案复制隔离 code/RSmol/mizar_l14_17x3_strong_reason/，新增 run_mizar_l14_17x3_strong_reason_5090.sh。
原L9–12训练/准备/测试/报告及本任务开始3950份tracked Python/Shell/JSON/RM的normalized-LF SHA保持；RM仅前置本条，原全部字节保留。
本地Windows改代码、GitHub中转，生产权重/音频/输出仍在远程Linux；未代为commit/push或提交GPU作业。

完整严格继承原Mizar final stage3_q4_seed20260905.ckpt的454项状态及已训练CED/projector/全部30层LM；18个release文件SHA/大小固定。
原始1-based层轨迹1..13,(14..17)x3,18..30，共30物理/38次block执行；四层共享一份参数、完整683位置与三轮BPTT。
只训练14–17全部attention/MLP/norm，每层3540096、合计14160384参数；其余145139456冻结eval。
彻底移除原A8捕获/共享audio_gate/两次音频残差注入，不新增模块或参数，state_dict恰为原454项。
前13层no_grad；冻结后缀18–30及最终norm/head保留输入梯度；三轮不detach、无KV/decoder checkpointing，final norm只在30层后一次。

数据准备及数据/音频/loss/optimizer更新循环逐字节或AST保持；默认只读复用原l9_12x3_strong_reason_v1目录及v2 sealed manifest。
旧训练未使用才过滤/已使用复用旧audit路径/失效拒绝规则保持；实际strong视图N配ReasonAQA N，总2N，不能把218引用当QA行数相减。
新CPU index绑定本路线源码和原dataset seal，独立cache；未访问远程旧hybrid/音频，不猜实际N。
8GPU×micro4×GA4=最大GBS128，完整1epoch、ceil(2N/128)真实更新，任意尾批/空rank、无drop或重复保持。
原1e-4/1e-5、round5%warmup/cosine最后更新到min、AdamW(.9,.999)/eps1e-8/WD.01与低维免decay、GradScaler/GradNorm保持。
FP32/无autocast/TF32off；fixed20中心截取/右补零/mono32k/10+10在线CED；126+1+256+300=683，PAD17/EOS0、原EOT截断及answer-only shifted CE保持。
每卡micro有效token mean→实际GA mean→活跃rank mean；空rank零任务损失只参加通信，原固定seed和分片保持。

pdgpu-5090/8GPU/32CPU/256G/1node、原镜像/mellow_c8204d8/Transformers4.54.1与conda覆盖保持。
单次vc formal，无额外GPU smoke；CPU seal/index/release/resume校验失败不提交。
独立output/log/index/tmpfs身份；/dev/shm/mizar_l14_17x3_sr_<token>与真实路径守卫同时更新，先stop/reap producer后按owner/token清理。
共享盘立即训练/后台4workers复制/optimizer边界8rank MIN共同切本地保持；恢复重建staging。
每500步及真实最后一步完整保存、保留3份；同本路线源码/模型/数据/环境/配方在原run-root完整resume，旧L9–12及其他路线拒绝。
模型/具名optimizer moments/scheduler/scaler/tracker/8rank RNG/游标、metrics备份恢复和最终checkpoint重读及PASS完成检查保持。

最终本地29/29 CPU回归通过、无skip：真实30层小Llama eager/SDPA的native单遍及独立38次纯循环对照，三轮全位置直接衔接、12次核心梯度及冻结后缀输入梯度/因果性。
真实2进程Gloo GA4/尾GA1与空rank零任务梯度对照、完整AdamW恢复下一步逐值一致、错旧route/库存/非有限/冻结state拒绝通过。
真实fixture manifest CLI/历史成员过滤/动态等量/任意尾批覆盖与恢复、seal/index篡改拒绝、Bash单次提交/空格参数/生命周期、实际tmpfs赋值与生产清理守卫通过。
已有本地完整官方18文件SHA/454继承、576维原30层383prefix native逐值一致、真实CED/projector+683token静音前向/反传/AdamW后冻结状态保持通过。
初轮新增POSIX测试夹具read_text漏接encoding参数，一项ERROR；修正夹具后完整29项及官方检查复验通过，历史失败保留previous_runs，生产代码不因此变动。
Torch2.15 CPU/Transformers4.54.1，已有torchaudio2.9未用native扩展仅测试进程适配；Windows沙箱依赖权限获准沙箱外离线验收。
报告audit_reports/mizar_l14_17x3_strong_reason_validation.json；实际生产音频/CUDA/NCCL/5090显存吞吐/收敛仍待远程，不将CPU结果写为GPU PASS。
代码、精确Git同步清单、数据复用、正式提交/完整恢复及输出位置见新包README.md和sync_files.txt。
GitHub同步后远程仓库根执行 bash code/RSmol/run_mizar_l14_17x3_strong_reason_5090.sh。
以下原RM全部字节完整保留。

<!-- mizar_l14_17x3_original_rm -->
## 2026-11-26：L9–12训练manifest的218个未解析引用，按旧训练使用证据过滤并动态配额

远程CPU prepare报218个不同音频引用未解析，不是218条训练行：202 AudioCaps候选文件缺失、15 MusicCaps旧S2记录不可用、1 Clotho basename歧义。
preparing rule日志是正常初始化；此次尚未提交GPU。不能根据这些摘要声称所有文件物理不存在或旧训练全部未用过。
用户确认条件：旧完整AudioMCQ训练也没用过的直接过滤，用过的按以前路径；保持ReasonAQA与实际strongAC视图等量。
参考完整训练入口确实使用 s1_stage1_hybrid_filtered.json（1477492行）及原hybrid audit TSV，而初版从1477976原S1取全部strong，遗漏历史筛选成员核对。

新增隔离 previous_training.py，CPU读取旧hybrid并核对参考配方各源计数；强制所有旧使用引用有exists audit，记录原manifest SHA及原始完整strong行多重集合SHA。
原始S1仍固定release SHA/1477976总量/756968 Reason候选/原strong视图上限；只保留旧训练使用的完整原strong行，原字段/直接答案/CoT保持。
仅证明旧训练未使用的行可过滤；相同引用旧训练用过而strong行变化/缺失则拒绝，避免把历史输入不一致伪装为可过滤。
旧用过音频必须复用旧audit成功路径；路径现在失效/空文件仍报错，不猜文件、换音频或静默删除。原有audit脚本/JSON映射及所有其它训练路线不修改。
新增 excluded_strong_rows.jsonl 保留逐行原索引/ID/视图/原行SHA/raw引用/旧未使用证据与原audit记录；报告直接/CoT过滤数与唯一引用数。
manifest schema v2绑定原始及旧训练来源、动态source/view配额、五文件SHA；旧失败building目录保留，不需清理即可重跑。
设保留strong视图N，Reason无放回采N，总2N；保留seed1234/canonical原录音轮转尽量多样，实际N须远程读取旧manifest得出，不凭218直接相减。
CPU日志 PREVIOUS_TRAINING_COUNTS/RETAINED_DATASET_COUNTS 输出精确配额；index/训练合同/checkpoint总步数/warmup/保存最后一步均按sealed真实行数派生。
去除固定325648/651296/5089在当前生产配额与完成判定中的限制；这些数保留为原始元数据上界与旧版回归参照。
8GPU×micro4×GA4=最大GBS128、1epoch、max1e-4/min1e-5及模型架构保持；尾批支持每卡1..4条及空rank，无sampler padding/drop。
空rank只用零损失上下文参加相同DDP通信，不计样本消费/任务梯度；真实非空rank的micro loss均值→实际GA均值，完整batch行为不变。
balanced连续分片/精确恢复覆盖所有真实行；最后LR到min。index key新增dataset seal，防同正文/不同历史筛选证据复用缓存。
路由身份v2_previous_training，旧配方不跨版本resume；本次CPU失败尚无新GPU训练，可直接重新生成manifest后新目录训练。

本地最终26项CPU回归全部通过无skip：原20项+旧成员过滤/等量、旧路径复用及失效拒绝、坏history/audit/变化行拒绝、动态配额/任意尾批覆盖与恢复、真实两进程Gloo空rank零任务梯度对照。
最终cache seal key补充后定向复跑实际CLI/manifest发布/幂等/index篡改拒绝与不同seal重建，1项通过，是已有用例加强而非新增独立用例。
第一次新增history测试揭示CoT为0时Counter与缺key字典比较问题，已按两种视图显式归零修复后最终26项通过。
当前报告 audit_reports/mizar_l9_12x3_previous_training_fix_validation.json；初版完整官方模型验证报告保留，模型/loader本次未改。
测试Torch2.15 CPU/Transformers4.54.1，沿用明确音频测试适配；未访问远程旧manifest/音频，实际保留数与218引用旧成员身份由远程新版prepare确认。
README/sync_files/执行及提交命令已更新；不自动commit/push或提交GPU。以下原RM全部字节保留。

## 2026-11-26：原始 Mizar L9–12共享三轮 + 零初始化音频门控，strongAC/ReasonAQA，GBS128

用户确认后实现隔离 code/RSmol/mizar_l9_12x3_strong_reason/，新增 prepare_mizar_l9_12x3_strong_reason.sh 与 run_mizar_l9_12x3_strong_reason_5090.sh。
本地改代码、GitHub中转，正式数据/权重/输出在远程Linux；未自动commit/push、连接远程音频或提交GPU作业。
原有4231份代码/测试/RM基线normalized-LF SHA校验保持；RM仅前置本记录，原全部字节保持。

完整严格继承 models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt 的454状态，包括已训练CED/projector/全部30层文本基座。
18个官方release文件SHA/大小固定，454源状态加载完才新增门控，不混用旧recursive checkpoint或随机projector。
原始1..8、(9..12)x3、13..30，共38次block；四层共享参数和两次共享Linear(576,1)，weight/bias全0初始化。
A8为第8层raw post-block音频前126位置；第2/3轮前 H_audio += tanh(Linear(H_audio))*A8，条件取前一轮音频输出，注入下一轮输入。
SEP/text不直接加残差，final norm仅第30层后一次；无跨forward状态、无KV/decoder checkpointing，三轮完整BPTT。
仅四层全部参数含norm和577门控参数可训练，总14,160,961；其余145,139,456原参数冻结且eval，冻结后缀保留输入梯度。

CPU manifest从原始S1 1,477,976行开始，强制原始release SHA，拒绝旧filtered S1。
strongAC直接答案257,892 + 额外CoT 67,756 =325,648；ReasonAQA单原始音频候选756,968，固定无放回抽325,648，总651,296。
Reason按canonical resolved原录音分组，seed1234打乱组内QA并逐轮打乱活跃录音组、每组每轮一个QA，尽量覆盖不同录音。
历史filepath2只是处理后第二chunk；只有显式第二原录音才算多音频。保留完整原字段/答案/两个strong视图。
复用原hybrid audit TSV与原audit脚本/S2 hybrid mapping fallback；任一候选原音频缺失/空/歧义或配额不符即停止，不丢行、不补weak。
train.jsonl、selection.jsonl溯源、resolved_paths.tsv、dataset_report.json与SHA seal；tokenizer实际审计直接答案/CoT截断及EOS。
独占准备锁、同级临时目录、no-replace整体发布、相同输入幂等复用；失败保存unresolved_audio/failure诊断。

一完整epoch，8GPU x micro4 x GA4=GBS128；5088个128更新+最后32条/每卡4条GA1，共5089次更新，无drop/补齐重复。
AdamW全训练参数max1e-4/min1e-5，原betas/eps/WD及bias/一维免decay，warmup254，cosine最后真实更新到min。
保留FP32/无autocast/TF32off、GradScaler/GradNormTracker；原fixed20中心截20s/右补零/mono32k，在线CED/projector。
126audio+SEP1+prompt256+answer300=683，PAD17/EOS0、空格EOT后截断、不补回EOS、PAD可见因果注意力、answer-only shifted CE。
每卡micro有效token mean→实际GA mean→DDP mean，尾批独立缩放。

提交前CPU seal/index/完整release/resume身份校验，失败不提交；正式单次vc，pdgpu-5090/8GPU/32CPU/256G/1node，原镜像及mellow_c8204d8。
原conda路径与RSMOL_CONDA_BASE/MELLOW_CONDA_BASE覆盖，要求Transformers4.54.1，不改共享环境，无独立GPU smoke。
单后台producer仅复制本manifest去重原音频到 /dev/shm/mizar_l9_12x3_sr_<token>；先读共享盘，完整文件/字节校验后八rank在optimizer边界MIN集合切换本地。
检查producer/心跳/错误/容量；退出先stop/wait回收producer再按root/token清理。恢复重新staging。
每500步及5089完整保存，保留3份；full resume严格同源码/数据/模型/环境/recipe/命名optimizer moments/游标/RNG/scaler/tracker。
恢复保持原外层run root，拒绝比所选更新的完整checkpoint；日志较新行/partial tail原样备份后回退，损坏完整行拒绝。
逐角色任务梯度/冻结state、真实source probes、确定性完整分片消费、final checkpoint重读与PASS训练报告；rank/shell/staging失败诊断保留。

本地20项最终测试通过、无skip：真实小30层Llama eager/SDPA、单遍native一致、38次调用、注入数学/位置/共享、因果性、完整BPTT/冻结保持。
实际2进程CPU Gloo的GA4/GA1与独立单进程平均梯度更新对照通过，完整checkpoint恢复下一步bitwise一致，坏moment/optimizer/scheduler/frozen state拒绝。
实际小fixture manifest CLI原子发布/幂等/immutable index与seal篡改拒绝；shared-first完整切换/失败检查、真实Bash参数及成功/训练失败/配置失败stop-reap-cleanup时序通过。
本地完整原始S1实际抽样核验651296精确各源配额；以raw路径分组覆盖全部53,673录音组，远程仍须canonical resolved路径重分组，不声称声学去重。
本地已有官方完整18文件SHA/454继承和原576维30层383prefix native bitwise对照通过；真实CED/projector+683token静音fixture前向/反传/AdamW后所有冻结状态不变。
审计 audit_reports/mizar_l9_12x3_strong_reason_validation.json；代码/执行/提交/恢复/输出位置见包内README与sync_files.txt。
测试Torch2.15 CPU/Transformers4.54.1；本地torchaudio2.9 native扩展ABI不匹配，仅测试进程禁用未用native扩展，SoundFile和PyTorch音频变换真实，生产代码无适配。
Git Bash无flock，生命周期夹具显式模拟flock/vc/torchrun；未验证远程音频齐备、CUDA/NCCL/5090显存吞吐和实际收敛。
以下原RM全部字节完整保留。

<!-- mizar_l9_12x3_strong_reason_original_rm -->
## 2026-11-26：MMAU/MMAR完整1000题静音tracing，所有合格题及三类/四层级分析

用户确认计划后新增隔离code/RSmol/mizar_benchmark_tracing/，共用干预核心、两个独立任务和输出。
顶层run_mizar_mmau_tracing_8gpu_5090.sh、run_mizar_mmar_tracing_8gpu_5090.sh，各单次vc提交。
原ReasonAQA tracing/500、cutoff/剪层/CKA/MMAU/MMAR ablation与letters实验138份已有文件normalized-LF SHA保持，158份复用依赖固定。
本地代码经GitHub同步；生产权重/数据/输出仍在远程，无自动commit/push或GPU提交。

两个任务各自完整1000题、seed1234打乱、每卡125题；每题均运行clean与silence，不因clean错误而跳过silence。
筛选clean字母正确、silence字母错误且目标Pclean>Pcorrupted，保留所有合格题N，无500/1000入选配额、不补抽/分组配额。
完整1000题总体/分组clean与silence正确率、入选数/比例及clean_wrong/silence_correct/nonpositive_gap/unmapped_gold独立报告。
选中manifest按原数据行号排序，保留ID/原slot/分类/shuffle位置/全部候选分数和概率/audio和cache指纹。
N=0正常完成筛选，NO_QUALIFYING退出0、输出完整基线/空manifest/解释性空图，不做patch；分组n=0指标null/图例标注。

MMAU沿用用户27子类映射perception331/semantic230/reasoning439；MMAR Signal43/Perception404/Semantic412/Cultural141。
MMAU原重复正文和GT槽位保持；MMAR原2..6等实际选项数及空槽保持，不连续重标，空选项null且不参与argmax。
MMAR原5题GT无法唯一映射，完整1000分母计错，target概率null、按unmapped_gold排除RR。
真实release tokenizer核对各实际纯小写候选字母为单普通token，完整词表FP32 log_softmax后FP64 exp，不在候选内重归一化。
readout的log_softmax在CPU计算，沿ReasonAQA tracing实现；派生exp最多8FP64 ULP，预测/logprob等严格，原保存数值不重写。

完整原Mizar final encoder+已训练projector+30独立LM，FP32源参数/BF16前向/FP32 frontend/TF32off/math SDPA或eager。
SoundFile/resample32k→mono/first20右补零；compact lowercase+空格EOT后prompt256/可见PAD17，126audio+SEP0+256text=383，首logit382。
静音640000零波形正常通过encoder/projector，每rank编码一次并校验8卡一致，不直接清零音频embedding。
clean临时capture30个raw post-block文本状态，位于residual后/final norm前；仅合格题缓存精确30x256x576与两份prefix，保留实际dtype。
各题audio encoder/projector一次并跨30个patch复用；每层从原静音prefix重启，仅目标层text127:383全256位置替换，audio/SEP保持。
每卡按selected_index%8分配动态N题，同一layer全部N合并/打印/barrier后换层，共30N patched决策。
空分片仍参加通信；N<8时空分片rank读取共享入选probe做同样controls，不伪造其patch行。
原capture/native、layer0=corrupt、layer30=clean、cache=重新capture完整logits/raw状态bitwise对照保持。
逐题RR用FP64概率差比，不加epsilon/clip；总体逐题mean及median/分位/2000次点态bootstrap CI、各组RR/acc/P、小gap敏感性保持。
最后层是readout正对照；所选clean100%/silence0%为条件子集，不当完整1000分数。

CPU prepare优先只读旧letters_eot1000 manifest；缺失则只读旧body manifest转换；两者均缺失则调用旧builder只把返回数据写新路径。
不调用旧ensure在旧目录发布文件，不静默忽略存在但不匹配的旧manifest。
新输入data/mizar_{mmau,mmar}1000_tracing_inputs_v1/manifest.json；CPUprepare不做模型前向。
GPU任务自动ensure输入、完整筛选、动态N缓存/逐题恢复/最终审计与作图；支持--stage select后同目录默认命令续跑。
原子缓存、每题fsync journal/partial tail恢复、完整1000覆盖、输出锁及模型/源码/数据/环境身份保持。
源冻结保留逐行原记录SHA；rank错误error/traceback/phase收集到analysis_report，减少只看到torchrun汇总。

pdgpu-5090/8GPU/64CPU/256G/1node，原镜像与mellow_c8204d8/原Miniconda和RSMOL_CONDA_BASE覆盖。
默认输出outputs/RSmol/mizar_{mmau,mmar}_tracing_all_eligible_letters_eot_bf16_8gpu5090_v1。
日志code/RSmol/log/mizar-tracing-{mmau,mmar}.<tag>.1.log；提交打印log_path/input/selected manifest/output/submission报告。
输出screening_summary.csv/完整1000逐题clean与silence/selection_report/selected manifest/30层逐题RR和统计/总与分组CSV及JSON。
四类图各PNG+SVG：总体RR、三/四组RR与patched acc、目标概率、分组热图；热图组内按原序号并输出heatmap_row_map题ID对照。
正文README/sync_files列CPU准备、两个提交、预览、仅筛选、续跑、自定义路径、日志和CPU重画图命令。
26份交付文件；analysis_report.json PASS才代表正式完成，N0为NO_QUALIFYING。

本地24项独立验证通过无跳过（首轮23+补充新续跑用例1；其余补充是加强/复验已有用例），报告audit_reports/mizar_benchmark_tracing_validation.json。
真实2000题本地标注与release tokenizer核对三/四分母、MMAU27重复正文、MMAR4空槽/5未映射/最多6槽；所有候选决策与旧字母口径逐题一致。
实际CPU input build/ensure/load与只读旧manifest/原路径不发布、两种fallback、wrong identity/字段篡改拒绝通过。
真实30层小Llama eager/SDPA/BF16/383prefix捕获与全部30patch、不累积/权重保持，空槽/完整词表概率；另原576维/缓存/SoundFile first20回归通过。
完整1000筛选/各组基线/任意N分片/同层8rank汇总/续跑不重做/idle probe/N0图表与审计为明确合成模型/分布式夹具。
真实合格题BF16缓存保存读取及prefix音频/文本隔离通过；完整两数据集30层finalizer/CSV/8图与分组heatmap映射合成验证通过。
真实Bash三个入口/conda激活夹具/两个单次5090提交参数及mock vc参数链/Python3.10/whitespace通过，图形布局目视核验。
补充测试从仓库根python -m unittest曾因既有code包遮蔽stdlib code导致Torch导入失败，从tests目录复跑两项均通过；未改生产路径。
Windows依赖权限沿已有工具批准在沙箱外CPU验证。QA图明确标记合成，不代表Mizar实际性能。
未读取远程生产权重/音频/实际结果、提交GPU作业或commit/push；实际CUDA与八卡通信仍需远程运行。
本条仅前置，以下RM原有全部字节完整保留。

## 2026-11-26：ReasonAQA500读取旧概率的严格浮点字典比较修复

用户执行CPU manifest时，source.validate_history调用原validate_outcome→validate_decision报
saved prediction/probability disagrees with candidate logprobs，尚未进入模型推理或写新500题manifest。
代码定位为原saved dict与当前math.exp(logprob)复算dict要求完全相等，exp派生概率1个FP64 ULP差异即可误报。
本地明确复现1ULP错误并验证修复；未收到原始失败decision，故不声称已独立测得该远程样本的具体误差。

仅改隔离mizar_reasonaqa_tracing500，新增validation.py；原1000题包与顶层入口17份SHA继续保持。
对candidate_probabilities/target_probability仅允许8个FP64 ULP以内，0/1端点仍完全相同，不用固定absolute epsilon。
候选logprobs/target_logprob、预测标签/索引/argmax/tie/正确性/字段类型仍严格；目标概率必须等于其自身保存的GT候选概率。
source导入继续验证seal、原数据/模型/环境、shuffle/acceptance顺序、原3项资格/gap/缓存SHA；不调用旧严格浮点dict检查。
保留原保存概率，不重算或改写源文件；first500成员、clean/corrupt基准与原gap不变。
超过容差报具体题ID/字段/保存与复算decimal和hex/abs误差及允许值，真实预测/logprob不一致仍拒绝。

同类校验同时用于500题runtime.controls与逐题结果；核心load/forward/from_logits等仍直接引用原实现，BF16/题面/音频/patch无改。
full-vocabulary logits与raw clean states对照仍逐值相等；candidate logprobs仍要求与原保存值完全一致。
第30层按其实际概率计算并保存RR，允许exp末位造成接近1的偏差，不把RR强制设1、不增加epsilon或clip。
源码身份包含新validator；默认报错发生于新500题输出claim前，更新整个500题包后重跑原CPU manifest/提交命令即可。
旧5000题pool、730份筛选缓存不重建、不重跑；未放宽原模型/源码身份或样本资格。
README与sync_files已补新模块和修复说明。

本地完整19/19通过无跳过：1/4/8 ULP通过与32ULP拒绝、tiny probability/零端点/错误预测/logprob/目标内部对应拒绝；
原5000/730合成来源带ULP偏移的实际first500导入与原目录全文件保持，15000调度/续跑/汇总/八图回归通过。
真实30层Llama BF16/SDPA原始状态和logits对照加保存概率1ULP偏移通过，修改完整logits/raw states仍拒绝；原eager/SDPA/576维回归保持。
第30层RR末位偏差实际保留、不同目标概率仍拒绝；原1000源码17份保护/新98份身份、Python3.10和真实Bash入口解析通过。
初次4个stdlib测试body无断言失败，但Windows沙箱临时目录清理报错，获准沙箱外完整复验，记录见validation report。
报告audit_reports/mizar_reasonaqa_tracing500_validation.json保留先前验证；测试产生的4份SVG时间戳/随机ID变动已还原为任务开始版本。
未读取远程原始失败decision/生产缓存或权重、提交GPU作业、commit/push；生产重试仍待用户远程执行。
本条仅前置，以下RM原有全部字节完整保留。

## 2026-11-26：ReasonAQA静音tracing仅730题合格，按用户要求复用first500继续

用户回传selection_report（本地未独立读远程）：原5000全部扫描，clean正确3921，silence_correct3190、clean_wrong1079、
accepted730、nonpositive_probability_gap1；原required1000故INSUFFICIENT_QUALIFYING，属于按既定筛选规则停止。
来源identity dd573f8424a5a3b692620b5f80e84e9abf79b586c5709a32058e7f908b652ff4。
用户明确“不够1000条就用500条”，本轮新增隔离mizar_reasonaqa_tracing500及run_mizar_reasonaqa_tracing500_8gpu_5090.sh。
原mizar_reasonaqa_tracing全部16份生产源码/guard与原顶层入口共17份normalized-LF SHA固定且不改，保留旧pool和缓存身份。
不直接修改原TARGET/放宽旧身份，也不删除旧结果或重新筛选5000题。

CPU只读验证原run_config/完整screening journal/selection_report及sealed内容、原5000pool、环境、严格完整Mizar模型资产。
按原shuffle中acceptance_order0..499选最先合格500，再按test.json原行号排序输出独立selected/manifest.json。
保留原来源/打乱/入选序号、候选token/题面、clean/corrupt概率和原audio/cache SHA；不按置信度或原行号先挑500。
逐一核对原30x256x576原始文本状态和prefix缓存SHA/shape/dtype，GPU仍用原cache.load直接读取旧路径，不复制数GiB状态。
来源目录完全只读；partial-tail也报错保留，不调用旧journal恢复器截断。旧任务合格<500则明确拒绝。

模型加载/原BF16干预/评分/capture/control函数直接引用原包，FP32源参数/前20s/EOT/单字母或yesno/全词表目标概率保持。
第1..30层post-block替换全部text127:383，audio/SEP静音状态保持，各条件独立，不跳层或累积；逐题RR不clip不加epsilon。
每rank62/63题，同一layer精确500题汇总/打印/barrier后切下一层，共15000次patched前向，无新筛选前向。
每卡一次静音encoder对照和首题clean重现/native/layer0/layer30/cache对照仍执行，不冒充完全零额外对照计算。
总/三来源RR与acc/概率/CI/small-gap敏感性、CSV/JSON和8份PNG+SVG；新coordinator把rank error/traceback/phase并入analysis_report。
新身份引用原identity及三份原结果文件SHA，源码/模型/环境/来源数据与缓存不符拒绝续跑。

默认来源outputs/RSmol/mizar_reasonaqa_tracing1000_letters_eot_bf16_8gpu5090_v1；
默认新输出outputs/RSmol/mizar_reasonaqa_tracing500_letters_eot_bf16_8gpu5090_v1，manifest在selected/manifest.json。
CPU可执行bash mizar_reasonaqa_tracing500/run.sh manifest，提交bash run_mizar_reasonaqa_tracing500_8gpu_5090.sh，作业内自动导入也支持。
单次vc pdgpu-5090/8GPU/64CPU/256G/1node/原镜像/mellow_c8204d8，原Miniconda路径与RSMOL_CONDA_BASE覆盖保持。
日志code/RSmol/log/mizar-reasonaqa-tracing500.<tag>.1.log，提交打印确切路径，新analysis_report PASS才表示正式完成。
新包README/sync_files含执行/自定义来源和输出/续跑/重画图命令；旧1000任务无需重提。

本地13项通过无跳过（主12+新增审计1），报告audit_reports/mizar_reasonaqa_tracing500_validation.json。
模拟原5000/730有序筛选结果的真实导入/CPU出版/原目录全文件字节保持/first500/不足与partial拒绝、源码固定通过。
实际500题x30层worker调度/62与63分片/续跑不重复/15000结果finalizer与8图为明确合成模型夹具。
真实BF16全状态保存载入保持；原真实30层小Llama eager/SDPA/BF16、576维、独立patch/控制和SoundFile测试复验通过。
新增最终audit实际检查500份原来源引用、八rank完成记录和错误63题覆盖拒绝（大缓存/音频字节为明确夹具）。
真实Bash两入口/conda激活夹具/5090单次提交参数解析、Python3.10语法通过，合成图布局已目视验证。
未修改旧生产源码、远程提交作业、commit/push或读取真实730题生产缓存/模型；生产CUDA与实际缓存重现待远程执行。
本条仅前置，以下RM既有全部字节完整保留。

## 2026-11-26：ReasonAQA 原5000中筛选1000合格题并逐层静音activation patching

按用户确认方案新增隔离code/RSmol/mizar_reasonaqa_tracing/与run_mizar_reasonaqa_tracing_8gpu_5090.sh。
参考Causal Tracing of Audio-Text Fusion in Large Audio Language Models arXiv2603.13768v2 §3。
原有剪层、cutoff、CKA、MMAU/MMAR ablation及字母EOT实验120份源码normalized-LF SHA保持，63份复用依赖固定。

候选池只读原reasonaqa/test_close_ended_sample5000_seed1234_v1/manifest.json，不重新采样。
CPU prepare冻结同5000题的原ID/原行号/原始题/音频路径与stat/tokenizer/源码指纹及独立seed1234打乱序列。
本地真实元数据重建与release tokenizer核对：AudioCaps2108/Clotho2369/yesno523，4477 MCQ显式GT标签可解析，全部单录音。
原5000 ID列表SHA183e34c08d34b4ad0e82088da250cf5f952425208527a4953acefda7a34e74d1。
GPU按shuffle顺序筛选clean分类正确、silence分类错误且Pclean>Pcorrupted；继续筛满1000，不保留仅clean正确但不合格题。
每窗口min(8,还差合格数,池剩余数)，按序commit，无超过最终第1000合格题的推测前向。
5000用尽不足1000则保存INSUFFICIENT_QUALIFYING/逐题审计，失败退出、不开始正式patch、不补池或放宽。
最终selected/manifest.json按原test.json行号升序，保留源5000索引/shuffle位置/入选顺序/新selected_index与clean/corrupt分数。

完整原Mizar final encoder/已训练projector/30物理层；FP32源权重/BF16前向/FP32frontend与log_softmax/TF32off。
SoundFile/32k先resample再mono/first20右补零；原紧凑小写题面加空格EOT后右截断补256、可见PAD17保持。
126audio+SEP0+prompt256=383，首答案logit382、无KV/无生成；MCQ纯a/b/c/d、yesno纯yes/no，强制单普通token。
正确目标概率取全词表softmax，不在候选集内重新归一化；分类则候选argmax。

静音在640000点波形上归零后正常通过encoder/projector，每rank/device一次编码并验证八卡一致；不直接清零音频embedding。
Clean capture为每个block残差后的原始text[127:383]全部256位置、final norm之前的30x256x576未池化精确tensor。
包括EOT/可见PAD/readout，保留实际dtype（BF16约8.24GiB/1000，若FP32约16.48GiB），逐题lossless cache。
Patched每次从原静音prefix重新执行所有30层，仅目标1..30层post-block替换text，不碰audio/SEP、不累积patch。
FP32 log_softmax后FP64 exp/概率差/RR，不加epsilon/不clip；mean逐题RR为主，不是ratio of means。
另median/分位/按题bootstrap2000次点态CI/三来源分组/acc/目标概率与small-gap敏感性，保留负值和>1。
每rank首个入选题检查capture/native完整logits逐值一致、layer0=corrupt、layer30=clean、cache=重新capture。
最后层RR1是readout恢复正对照；所选clean100%/corrupt0%是条件子集，不当全5000准确率。

一个vc作业内串行筛选及patch；pdgpu-5090/8GPU/64CPU/256G/1node，原镜像与mellow_c8204d8。
每rank125入选题，prefix和full clean states本卡复用，所有卡同一layer合并/打印/barrier后切下一层，正式30000 patched决策。
完整候选outcome与commit/逐题patch日志fsync/partial-tail恢复/身份与音频缓存SHA/输出锁；--stage select支持先选后续跑。
默认pool data/reasonaqa/mizar_tracing_pool5000_letters_eot_v1/manifest.json；原5000manifest保持。
默认输出outputs/RSmol/mizar_reasonaqa_tracing1000_letters_eot_bf16_8gpu5090_v1；入选manifest在selected/manifest.json。
日志code/RSmol/log/mizar-reasonaqa-tracing.<tag>.1.log，提交打印确切log_path/output/manifest及submission report。
输出逐层逐题RR/三来源统计/CSV/JSON/八份PNG+SVG，最终看analysis_report.json的PASS；CPU画图可独立重跑。
执行/提交/分阶段/续跑/日志命令见新README；24份交付文件见包内sync_files.txt。

本地最终19/19通过、无跳过：真实release tokenizer/原5000元数据与pool builder/只读源保证、真实SoundFile/重采样first20；
真实30层小Llama eager/SDPA/BF16全部30个patch与独立hook、参数不变/不累积/raw末层norm之前/0和30对照；另真实576维短prefix。
真实full-state tensor保存载入/错身份和篡改拒绝，pending candidate缓存续用不重复前向。
八rank线程集体调度夹具验证实际select循环按序停满/不足/续跑，实际层调度125/1000题与续跑不重复。
完整30000逐题结果、240rank日志、finalizer/audit/图表为明确合成数据；大缓存/音频字节审计在该全量夹具中显式替代。
真实Bash两入口/conda激活夹具/单次vc参数链mock通过，Python3.10语法通过，合成图布局已目视核对。
报告audit_reports/mizar_reasonaqa_tracing_validation.json，保留初轮18项PASS；Windows依赖沙箱不可读后获准沙箱外CPU验证。
Torch2.9.0+cpu/Transformers4.54.1，不冒充远程生产Mizar/CUDA或真实八卡通过；未读生产权重、commit/push或提交GPU作业。
本条仅前置，以下RM既有全部字节保留。

## 2026-11-25：三个诊断实验重做为小写字母likelihood＋题面EOT，八卡5090

用户明确只改评分目标为纯小写a/b/c/d（无括号/空格/正文/答案EOS）以及原题面结尾加空格EOT再tokenize；
保持BF16与前20秒音频，不改成中心裁剪。新隔离code/RSmol/mizar_letter_analysis/及三份letters_eot_8gpu_5090顶层入口。
原cutoff1000、MMAU/MMAR ablation、CKA和剪层搜索相关88份文件normalized-LF SHA保持。
复用原strict final模型加载、audio+SEP mask和单层bypass、prefix音频前处理，不修改训练或正式四项eval。

三个任务分别为cutoff：原四数据集各250的同1000题，baseline+cutoff3..28共27组；
mmau：原完整1000题，baseline+单层bypass2..29共29组；mmar：原完整1000题，同29组。
合计85000决策、三个独立vc作业，全部pdgpu-5090/8GPU/64CPU/256G/1node，原镜像与mellow_c8204d8。
每题只做一次383prefix LM forward，读取logit382的FP32全词表log_softmax，再gather各有效纯字母token。
实际release tokenizer逐字母严格检查单普通token/不同ID；题面仍包含原问题与全部选项，不增加only-letter指令。
MMAR非四选保留真实a..f等范围，空槽排除argmax且分数null，5题未映射GT继续保留分母计错。
MMAU重复正文选项/原GT槽位保持，MMAR官方matcher诊断另存，不替换主字母acc。

原冻结manifest直接转换，不重抽题、不改ID/顺序/audio/GT/选项，记录原manifest SHA和每行source_record SHA。
原manifest存在时只读；不存在则先调用原生成器，保留原seed与资格规则。
题面整体追加' <|endoftext|>'后按正式tokenize右截断/补齐到256；若长题截掉EOT，显式保存retained/truncated审计。
新manifest data/mizar_{cutoff,mmau,mmar}1000_letters_eot_v1/manifest.json，独立新prefix缓存/输出/逐题恢复身份。
cutoff仍rank0预计算1000prefix，各卡读；ablation仍每卡预计算本卡125题；各题encoder/projector一次跨全条件复用。
八卡同一条件精确合并1000后打印/落盘/barrier再切换，源FP32/BF16/first20/可见PAD17/原RoPE/无KV/无自由生成保持。

MMAU按原用户27子类8/8/11映射直接输出perception331/semantic230/reasoning439；
MMAR保留Signal43/Perception404/Semantic412/Cultural141。分组按题数加权，不平均子类准确率。
输出总与分组baseline/逐层acc/百分点变化/paired题目/并列排名/CSV/JSON/PNG/SVG及原始字母分数。
默认outputs/RSmol/mizar_audio_sep_cutoff03_28_1000_letters_eot_bf16_8gpu5090_v1，
mizar_mmau_ablation_letters_eot_bf16_8gpu5090_v1，mizar_mmar_ablation_letters_eot_bf16_8gpu5090_v1。
日志code/RSmol/log/mizar-letters-{cutoff,mmau,mmar}.<tag>.JOB.log，提交打印确切路径。
三个提交命令、CPU manifest转换、日志与输出说明见新README；精确同步清单sync_files.txt。

本地19/19通过、无跳过：真实30层小Llama eager/SDPA/BF16/383prefix单次forward字母gather与独立log_softmax逐值一致，
无正文token输入、空槽最高logit仍排除、全部27/29轨迹、native baseline、3/28 cutoff隔离及2/29 identity等价、参数保持。
原题单转换与旧文件字节不变、真实本地MMAU27重复正文/三分类和MMAR4空槽/5未映射GT/六选项核对。
真实BF16本卡125prefix和rank0缓存复用、错身份/篡改拒绝；三任务85000决策/8rank条件调度及续跑为明确合成夹具。
Python3.10、真实Bash三入口/5090资源/单次提交计划通过，合成图布局目视核对；Windows隔离Torch沙箱问题获准外部CPU复验。
报告audit_reports/mizar_letter_analysis_validation.json保留早期沙箱失败；Torch2.9.0+cpu/Transformers4.54.1。
未读取远程生产权重或音频、commit/push或提交GPU作业；实际release tokenizer与生产CUDA仍由远程验证。
本条仅前置，以下全部原RM字节保留。

## 2026-11-25：完整Mizar在MMAR1000上四类逐层消融，八卡5090隔离实现

按用户确认计划复制原mizar_layer_analysis_common/mizar_layer_ablation，新增独立
code/RSmol/mizar_layer_ablation_mmar/及run_mizar_layer_ablation_mmar_8gpu_5090.sh。
旧MMAU ablation、CKA、cutoff1000与剪层搜索的71份相关文件normalized-LF SHA保持，原输出保留。

完整原始Mizar final checkpoint含encoder/已训练projector/30层LM；baseline首先，随后逐一跳过原始1-based第2..29层。
消融h_out=h_in、每次从全层集合选一层、不累积、第1/30层保护；共29条件/29000题目决策。
原FP32源参数、BF16 autocast、音频前处理/log_softmax/正文累加FP32、TF32off保持。
原SoundFile/resample32k→mono→first20右补零、compact lowercase/无问题EOT/prompt256/PAD17可见/
126audio+SEP0+prompt256=383/原position与RoPE/无KV保持；全文正文sum likelihood取最大，不自由生成。

新manifest完整保留MMAR原1000唯一ID及顺序，固定category Signal43/Perception404/Semantic412/Cultural141。
4题空选项保留原标签槽位，不重新编号；空token/分数null、不进入argmax，避免空串sum0错误胜出。
5题GT不能唯一对应完整选项，gold_index=null、保留全1000分母并计错，不自动模糊修正。
主分数严格选项匹配；同一likelihood预测另交原MMAR string_match，诊断分数单独保存，不进入主图/排名。
官方函数实现保持，与远程evaluation.py函数AST核对且绑定文件SHA，不执行外部脚本。
CPU冻结音频/文件/token/源码指纹并记录异常，缺音频/坏题/分类分母变化硬失败、不筛除补抽。

单次vc提交pdgpu-5090/8GPU/64CPU/256G/1node，原镜像和mellow_c8204d8。
每卡global_index%8负责125题，encoder/projector prefix各题一次、本卡BF16缓存跨29组复用。
8卡共同完成一个条件，rank0完整合并落盘/打印总体四类分数/barrier后切换；durable journal逐题续跑。
默认manifest data/mizar_mmar1000_likelihood_v1/manifest.json；输出outputs/RSmol/mizar_mmar_ablation_bf16_8gpu5090_v1。
日志code/RSmol/log/mizar-layer-ablation-mmar.<tag>.JOB.log；生成/提交/查看日志命令见新README。
四类主acc/baseline差值/paired变化，另子类/模态/语言统计；原始分数/总与四类并列排名/CSV/JSON。
三组图均PNG+SVG：总体双面板、四类绝对acc、四类相对baseline百分点变化，标明43/404/412/141分母。

本地19/19通过、无跳过：真实30层小Llama eager/SDPA/BF16 baseline/全部28跳层轨迹/参数不变/
第2与29层跳过-输出替换等价、prefix383逐token LL右移和空槽不推理；125份真实BF16缓存一次编码及续跑。
真实SoundFile stereo16k→32k/mono/first20/pad；只读真实MMAR1000标注与原matcher AST核对。
1000题manifest builder/异常GT与空槽/来源和篡改拒绝、8rank同条件与恢复、完整29000决策/六图为显式合成夹具。
Python3.10语法、真实Bash+模拟conda的5090资源/一次提交检查通过，合成图布局目视检查完成。
Windows沙箱隔离Torch读取异常获准沙箱外复验，早期失败保留在报告。
报告audit_reports/mizar_layer_ablation_mmar_validation.json；精确同步清单包内sync_files.txt。
未读取远程生产权重/音频、commit/push或提交真实GPU作业；生产CUDA与八卡运行仍待远程验证。
本条仅前置，以下全部原RM字节保持。

## 2026-11-25：三循环audio8/read3与HRM BP5 PrefixLM两个checkpoint的八项4090评测

用户按只读调研计划授权实现，新增隔离code/RSmol/ced_eval_20261125/及run_ced_eval_20261125_4090.sh。
固定targets.json完整绑定用户两个step34626：audio8/read3 formal_20261124_002101117114787，
HRM PrefixLM formal_20261124_184519991594964，各ReasonAQA5000/MMAU test-mini/MMAR/MMSU，共8项。
别名audio8_read3、hrm_prefixlm；pdgpu-4090、1GPU/8CPU/32G/1node、原镜像和mellow_c8204d8。
未修改既有训练/共享模型/旧评测源码，不将其他旧任务加入本轮。

固定训练源码：audio8/read3=2c7d4485979f30f23c96196533c6de5914c47888，code_sha256=3c3a4619d3ab1d019fd9d3fc288c2f4ebe81d8d6c3cf707b3af64d18627e7f2d；
HRM=ee6f54474139fa5b86a9189617a2d2c0b9c84492，code_sha256=0814180a787ab29ab439403fff5befab2b870e4bb6b6cb50b07822a677c1ca58。
仅git archive必要训练后端/共享源码到log/ced_eval_20261125_sources；逐文件/库存/训练与共享摘要固定核对。
HRM还绑定原顶层训练Shell及两HRM包所有直接Python源码/原raw_audio审计器；真实合同导入与完整训练导入图路径核验。
CPU预检每checkpoint一次header/step/合同/geometry8,4,4/源码/真实函数签名后再申请GPU；失败目标不申请，其余继续。
GPU再核对source text/CED资产和完整state_dict库存/shape/dtype/finite、tied aliases、冻结CED及HRM固定zL_init来源值。
只加载完整音频模型权重，文本源作bootstrap/tokenizer/资产校验；不恢复训练optimizer/RNG/EMA，不重置已训练参数。

audio8/read3显式MizarText7_14x3_7MeshForCausalLM，28物理/56执行，四组write/read原六槽，前三音频read八槽，最后六槽。
只读A4/A8槽6/7、两独立CED4/8 MLP，每题编码一次，显式pair传入每次LM完整forward。
11router/22张量/41544参数；三轮各98共294投影、588矩阵/3419136 LoRA参数，总新增5876760。
真实validate_audio_read3(model)只传model，不传fresh；三轮scope每次finally复位，memory每forward重建。

HRM复用训练AudioPrefixHRMForCausalLM和Decoder.inference_logits；20物理/50执行，P5+(L5x3 H5)x2+C5，固定BP5。
保留MagicNorm/已训练attention gates/固定zL_init和tied head；全量音频checkpoint覆盖构造初值后再冻结eval。
原生PrefixLM383：有效audio/SEP/prompt双向、prefix不能看answer，answer看有效prefix和当前/以前answer。
prompt padding按tokenizer validity屏蔽K/V，query382即使PAD也保留计算；生成历史token按native接口有效。
每生成token传原prompt IDs/validity和全已生成IDs，完整重算50次；每forward从固定zL_init和Prelude结果开始，无跨题/跨token隐藏状态carry。
不套旧attention_mask=None；只读hooks核对全部50次同一正确4D mask、padding/边界/轨迹/无cache及无任务梯度。

共同评测口径保持：FP32/eval/batch1/seed1234/TF32off/无autocast，mono32k/fixed20中心裁剪右补零/10+10再whole CED，
原EOS后缀/PAD!/prompt256/prefix383/首logit382，argmax/无KV/全序列重算/最多300/首EOS停和原raw decode。
Reason原固定seed1234的5000manifest及完整答案exact；MMAU/MMAR各1000原raw官方评分；
MMSU最新紧凑全小写四槽/无only-letter指令、原答案文本包含v2与官方5000分母，GT不进生成。
原预处理、生成适配器、causal循环及评分核心AST对照保持；报告按模型真实记录mask/padding，避免将HRM写成可见PAD。
比较工具核对共同benchmark协议/数据/输入fingerprint/原scorer与共享评分代码，同时独立验证并展示注册的native attention/padding。
支持本批次及注册的历史完整结果；拒绝旧MMSU题面、不同token上限、错模型协议/不完整覆盖/原始生成和评分矛盾。

新结果<run>/eval/ced_eval_20261125_v1/<task>/model--step-34626/；MMSU task=mmsu_compact_lowercase_v1。
提交报告/GPU日志code/RSmol/log/ced_eval_20261125；READY_FOR_GPU/SUBMITTED不是推理PASS，最终看evaluation_report.json。
新旧目录隔离；本批次同模型/数据/代码/协议身份可续跑剩余题，完整raw可重评分。无额外GPU smoke作业。
GitHub完整同步后远程根执行bash code/RSmol/run_ced_eval_20261125_4090.sh all，或README八条单项命令择一。
32份精确同步清单code/RSmol/ced_eval_20261125/sync_files.txt，包含新包/入口/测试/报告/RM；共享Reason/MMSU/原scripts依赖也须已同步。

最终联合38/38通过、无跳过：6真实CPU用例覆盖两架构生产576维实际审计、原native read3/LoRA与50次PrefixLM，
非初始参数重载、首/第二答案训练-推理logits/argmax、未来答案隔离、padding/query382、跨题无carry，
真实训练完整导入图/合成WAV解码重采样/四种实际生成适配器每题一次encoder与正确context。
两组各Reason5000/MMAU1000/MMAR1000/MMSU5000完整尺寸合成流程、续跑/原评分、真实Git/Bash参数链/mock vc，
严格合同/加载控制流（含错误固定zL_init拒绝）、比较策略、7项MMSU紧凑题面回归通过。
全尺寸流程模型/音频/Parquet/CUDA及bulk fsync明确fixture；生产CED encoder在实际生成适配器测试中为显式夹具。
本地Python3.13.5/Torch2.9.0+cpu/Transformers4.54.1/Torchaudio2.9.0+cpu/SoundFile0.14.0；不冒充远程Torch2.11/CUDA生产PASS。
Python3.10、两入口/README6块Bash、whitespace和Git可见性通过；现有本地依赖复用，不修改共享环境。
首轮测试库存数量误用全仓预期已修正；工作期间另一实验提交42f7fdf更新audio_cutoff1000的identity/submit两文件，
本轮保留并精确登记audit_reports/ced_eval_20261125_concurrent_sources.json；3130份任务开始已有生产源码中其余3128保持。
旧失败快照/各轮验收结果保留，完整复验通过。报告audit_reports/ced_eval_20261125_validation.json。
未代为commit/push、读取远程生产资产或提交真实GPU作业。本条仅前置，以下RM既有全部字节保留。

## 2026-11-25：1000题音频cutoff增加八卡5090入口

新增code/RSmol/run_mizar_audio_cutoff1000_8gpu_5090.sh，单次vc提交pdgpu-5090/8GPU/64CPU/256G/1node。
默认输出outputs/RSmol/mizar_audio_sep_cutoff03_28_1000_bf16_v1_8gpu5090；原3090/4090入口保留。
提交层单独解析--partition并显式转交最终output-dir，评测/BF16/缓存/27组条件不改。
manifest生成依赖14份源码SHA保持，无需重建已有1000题manifest。
补充6项提交合同及三入口真实Bash/资源解析检查通过；报告audit_reports/mizar_audio_cutoff1000_5090_validation.json。
Windows沙箱临时目录清理限制获准在沙箱外复验，未提交真实GPU作业。

## 2026-11-25：完整 Mizar 四类各250题音频访问截断，范围扩为第3～28层

用户授权扩大原100题实验并扩展截止层范围。复制原音频cutoff，新增独立
code/RSmol/mizar_audio_cutoff1000/及run_mizar_audio_cutoff1000_8gpu_3090.sh、run_mizar_audio_cutoff1000_4090.sh。
原100题包、manifest、结果及剪层搜索源码保持；41份原实验文件normalized-LF SHA逐一核对。

CPU manifest固定seed1234，MMAU test-mini/MMSU/MMAR/ReasonAQA各250、合计1000。
沿原规范化/唯一选项/唯一GT/token资格，MMSU删除null选项后连续重标并保留原slot；
ReasonAQA仅两类test单录音MCQ。各数据集独立无放回、保存来源顺序，不强制包含或排除旧100。
选中音频缺失/解码失败硬失败，不补抽；SoundFile和原first20/32k/mono/右补零保持。
新manifest独立schema/源码和文件SHA，支持可选旧manifest交集审计，不修改历史manifest校验。
默认data/mizar_audio_cutoff_balanced1000_seed1234_v1/manifest.json；可独立CPU生成，作业内也可自动ensure。

完整原30层final checkpoint（encoder+已训练projector），baseline首先，再cutoff03..28共27组/27000题目决策。
从进入1-based第L层前起，L..30层文本query>=127不能访问audio126+SEP1的key0..126。
保留audio/SEP自己合法causal行、文本早期已吸收信息、全部层执行、原position/RoPE及序列长度。
源权重FP32、前向BF16 autocast、音频前处理/log_softmax/正文累加FP32、TF32off。
原compact lowercase/no question EOT/prompt256/PAD17可见/prefix383/无KV及正文sum likelihood保持；无自由生成。
rank0预计算1000份BF16 prefix，各题encoder/projector一次并跨27组复用。
8卡共同跑同一条件，每卡global_index%8的125题，完整合并/落盘/打印/barrier后切换；单卡每条件1000。
总/四子集acc、baseline差值、paired变化、完整候选分数、并列排名、CSV/JSON/PNG/SVG及逐题恢复/身份检查保持。

8卡pdgpu-3090/64CPU/256G/1node；单卡pdgpu-4090/8CPU/32G/1node；原镜像与mellow_c8204d8。
默认outputs/RSmol/mizar_audio_sep_cutoff03_28_1000_bf16_v1_{8gpu3090,1gpu4090}，
日志code/RSmol/log/mizar-audio-cutoff1000.<tag>.JOB.log。CPU生成及提交命令见新README。
本地26/26通过、无跳过：真实30层小Llama eager/SDPA/BF16 prefix383/全部27条件/3与28边界隔离、
sum likelihood右移/参数保持，真实BF16缓存序列化和无需重编码恢复；抽样/manifest builder/1与8卡调度/27000完整汇总为显式合成夹具。
Python3.10语法及真实Bash+模拟conda/资源解析通过；初次缺少远程MMSU资产的测试夹具已修正，历史保留在报告。
报告audit_reports/mizar_audio_cutoff1000_validation.json；精确同步清单在包内sync_files.txt。
未读取远程生产资产、提交GPU作业或commit/push；远程真实数据/CUDA/3090八卡仍待执行。
本条仅前置，以下原RM记录完整保留。

## 2026-11-24：HRM H2L3固定BP5、CED-small PrefixLM隔离音频正式训练

用户明确从HRM文本最终raw权重开始音频训练，并最终指定LM max/min6e-4/3e-5、mapper8e-4/6e-5。
完整复制隔离参考mizar_s1_gbs128_text71427_mesh_ced_small，新增
code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small/
及run_mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small_3090.sh；既有音频/HRM文本/评测代码不改。

默认来源/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_reinit3600_bp5/formal_20261122_200554_2644345/checkpoint-step-009244。
CPU核对原reinit3600 BP5 seed1完整9244身份；GPU校验根部raw文件SHA与全部FP32 tensor库存/形状/finite/值及tied绑定。
不读EMA权重、不反序列化文本training_state，不继承文本优化器/RNG/游标；不重新转换/折叠Norm/清gate/补偿o_proj。
继承P5/L5/H5/C5、20物理层/50执行、H2L3、MagicNorm、已训练gate与固定zL_init；无MeSH/LoRA。
音频CED-small严格离线冻结eval，原freq_merge_126 mapper随机训练、外置Identity；whole fixed20 mel/BN及正确时频重排保持。
删除母本默认旧音频step34626初始化入口；音频AdamW/scheduler/Scaler/GradNorm/RNG/数据游标从step0新建。

全程BP5：H1/L4/L5/L6/H2记录梯度，Prelude/Coda正常求导，E不detach；保留loss→H1→E/Prelude→mapper路径。
首micro及首完整GA4累积unscale后、clip/optimizer/WD前审计前五物理层逐层任务梯度，存在/finite且每层聚合L2>0。
边界/prelude/entry/H1/H2/L4..L6及mapper梯度、冻结CED无梯度检查保持；立即释放retain_grad引用，每更新继续全参数finite/缺失检查。

保留audio126+SEP1+prompt256+answer300=683，prefix383；所有50次层执行共用同一显式PrefixLM mask。
audio/SEP/prompt之间双向，prefix完全不能读任何answer；answer读全部有效prefix及当前/以前answer。
prompt/answer PAD按tokenizer mask屏蔽K/V，query382即使为prompt PAD也保留计算，继续预测首答案，不把整行mask/清零。
原S1 prompt/answer EOS、256/300截断/右pad、PAD!且不扩词表、模板/小写/CoT规则保持。
外部answer-only shifted CE仍logits382..681对齐300答案，rank-local有效token mean→GA4→DDP平均。
正式第一个source探针核对完整683与prefix383首答案logits/argmax，以及改变answer后prefix输出逐值不变；非独立smoke。
Decoder提供PrefixLM一致的无cache inference_logits；旧通用generate/benchmark本次未适配。

8GPU×micro4×GA4=GBS128，原hybrid1477492行，3epochs/每epoch11542/总34626更新，warmup1731。
LM max/min6e-4/3e-5，mapper8e-4/6e-5；预热保留startup最低.1×max，之后分别cosine到.05/.075下限。
scheduler在optimizer之后更新，cosine分母total-1-warmup，最后一次真实更新即使用指定最小LR。
AdamW beta(.9,.999)/eps1e-8/WD.01及bias/ndim≤1不衰减，原FP32/GradScaler/GradNorm/TF32off保持；不沿用文本Adam-atan2/EMA/BF16。
pdgpu-3090/64CPU/256G/8GPU/1node，原镜像/mellow_c8204d8；resolved config与trainer核对Transformers4.54.1，不自动更改共享环境。
保留CPU index、共享音频立即训练、4workers后台复制及optimizer边界8rank MIN共同切本地；先stop/reap再owner/token清理。
tmpfs为/dev/shm/mizar_s1_gbs256_hrm_prefixlm_<token>，其中gbs256仅保留原安全守卫前缀。

输出/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_hrm_bp5_prefixlm_ced_small_3090/formal_<timestamp>/formal/。
保存5771/11542/17313/23084/28855/34626和epoch3；完整模型/具名optimizer库存及moments、scheduler/Scaler/GradNorm/8rank RNG/下一epoch-update游标。
同新路线/源码/文本与CED资产/数据/依赖/PrefixLM/BP5/LR合同在同外层run-root完整恢复；拒绝错路线、错库存/顺序/dtype/非有限、错LR/步数、冻结CED或固定初值变化。
写checkpoint或source探针/首micro审计失败集体报错；metrics异常也关闭。直接formal，无额外GPU smoke或人工PASS前置。

本地最终22/22通过、无跳过：10路线/脚本/staging、11真实CPU张量/音频、1真实双进程Gloo/DDP。
真实HRM/SDPA/mapper/autograd验证50次mask、双向影响/答案未来泄漏隔离、padding/query382、首答案训练-推理对照、逐层Prelude/mapper梯度；
源raw各tensor/非零gate继承且旧optimizer不反序列化，缺失/多余/dtype/nonfinite/tied冲突拒绝；两角色全部34627个LR位置对照。
完整checkpoint/RNG/GradNorm/AdamW恢复后下一更新逐值一致；实际训练函数128行合成夹具三epoch×一步、每步4micro，之后从step1走实际resume到step3，最终参数逐值一致。
CUDA、八rank及生产CED encoder在上述完整流程中明确fixture；另真实两进程Gloo使用实际HRM/Decoder和原mapper/loss精确AST，梯度差约2.98e-8。
合成WAV真实解码/fixed20/whole mel/时频排列通过，真实Bash+模拟vc/torchrun检查3090资源/含空格参数/失败清理，复制完成前继续训练的原staging线程测试通过。
音频CPU环境Python3.13.5/Torch2.9.0+cpu/Transformers4.54.1/torchaudio2.9.0+cpu/soundfile.14；Torch2.9 Windows Gloo初始化unsupported，
双进程另用已有Torch2.15.0+cpu，版本和限制单列。torchaudio/soundfile仅装到忽略的.hrm_test_runtime/audio_packages，不改共享环境。
早期Bash夹具cwd、CE舍入断言和模拟encoder闭包deepcopy已修正，失败历史在报告保留；未修改旧测试或历史快照。
3701份本任务开始tracked Python/Shell/JSON normalized-LF SHA保持；14份数据/索引/staging/CED源码逐字节继承，Mellow两类及answer_loss/GradNorm AST保持，HRM recurrence AST保持。
Python3.10、两Shell、README四块Bash、whitespace和精确Git同步清单通过；报告audit_reports/audio_hrm_prefixlm_validation.json及包内sync_files.txt。
未访问远程生产资产、提交真实GPU作业或commit/push；生产来源/CED加载、Torch2.11/CUDA、3090显存与八卡训练仍待远程验证。
GitHub同步后远程仓库根运行bash code/RSmol/run_mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small_3090.sh；精确提交和完整恢复命令见新README。
本条仅前置，以下原RM全部字节及用户已有记录完整保留。

## 2026-11-24：三循环71437 CED4/8 write-value融合与三轮独立LoRA r8隔离训练

用户按确认计划授权复制隔离后实现。新增
code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71437_mesh_ced48_wfuse_lora_r8/
及run_mizar_s1_gbs128_text71437_mesh_ced48_wfuse_lora_r8_3090.sh。
完整复制三循环CED-small母本，迁入二循环wfuse的原两处value融合、具名优化器库存及完整恢复。
既有三循环/二循环wfuse训练、共享文本模型和所有评测实现保持；并行audio8/read3改动保留。

初始化固定为/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x3_7_mesh/
formal_reinit1200_seed1_20261119_084545_3432609/checkpoint-009244。
严格核对9244完整标记、全部schedule别名/来源映射、FP32 safetensors及16个已训练router张量，
注册新模块前后核对全部继承值/Parameter对象/tied embedding-head；不读文本training_state.pt。
音频optimizer/scheduler/GradScaler/GradNorm/RNG/游标从step0新建，不继承文本seed1或BF16配方。

28物理层/56执行，0..6、(7..20)三次、21..27；六槽全683位置memory每forward新建，
四组write/read、八独立router/16张量/27696参数保持；四边界7/21/35/49原读写完整执行。
仅边界7将prefix output前126位置与A4融合，边界21将第一轮output前126位置与A8融合。
两个独立无bias Linear1152→576，输入[边界hidden;CED MLP输出]、初始[I576,0]、整个矩阵训练。
只替换write value；prefix query仍prefix output，循环query仍该轮input，SEP/text value保持。
边界35/49原output直接写入，第三轮无新增融合，但仍可使用先前进入memory的音频信息。
无额外seed槽、listen router、gate或hidden残差；CED单次冻结eval提取final和完整block4/8 norm前输出，
原freq_merge_126 mapper与两个独立1536-576-576 MLP保持，正确时频排列及MLP在no_grad外。

物理7..20七投影三轮独立LoRA r8/alpha16/scale2/dropout0，294对/588矩阵/3419136参数；
基础W共享并全量训练，前后七层无LoRA；逐调用finally复位、无KV、拒绝decoder gradient checkpointing。
加两MLP2443776与融合1327104，相对普通三循环CED新增7190016。
首次三轮边界反传和全部16router梯度审计后立即释放引用；融合Encoder半初值0时MLP首步任务梯度0、
LoRA B零时A首步任务梯度0均按预期检查；新增10角色32次真实更新内全rank累计解锁并保存恢复。

原hybrid1477492行/fixed20/原S1 EOS/PAD/683布局、answer-only shifted CE、
rank-local token mean→GA4→DDP、FP32/GradScaler/AdamW/LambdaLR/GradNorm/TF32关闭保持。
pdgpu-3090/64CPU/256G/8GPU/1node，micro4×GA4=GBS128，3epochs/34626更新/warmup1731。
LM/原router maxLR3e-4；mapper/MLP/fusion/三轮LoRA6e-4。CPU index/共享音频立即训练、
4workers后台复制、optimizer边界8rank MIN集体切换及先reap再owner/token清理保持；直接formal。
默认输出outputs/RSmol/mizar_s1_ced48_wfuse_lora_r8_gbs128_text71437_mesh_3090/formal_<timestamp>/formal/。
保存5771/11542/17313/23084/28855/34626及epoch3；完整模型/AdamW具名库存和moments、scheduler/
Scaler/GradNorm/8rank RNG/下一数据游标/解锁状态；只接受本新路线同源码/资产/数据/依赖/配方完整恢复。

最终联合139项，111执行通过、28缺torchaudio/soundfile音频依赖明确跳过；新路线57通过/11跳过。
复用既有本地依赖，6项真实CPU核心覆盖严格继承/初始logits和梯度、56调用/683位置/三轮反传/16router、
588矩阵独立与第三轮选择、非零两次融合且后两次value原样、完整模型/AdamW moments/RNG恢复后下一更新逐值一致。
CPU collective/CUDA RNG为显式fixture，NumPy及Bash/mock vc覆盖四次读写、优化器/恢复拒绝/启动清理；
15份data/staging与三循环母本字节一致，CED/mapper/接口与wfuse母本字节一致，核心函数及完整epoch更新AST对照通过。
Python3.10、两Shell、README两块Bash、48份Git可见文件和新增whitespace通过。
本任务开始3654份tracked Python/Shell/JSON中3652保持，两份并行audio8/read3修改单列concurrent_source_changes.json。
初次两母本历史摘要守卫已在任务前过时，原失败报告保留；新validator只以本任务库存替代这两条旧断言，旧测试/快照不改。
后续测试入口的stdlib code遮蔽和runpy测试路径问题已修正，previous_runs保留失败过程。
本地Python3.13.5/Torch2.9.0+cpu/Transformers4.54.1，与远程Torch2.11/CUDA不同；不冒充3090/八卡PASS。
报告audit_reports/mizar_s1_text71437_ced48_wfuse_lora_r8_validation.json；Windows临时目录/Bash限制获准沙箱外离线验证。
GitHub同步后远程仓库根执行bash code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced48_wfuse_lora_r8_3090.sh。
精确Git清单、提交/完整恢复、输出及验证说明见新包README和sync_files.txt；既有评测器需后续独立适配。
未代为commit/push、读取远程生产资产或提交真实GPU作业。本条仅前置，以下全部RM字节完整保留。

## 2026-11-23：7–14×3–7 CED4/8 音频八槽read3及三套LoRA r8隔离正式训练

按用户确认计划复制三循环CED-small母本，新增独立
code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71437_mesh_ced48_audio8_read3_lora_r8/
及run_mizar_s1_gbs128_text71437_mesh_ced48_audio8_read3_lora_r8_3090.sh。
迁入双循环audio7_read2的CED特征/只读seed/LoRA机制；用户明确三轮前均读A4/A8。
原双循环、普通三循环、共享文本模型与全部既有评测源码保持，3609份本任务开始时tracked Python/Shell/JSON normalized-LF SHA保护。

默认严格继承/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x3_7_mesh/formal_reinit1200_seed1_20261119_084545_3432609/checkpoint-009244。
不重新转换，校验完整9244步/完成标记/全部schedule别名、FP32/finite/全部safetensors及16个已训练router张量。
先strict加载，再注册新read/LoRA，复核原值、Parameter对象及embedding/head绑定；不读文本training_state.pt。
音频optimizer/scheduler/Scaler/GradNorm/RNG/数据游标从step0新建，文本seed1/来源1200不成为音频训练游标。

28物理层/56执行，0..6,(7..20)三次,21..27；suffix逻辑起点49。
每forward在prefix前重建audio8 [E,0,0,0,0,0,A4,A8]及SEP/text6 [E,0,0,0,0,0]。
四个原write均只更新前六槽；三个新audio read在累计7/21/35层后读取八槽，49层后最终audio read仍为原六槽read3。
A4/A8槽6/7全程只读，不加gate/listen/residual或重复注入。prefix query/value为prefix输出，循环query为本轮输入、write value为输出。
三新read分别复制对应原read0/1/2：前六行W保持、bias减log2；新两行W/b均取对应原第0行，独立存储/CPU RNG保护。
原六槽内部相对概率保持、新两槽初始等权，组概率随query变化，不固定为1/2、1/4、1/4。

CED-small严格离线/冻结eval，每批一次whole20编码捕获完整block4/8 final norm前输出；正确时频排列、原freq_merge_126主mapper、
两套1536-576-576 MLP及其no_grad外梯度保持。主mapper随机训练；不是继承旧双循环音频权重。
物理7..20七投影挂三套独立r8/alpha16/scale2/dropout0 LoRA，A Kaiming/B零，基础W共享且全量训练。
588矩阵/3419136 LoRA参数，三个read13848、两MLP2443776，总新增5876760（主mapper另计）。
全部routers 11模块/22张量/41544参数；三轮各98共294次投影，逐调用选轮次且finally复位，无KV/decoder checkpointing。

原hybrid1477492行、fixed20中心裁剪/右补零、683布局、EOS/PAD及answer-only shifted CE和rank-local token mean→GA4→DDP保持。
pdgpu-3090/64CPU/256G/8GPU/1node，micro4×GA4=GBS128，3epochs/34626更新/warmup1731。
基础LM/原routers maxLR3e-4，mapper/两MLP/三新read/三套LoRA6e-4；原AdamW/GradNorm/LambdaLR/GradScaler、FP32/无autocast/TF32off保持。
15份数据/索引/staging运行文件逐字节继承，完整epoch更新循环与read2母本AST保持，仅三轮错误提示修正。
保留CPU index、共享立即训练/默认4worker后台复制、optimizer边界八rank MIN共同切换；tmpfs保留mizar_s1_gbs256_守卫前缀及独立token，
先stop/reap producer后owner/token清理。直接formal，无额外GPU smoke前置。

独立输出/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced48_audio8_read3_lora_r8_gbs128_text71437_3090/formal_<timestamp>/formal/。
每5771及最终保存5771/11542/17313/23084/28855/34626和epoch3。
完整resume保存/校验模型全部分支、具名optimizer库存与顺序/moments、scheduler/Scaler/GradNorm、8rank RNG和数据/梯度审计游标；
同新路线/源码/资产/配方在原外层run-root恢复到总34626，恢复不重置已训练read或LoRA，其他路线及已完成run拒绝。

最终联合96项：78执行通过、18真实张量缺Torch/Transformers/Torchaudio/Soundfile/Safetensors明确跳过（新路线9、原回归9）。
包含实际生产forward/LoRA/stack的NumPy独立递推、三轮非零分支/有限差分/294调用及scope清理、只读seed/六槽最后read、
16router分片库存/缺失重复shape/dtype/nonfinite/tied拒绝、真实optimizer分组588矩阵/13角色/34627个LR位置、
实际checkpoint writer/8rank字段、完整恢复分支第三LoRA/read值/各rank RNG及拒绝条件、真实Bash+模拟vc/torchrun、CPU index/复制/清理路径守卫。
首轮77执行通过后补充恢复回归，历史结果保存在报告previous_runs；早期fixture.events遗漏已修正，Windows临时目录受限后获准沙箱外离线复验。
Python3.10/Bash及README命令语法通过，Git同步清单核对无忽略遗漏。报告audit_reports/mizar_s1_text71437_ced48_audio8_read3_validation.json。
真实Torch/autograd、生产资产加载、3090显存与八卡训练未执行，不声明GPU PASS；未代为commit/push或远程提交。

经GitHub同步后，远程仓库根执行bash code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced48_audio8_read3_lora_r8_3090.sh。
完整提交、显式输出、同run恢复和精确Git清单见新包README；既有评测器尚未登记新合同。
本条仅前置，以下原RM全部字节完整保留。

## 2026-11-23：隔离实现 MyEval MMAR8 三模型全排列评测及 A4/A8 零 seed 干预

用户授权按既定计划复制隔离后实现。新增 code/RSmol/myeval_mmar8/、run_myeval_mmar8_4090.sh，
复制既有 ced_eval_20261122 的严格历史源码/模型加载及生成流程；既有正式评测、训练、模型文件均保持。
本轮开始时 3529 份 tracked Python/Shell/JSON 的 normalized-LF SHA 全部保持，库存见本包 source_guard.json。

固定 base 为 mizar_s1_ced_small_gbs128_text71427_mesh_3090/formal_20261119_141531507533438，
read2 为 mizar_s1_ced48_audio7_read2_lora_r8_gbs128_text71427_3090/formal_20261120_111448563391250，
均 /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/ 下 formal/checkpoints/model--step-34626.ckpt；
第三模型 read2_zero48 使用同一 read2 checkpoint，完整参数严格覆盖/审计后，仅将投影后的 A4/A8 只读 seed 值置零。
保留最终音频 prefix、七槽及原七路 softmax 分母、全部原五槽读写、两轮 LoRA 的392矩阵和28物理/42执行；
不删槽、不重新归一化、不改权重、不训练。每题显式独立 context，每token forward 核查本题零context身份，拒绝跨题陈旧值。

标注本地 C:\Xlance\assets\MMAU analysis\MyEval\mmar8selected.jsonl，远程 /hpc_stor03/sjtu_home/jinwei.zhang/data/MyEval/mmar8selected.jsonl。
normalized-LF SHA a670c97eef24859c6dc015ba137e723fa2e7205d198a0b4d234ced0e642b7815；8音频/67自定义题，
53选择题（P29/S11/W7/R6）和14开放题（含caption），只用 questions 的题目/选项，原MMAR问题和旧模型回答不进自定义推理。
按选项索引全排列并重新标a..e；2选需2/2，3选需6/6，4选至少21/24，5选按用户明确要求至少115/120。
734次排列+14开放=每模型748次生成；按原题通过与否统计四维和总ACC（分母53），不是按734次排列加权。
开放题只生成不评分，保存逐题raw及三模型并排MD/JSONL。保留原MMAR来源分类，不自动等同于新题能力维度。

四个独立作业为三模型各MyEval8，及read2_zero48额外完整1000题MMAR；共3244次生成。
完整MMAR使用原问题/原固定选项及官方完整评分，不对其1000题另做全排列。
每排列仍按约定将原始生成直接交官方MMAR string_match，不增加字母parser、EOS清洗或语义打分。
官方词集合匹配忽略词序和角色；已查出6道自定义题存在错误选项与正确答案词集合相同，
README/验收报告及scores.json逐题标明；全排列不能消除此评分局限，后续严格parser须另立版本。
另5段音频超过20秒，保持原中心裁剪/右补零后10+10协议，dataset_audit记录标注推算裁剪区间，可能丢失首尾证据。

FP32/eval/seed1234/TF32off、mono32k/fixed20、紧凑整体小写题面、原EOS后缀/PAD!、prompt256/prefix383/首logit382，
无mask/无KV/每token全序列重算/argmax最多300及首EOS停止保持。实际prompt token和解码音频指纹逐题记录，比较时一致性核对。
两checkpoint均绑定历史训练Git 8a9f1f76722e77cba4be429a578c63fdcf6ee527；不能用已演进的当前同名base训练目录。
只git archive指定训练后端和共享源码到新 log/myeval_mmar8_sources，逐文件/训练code_digest校验，旧正式缓存不变。
提交前八音频标注/路径CPU预检、两checkpoint各一次mmap合同/源码预检，失败目标不提交、其他继续。
全部pdgpu-4090、1GPU/8CPU/32G/1node、原镜像和mellow_c8204d8，无额外GPU smoke。

默认输出 <run>/eval/myeval_mmar8_v1/<alias>/<myeval8|mmar>/model--step-34626/；四目录隔离。
源码/权重/runtime/标注/任务/音频文件/scorer/门槛/干预身份锁定，追加raw/fsync/断点续跑，完整坏行拒绝，未完成不出完整分数。
compare/rescore仅CPU读取完整保存产物、校验raw/输入/scorer/代码并重算；源评测目录只读，不改其进度/分数/报告。
GitHub同步后远程仓库根 bash code/RSmol/run_myeval_mmar8_4090.sh all 提交四项；单项/预览/预检/续跑/比较和精确同步清单见本包README。

本地联合33项最终32通过，1真实微型CPU模型测试缺Torch/Transformers/Safetensors明确跳过；
覆盖3x748及额外1000完整尺寸合成流程、四门槛、同源题分母、数据隔离、真实Git/Bash、原核心AST、断点恢复、
真实未改MMAR官方scorer CLI及既有loader/hooks的5项回归。NumPy执行实际生产七槽混合/读写方法核对零seed与7路分母。
官方CLI测试仅补齐未调用的tqdm显示导入依赖，评分实现不替换；生成/解码/CUDA及bulk fsync在批量合成测试明确模拟。
初次发现重复排列ID检查缺口已补齐；随后评分测试缺tqdm及Windows环境变量测试夹具恢复问题已修正，最终复验通过，运行报告保留历史失败。
报告 audit_reports/myeval_mmar8_validation.json；未读取远程生产权重/音频、提交真实vc作业或commit/push，不声称生产4090 PASS。
本条仅前置，以下原RM的525110字节完整保留，原SHA256为1d1fcb12826d97ed479e48e682ec577dacad70c009c89c184164aa81980e7110。

## 2026-11-23：修复audio7_read2校验接口，新增三循环71437及两checkpoint八项4090评测

用户授权按计划实现。原read2报错由评测器向历史validate_audio_read2(model)误传fresh=False导致，
本地从固定Git真实函数复现相同TypeError；普通read2校验本就只检查训练后结构/存储，不需要fresh。
修正text71427_ced_eval/runtime.py及ced_eval_20261122/runtime.py各新增5行明确read2分支，只传model；
ar6/wfuse支持fresh的原调用保持，训练源码/权重加载/数据/生成/评分不改，不捕获异常跳过审计。

新增独立code/RSmol/ced_eval_20261123/及run_ced_eval_20261123_4090.sh。
固定targets.json为用户指定71437 formal_20261121_150120579947686及read2 formal_20261120_111448563391250，
均formal/checkpoints/model--step-34626.ckpt，各ReasonAQA5000/MMAU test-mini/MMAR/MMSU，共8项；旧19项不自动重投。
别名mesh_71437、audio7_read2。全部pdgpu-4090、1GPU/8CPU/32G/1node，原镜像/mellow_c8204d8。

三循环显式MizarText7_14x3_7MeshForCausalLM，28物理层/56执行，0..6,7..20三次,21..27；
六槽memory，四组write/read共8独立Linear(576,6)，16张量/27696参数；路由在执行7/21/35/49层后。
read2保留42执行、392矩阵两轮LoRA、CED4/8两个独立MLP/两七槽audio read，完整新增参数4731278。
严格源码/架构/来源/资产/完整state_dict、FP32/finite/绑定及训练后值审计；不恢复或重新初始化optimizer/RNG。
只读forward hooks核对56次轨迹及4组router边界；read2复用原42次/LoRA hooks，memory每完整forward重建。

三循环训练后端固定75dcda3a7bcd8765044b14ecd4e408fe33dfdda6，code_digest=ea3677917df06121dfb0cbbc6254f2cd5b707ea4787bfd3eb9e5f4daa1fece9d；
read2固定8a9f1f76722e77cba4be429a578c63fdcf6ee527，原f0ab5b...code_digest保持。
仅git archive各训练目录/对应x2或x3共享模型/合同/release到新共享log/ced_eval_20261123_sources，逐文件SHA/库存固定。
动态隔离x2/x3共享包并核对实际导入路径；不扩写旧按revision缓存清单。当前评测器及紧凑MMSU保持当前checkout。
CPU预检每checkpoint一次，核对合同/源码及真实函数签名再申请GPU；失败目标不提交，其他继续；GPU再严格模型审计。

推理/评分与上一版核心函数AST对齐：FP32/eval/seed1234/TF32off、mono32k/fixed20、prefix383/首logit382、
PAD!/EOS0/原EOS后缀分词顺序、无mask/无KV/全序列重算/argmax最多300及首EOS停止。
Reason原seed1234固定5000及完整答案exact；MMAU/MMAR各1000原raw官方评分；
MMSU最新紧凑全小写四槽，原答案文本包含v2/官方5000分母不改；GT不进生成输入。
完整覆盖、身份/进程锁、raw坏行拒绝与断点续跑保持。

新输出<run>/eval/ced_eval_20261123_v1/<task>/model--step-34626/，MMSU task=mmsu_compact_lowercase_v1。
旧失败/半程输出保留，read2本次从新目录重跑；本批次之后中断同身份可续跑。
日志/提交报告code/RSmol/log/ced_eval_20261123，最终状态看各evaluation_report.json；READY_FOR_GPU/SUBMITTED不代表推理PASS。
GitHub同步后远程根执行bash code/RSmol/run_ced_eval_20261123_4090.sh all提交8项，或README八条单项命令择一。
无额外GPU smoke作业；未代为commit/push、远程读取生产权重/数据或提交真实作业。

首轮新包30项全通过。最终联合39项37通过2真实CPU张量因缺Torch/Transformers/Safetensors跳过；
包含实际audit_model及真实固定源码router/LoRA/parameter_audit函数（明确元数据张量/模块夹具，非模拟审计返回值），
原fresh错误/修复、两个旧入口、错层/形状/共享存储/LoRA未复位/dtype/来源拒绝，56次hooks和架构源码合同，
真实Git/Bash参数链、mock vc、两组各Reason5000/MMAU1000/MMAR1000/MMSU5000完整尺寸合成流程及续跑、原官方scorer。
联合初次一项历史整文件SHA守卫因本轮授权read2修复失败；保留历史报告，compatibility_reference登记精确前后SHA，
并确认其他函数AST不变；修正对应测试后7项MMSU回归及完整39项复验通过，初次失败记录保留。
2927份其他已有生产源码normalized-LF SHA保持，仅两旧runtime本轮授权修复；Python3.10/两入口/README5块Bash/whitespace通过。
报告audit_reports/ced_eval_20261123_validation.json；音频/Parquet/CUDA/生成及bulk fsync在全量合成流程明确模拟，不冒充生产4090 PASS。
本条仅前置，以下RM既有全部字节完整保留。

## 2026-11-22：HRM raw step3600重新完整9244步，seed1/BP5/升温后cosine隔离正式训练

按用户确认计划新增code/RSmol/text_hrm_reinit3600_bp5/，复用原text_hrm_h2l3模型/数据/官方Adam-atan2/真实DDP累积。
默认来源为用户指定/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_h2l3/formal_20261122_000344_4005340/checkpoint-step-003600。
只用根部raw权重/tokenizer，继承已训练gate、全部20层、Norm、tied embedding/head和固定zL_init。
不重新转换/折叠gamma/清gate/乘o_proj；优化器动量与计数从0建立，EMA.9999从加载的raw开始，RNG/数据游标重建。
来源3600仅作初始化血缘，本轮从1完整训练到9244；本轮中断后才用本新路线完整resume并继续总计9244。

P5/L5/H5/C5、H2L3、20物理/50执行、MagicNorm和causal padding保持；全程BP5：H1/L4/L5/L6/H2求导。
Prelude五层全部trainable/进入optimizer，E不detach，保留H1到E/Prelude的任务梯度路径。
首micro和首64micro完整累积后、optimizer/weight decay前，分别审计边界和每个Prelude物理层；grad存在/finite、每层聚合L2>0。
任何rank审计失败集体报错；每更新继续全参数缺失/非有限检查，不以weight decay造成的参数变化冒充任务梯度。

8GPU×micro2×GA64=GBS1024；Adam-atan2 beta(.9,.95)、全部唯一可训练参数WD.1、EMA.9999。
第1步3e-4，第462步6e-4，第9244步6e-5；前5%=round(9244*.05)=462，线性含首尾，之后cosine。
精确公式：s<=462为3e-4+3e-4*(s-1)/461；其后6e-5+2.7e-4*(1+cos(pi*(s-462)/8782))，optimizer更新前设置。
FP32主参数/状态/残差、BF16前向、FP32 Norm/CE、TF32off；不裁剪/无GradScaler/GradNorm/异常跳步。
原85片Parquet全局shuffle与rank分配用data seed1，rank内buffer4096用1+1009*rank；max1024/动态右pad/不packing。
全8rank/64micro统一有效next-token平均；固定L初值继承不随data seed改变；9465856名义样本，不承诺每篇文档恰好一次。

新入口run_text_hrm_reinit3600_bp5_3090.sh，CPU准备scripts/prepare_text_hrm_reinit3600_bp5.sh，GPU运行scripts/train_text_hrm_reinit3600_bp5_ddp.sh。
正式wrapper先校验原完整checkpoint SHA，在CPU仅复制raw模型/tokenizer到独立外部快照，成功后才vc submit；不反序列化旧training_state。
默认快照models/SmolLM2-HRM-H2L3-reinit3600-raw，记录完整来源marker/binding；同来源已验证快照可重复用，源随后清理不影响训练。
不存在源且无快照则拒绝，不换别的step；失败不提交GPU，无新增GPU smoke或全量CPU数据内容扫描前置。
pdgpu-3090/64CPU/256G/8GPU/1node、原镜像/rsmol；环境仅RSMOL_HRM3600_*，不继承旧resume变量。
默认输出/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_reinit3600_bp5/formal_<timestamp>_<pid>/。
每400及9244保存raw/独立ema/、完整optimizer/8rank RNG/数据游标/合同，最近3份正常为8800/9200/9244。
本轮完整resume必须新路线相同代码/依赖/配方/初始化/数据seed，输出到新目录；不恢复旧step3600训练态，不重置本轮已存状态。

本地联合42/42通过、无跳过：新路线13项及原HRM29项，含真实CPU raw3600合成checkpoint及非零gate继承、新optimizer/EMA状态，
全部9244个LR/BP位置、seed1真实Parquet顺序/重放、462/463边界完整恢复后下一更新逐值相同、保留/损坏/错路线/错LR/参数次序拒绝。
真实两进程Gloo/DDP全局梯度对照与逐层Prelude审计通过；实际正式启动+首64micro更新CPU控制流通过，CUDA/八rank通信明确模拟。
真实Bash→模拟prepare/vc/torchrun验证3090资源/含空格参数、准备失败不提交、resume跳过原来源准备。
原历史全仓guard早于后续已授权MMSU修改，报告明确用本任务开始时3280份tracked Python/Shell摘要替代该历史库存断言；全部保持。
首轮两项异常类型断言不符已修正；补查修正startup审计顺序并加入实际启动回归。原HRM及其他既有源码/测试不改。
Python3.10语法、三Bash与README四块命令语法、全部新增文件whitespace/Git可见性通过；Windows受限测试获准沙箱外离线运行。
环境Python3.13.5/Torch2.15.0+cpu/Transformers4.54.1，与远程Torch2.11/CUDA不同；不冒充3090/真实八卡/生产checkpoint PASS。
报告audit_reports/text_hrm_reinit3600_bp5_validation.json，精确Git同步与初始化/正式提交/完整resume命令见新包README。
未读取远程生产资产、提交作业或commit/push；以下RM既有全部字节完整保留。

## 2026-11-22：七个指定checkpoint的19项统一4090评测适配完成

用户授权按计划实现，新增独立code/RSmol/ced_eval_20261122/及run_ced_eval_20261122_4090.sh。
固定targets.json完整绑定七个用户路径，均step34626；MMSU7项，ReasonAQA5000/MMAU test-mini/MMAR各4项，共19项。
mesh_original/listen/loop_lora仅MMSU；audio7_read2/wfuse/mesh_reinit/smol_gbs128四项。
audio7_read2两组需求去重，其四任务从本批次新目录重跑，旧半程目录保留。
全部pdgpu-4090、单卡8CPU/32G/1node，原镜像与mellow_c8204d8。

所有目标复用当前同一评测数值和评分实现，MMSU统一最新紧凑全小写四槽，无只答字母指令。
FP32/eval/seed1234/TF32off、fixed20/32k、prefix383/首logit382、PAD!/EOS0、无mask/无KV/argmax300保持。
Reason固定5000manifest及完整答案exact；MMAU/MMAR原raw官方评分各1000；MMSU原答案文本包含v2及官方5000分母。
完整分母、坏行拒绝、身份锁/进程锁及断点续跑保持；真实生成/预处理/runner核心函数AST对齐原实现。

前五目标训练源码固定8a9f1f76722e77cba4be429a578c63fdcf6ee527，重训/普通Smol固定75dcda3a7bcd8765044b14ecd4e408fe33dfdda6。
仅git archive所需训练后端到共享log/ced_eval_20261122_sources，逐文件SHA/库存及各训练code_digest固定核对。
旧/重训MeSH同route但不同源码，严格区分；不放宽checkpoint/source合同，不用历史评测器覆盖最新MMSU题面。
每目标提交前CPU mmap合同预检一次，失败目标不申请GPU，其他目标继续；GPU再严格加载完整state并核对资产/架构。
MeSH保留28物理/42执行及各自音频分支/LoRA；普通Smol为标准HF LlamaForCausalLM的30独立层/0..29，无MeSH/LoRA。
重训额外核对原音频仅权重初始化来源与2e-4/2e-5配方；eval不初始化或恢复optimizer/RNG。
并发读取已核对源码快照不争用准备锁；run_identity/model_audit显式记录目标、训练route与源码revision。

新输出<run>/eval/ced_eval_20261122_v1/<task>/model--step-34626/，MMSU task为mmsu_compact_lowercase_v1。
提交报告和GPU日志在code/RSmol/log/ced_eval_20261122；各输出evaluation_report.json才是评测完成依据。
GitHub同步后，远程仓库根运行bash code/RSmol/run_ced_eval_20261122_4090.sh all提交19项；
或七个目标逐条提交，不能与all重复；README同时给出19条单项命令、预览/CPU预检/日志/只读比较方式。

最终新包35项28通过7真实CPU张量缺Torch/Transformers/Safetensors跳过；另7项紧凑MMSU回归全通过，总42项35通过7跳过。
覆盖真实Git快照/训练合同、七架构加载控制流/错route或同route错源码拒绝、真实Bash参数链和mock vc，
四组Reason5000、四组MMAU1000/四组MMAR1000、七组MMSU5000完整尺寸合成流程、原官方scorer和中断续跑。
2898份本轮开始前生产源码normalized-LF SHA保持，Python3.10/两入口/README5块Bash和whitespace通过。
首轮2项夹具错误已修正；长测期间曾修改缓存锁源码，旧身份保护如预期拒绝重评分；固定源码后完整重测通过，记录保留。
报告audit_reports/ced_eval_20261122_validation.json；模型/音频/Parquet/CUDA和bulk fsync合成模拟，未声称生产GPU PASS。
未读取远程权重/数据、提交真实作业、commit/push。本条仅前置，以下RM原字节完整保留。

## 2026-11-22：MMSU正式输入修正为紧凑格式＋全小写，等待用户指定重评范围

用户明确授权修正实现，后续具体待测模型/checkpoint再提供；本轮未挑选目标、未变动历史targets/excluded或提交真实作业。
共享mizar_s1_ced_mmsu_eval/dataset.prompt改为question a) choice_a b) choice_b c) choice_c d) choice_d整体lower。
逐字符对齐已有MMAU build_fixed_order_prompt(...).lower()及真实三题诊断compact_lowercase。
问题/选项外侧strip，保留全部四槽原顺序，null输入占位none；无新增Question/Choices前缀、模板换行或字母指令；GT/分类不入输入。
通用MMSU38个已有variant与text71427六路线共用同模板；原始元数据和raw不改写大小写。
原官方模板保留为dataset.official_prompt，仅诊断工具显式使用；历史Git快照无此函数时仍取该快照旧prompt。

两套MMSU protocol统一升级mmsu_compact_lowercase_ced_fixed20_raw_greedy_answer_text_v3，prompt_policy明确紧凑小写。
评分仍mmsu_answer_text_contains_v2，原解析/官方scorer/完整5000分母保持。
音频、模型类/权重严格加载、28/42或各自架构分支、FP32/无autocast/TF32off、PAD!/EOS0、prefix383/首logit382、无KV/argmax300保持。
真实生成/音频/runner核心函数AST保持；训练目录、共享模型、MMAU/MMAR/ReasonAQA数值实现未改。

通用MMSU默认eval/mmsu_compact_lowercase_v1/stepN；selected5/mesh48快捷入口同步新命名空间，官方release用mmsu_compact_lowercase_v1。
text71427 MMSU默认eval/text71427_ced_v2/mmsu_compact_lowercase_v1/<ckpt-stem>，显式output-root也使用新任务目录名。
其他三个任务默认目录保持。旧模板目录显式传入也在模型加载前被身份拒绝，raw/报告不覆盖；同版新协议继续原续跑。
旧rescore/detailed report及历史rescore_selected5路径保持，旧raw重新评分不能代替新模板重新生成。
剩余27目标调度器按新prompt_policy拒绝旧结果，原历史排除清单不代表新题面已完成，具体重评范围等待用户。

本地联合134独立用例最终125通过、9真实ML/音频/Parquet依赖缺失跳过；第一轮47项针对检查通过。
覆盖题面与MMAU精确对齐/GT不泄漏/null不重排、通用38+六路线新协议与新默认路径、旧输出拒绝且字节保留、
通用5000题和六路线5000题合成流程逐行题面检查、断点续跑、旧raw离线重评分、原官方scorer、真实Bash+模拟vc资源/参数链。
联合初次1项旧源码保护未组合此前selected4/init0904授权注册；明确组合已有注册审计和本次前后摘要后，8项定向复验全部通过。
初次失败和后续通过均保留报告，不重复计数；mmsu_v2_audit的生产源码保护不再把历史测试脚本演进当作模型/benchmark改动。
2886份范围外生产源码normalized-LF SHA保持，12份明确授权生产实现/入口变更；旧快照不覆盖，compact_prompt_reference单列前后摘要。
Python3.10、Bash参数链与git diff --check通过；沙箱临时目录受限后获准沙箱外完成离线测试。
报告audit_reports/mmsu_compact_lowercase_validation.json；本地模型/音频/Parquet及bulk fsync明确模拟，未声称新模板全量生产GPU通过。
细节code/RSmol/mizar_s1_ced_mmsu_eval/COMPACT_PROMPT.md和各README；当前base旧权重仍要求匹配历史源码，未放宽strict source。
未commit/push或远程提交。本条仅前置，以下RM既有字节及用户本轮开始前修改完整保留。

## 2026-11-22：MMSU三题真实回传确认题面触发退化，数值布局对照通过

用户回传71427 base step34626三题对照，远程报告为：
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_text71427_mesh_3090/formal_20261119_141531507533438/eval/mmsu_probe_20261122_112042_1279907/diagnostic_report.json。
证据来自用户粘贴日志，本地未独立读取服务器完整报告；不覆盖原本地NOT_RUN记录。
三题runner_parity均true，四题面共12项training_layout_passed均true，最终PASS_NUMERICAL_PARITY。
按工具实现，最终PASS还要求两音频解码器一致；训练683与推理383/384的首/第二答案位置logits在容差内一致且argmax相同，首自由生成token一致。
这些证据仅覆盖本checkpoint/三题的对应检查，不推广为所有token/所有模型无bug，也不代表答案正确。

官方原大小写题面重现ppe/ibe循环及无关文本；仅改小写仍混乱/复述反驳指令；紧凑原大小写仍输出south continent/dered sibling/humour。
紧凑全小写分别生成c) india、a) new zealand、c) singapore，均正常EOS结束。
此前用户提供GT分别India、New Zealand、Australia，按原v2解析核对为true/true/false；第三题仅格式恢复、仍答错。
官方小写第三题同时提australia与singapore，原v2会命中文本，但不能据此认为生成正常。
三题是已知异常样本而非随机抽样，不把2/3当完整MMSU分数。

同权重/音频/生成设置的受控题面变化足以触发或解除这三题的退化，支持输入模板/大小写适配问题。
仅lowercase不够，四case未独立拆分instruction/换行/Question/选项符号，不能单独断言某一项是唯一原因。
新旧MMSU实现一致只保证复现旧口径，未保证官方模板适配普通QA小写训练分布；此前仅复用旧模板的评测适配不充分。
正式处理方向为统一MMAU构造器的紧凑小写题面、原模型/音频/greedy和评分保持，并另立协议/新目录，所有比较模型统一重评。
本轮用户询问结果含义，仅记录与解释，尚未改正式评测题面、提交作业或重跑5000题。
README及audit_reports/mmsu_inference_review_validation.json追加user_supplied_remote_followup；原本地验收和生产代码保持。
RM仅前置，以下原字节完整保留，未commit/push。

## 2026-11-22：HRM转换layer5逐层校验误拒绝修正

用户远程转换在阶段2的target layer5报max_abs=1.52587890625e-5；旧校验为逐元素allclose(atol3e-6,rtol3e-5)。
gamma折叠使(RMS(x)*gamma)@W.T变为RMS(x)@(W*gamma).T，实数等价但FP32乘加次序与存储舍入不同；
接近零的输出元素可能触发原绝对容差。之前微型hidden16测试未充分覆盖生产576维和训练后数值规模。
本地真实576/1536维、GQA9:3合成block已复现旧allclose拒绝，max_abs约2.24e-5；独立FP64参考通过。
这支持舍入误报判断，真实远程权重仍需重跑新校验，不宣称实际生产转换已经通过。

仅修改本路线conversion.py的审计，不改变模型forward、权重转换公式、训练配方或其他路线。
先逐张量精确校验20层映射、无参数Norm类型/eps、QKV/gate-up gamma列折叠、零attention gate与o_proj×2。
FP32逐元素预算采用3e-6+3e-5*(当前token源输出RMS+abs(当前源元素))，另检查每token误差RMS预算3e-6+3e-5*源RMS。
不以全batch极值扩大其他token容差；若旧逐元素allclose未通过，额外独立FP64 Norm/投影/RoPE/causal attention/MLP，
分别比较源FP32、目标FP32与各自FP64参考，以及实际已存储折叠权重的FP64前后差（更严3e-7/3e-6预算）。
不能仅调用model.double()冒充全FP64参考，因为现有Norm内部会强制float32。模型参数和生产计算dtype不改。
conversion_audit.json保留每层误差/预算比、旧allclose状态及必要FP64复核；真正的张量或forward错误继续硬拒绝。

新增三项回归：生产维度舍入压力及FP64参考；gate/o_proj/gamma/层映射/eps/非有限故障拒绝；正确张量但错误forward拒绝。
本地CPU完整30/30通过、无跳过，含原两进程DDP、转换重载、checkpoint和数据恢复；3237份原源码摘要保持。
Python3.10语法、Bash/README语法、git diff --check通过；环境仍Python3.13.5/Torch2.15.0+cpu/Transformers4.54.1。
报告audit_reports/text_hrm_h2l3_validation.json已更新；Windows临时目录/Bash受限后获准沙箱外进行合成离线测试。
该远程失败发生在staging创建/目标发布之前，正常Python异常的finally会释放conversion lock；无须删除权重或目录。
本地提交修复并GitHub同步后，远程git pull --ff-only，再运行bash code/RSmol/scripts/convert_text_hrm_h2l3.sh；
转换通过后才提交bash code/RSmol/run_text_hrm_h2l3_3090.sh。未替用户commit/push/远程转换或提交训练。
本条仅前置，以下既有RM字节完整保留。

## 2026-11-21：20层P5/L5/H5/C5 HRM H2L3隔离文本转换与正式训练实现

按用户最终确认，先文本后音频；本次仅新增`code/RSmol/text_hrm_h2l3/`及三份独立Shell入口。
从远程原30层SmolLM2取零基层号0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29；
目标P0..4/L5..9/H10..14/C15..19，20物理层、50次执行、102420288独立参数。
E=无参数RMSNorm(Prelude(embedding))，H0=E/L0=固定向量；两轮各L三次后H一次，末H进Coda/原norm/head。
独立L/H权重各自跨调用复用；hidden576/MLP1536/GQA9:3/RoPE100000/tied head保持。
中段MagicNorm无参数PreNorm保留源eps1e-5，L/H出口及Prelude入口eps1e-6；
逐通道sigmoid attention gate在o_proj前，零gate及o_proj×2；源Norm gamma折叠进QKV/gate-up列。
固定L按官方fmod截断近似、CPU seed0生成FP32持久buffer；不训练、不逐次采样、不跨batch carry。
严格转换验证源库存/finite/绑定、逐层折叠补偿、实际50次轨迹/Norm调用/因果性、保存重载tensor/logits/tokenizer；
原源SHA复核，拒绝已有输出，不覆盖共享资产。整体HRM不宣称与原30层logits等价。

全部causal与真实padding mask，所有有效next-token监督，原85片Parquet/seed0分rank/buffer4096、max1024/动态padding/无packing。
8GPU×micro2×GA64=GBS1024，9244更新；官方Adam-atan2 beta(.9,.95)、全部唯一参数WD.1、EMA.9999。
第s次更新前LR=3e-4*min(s/3000,1)，3000步后恒定；无cosine、梯度裁剪、GradScaler或spike skipping。
BP：1..924为H1/L6/H2，925..1848加L5，1849..9244再加L4；E不全局detach，Prelude从首步收到任务梯度。
FP32参数/优化器/残差，BF16前向、FP32 Norm/CE，TF32关闭；loss为全8rank及64micro有效目标token的统一平均。
每400及9244保存raw HF、独立ema/、全部optimizer与8rank RNG/游标/合同，最近3份正常为8800/9200/9244。
完整resume用新output目录、同源码/数据/依赖/配方，继续总计9244；当前shard起始RNG重放恢复shuffle buffer。
数据身份为名称/大小/mtime_ns，不做全24GB内容哈希，不承诺所有有效文档恰好一次或CUDA bitwise exact。

入口：`scripts/convert_text_hrm_h2l3.sh`、`scripts/train_text_hrm_h2l3_ddp.sh`和`run_text_hrm_h2l3_3090.sh`，均位于code/RSmol。
pdgpu-3090/64CPU/256G/8GPU/1node，原容器、rsmol；直接formal，无新增历史smoke或全量CPU数据扫描前置。
默认转换产物`/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-HRM-H2L3-P5-L5-H5-C5`；
训练输出`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_h2l3/formal_<timestamp>_<pid>/`。
音频训练、MeSH、旧评测适配均未加入本路线；既有模型/训练实现不改。

本地真实CPU回归27/27通过、无跳过；包括实际两进程Gloo/DDP生产累积BP3/4/5与独立全局梯度对照，
真实微型Llama转换/重载/因果性/Prelude梯度、非平凡gamma、完整optimizer/EMA/RNG恢复后下一更新一致、
Parquet跨buffer/shard/epoch重放、85片manifest启动路径、checkpoint最近3份保留及损坏拒绝、真实Bash模拟vc/torchrun。
生产参数量用meta device核对；Python3.10语法、Bash及README命令语法、全部新增文件whitespace检查通过。
3237份原tracked Python/Shell normalized-LF SHA保持；原辅助函数AST保持；官方optimizer仅清两处空行尾空格、算法AST保持。
验证用Python3.13.5/Torch2.15.0+cpu/Transformers4.54.1，与远程Torch2.11/CUDA不同；不冒充3090/八卡GPU PASS。
报告`audit_reports/text_hrm_h2l3_validation.json`，详细Git清单、转换/提交/完整恢复命令见新包README。
Windows测试临时目录/Bash管道受沙箱限制，获准沙箱外运行同一离线合成测试。
未读取远程生产权重/数据、未提交作业或commit/push；3090显存与真实八卡运行仍待远程验证。
本条仅前置，以下RM既有全部字节保留，包括用户本次任务开始前的修改。

## 2026-11-21：MMSU低分/重复生成复核，新增题面与真实训练logits三题诊断

用户反馈多个架构MMSU个位数、三题accent_identification生成无关内容与重复，补充同checkpoint其他benchmark回答正常。
源码逐函数AST确认：text71427新旧MMSU的read_audio/decode_audio/generate/prefix_and_decode一致，
execute/run/summarize一致，dataset/parsing/scorer复用同对象；未发现新适配器改动这些生成或评分步骤。
核对六训练路线：普通QA经text_pair小写转换；训练EOS后缀、PAD!、prompt256、prefix383、attention_mask=None，
answer_loss取382:-1预测answer300，推理取prefix最后logit382，无静态off-by-one；完整683与增量前缀的真实数值仍需GPU验证。
音频mono32k/fixed20及whole CED、FP32/无autocast/TF32off、无cache全重算和完整模型/分支同训练类；eval关闭mapper dropout为预期。
明确差异：MMSU保留官方原大小写/Question:/换行/A.选项及only A/B/C/D指令，而其他正常MMAU/MMAR使用紧凑小写a)格式。
既有v2忽略生成字母、按完整GT文本包含评分，与字母指令不协调；该旧口径此前由用户明确指定，未暗改指标。
格式冲突可解释纯字母被判错，但不能单独解释无关重复生成。题面分布敏感是待实测假设，不宣称根因/修复已确认。

新增隔离code/RSmol/mizar_mmsu_inference_probe/及run_mmsu_inference_probe_4090.sh。
支持text71427和legacy已有MMSU后端，默认精确三题ID；官方/紧凑两格式×原大小写/小写四对照，保留全部选项与None。
每题四次自由生成后，再独立用GT做teacher-forced诊断，GT不进入任何自由生成或正式预测。
比较真实Mellow.forward的683长logits[382]/[383]与prefix383及prefix+首答案embedding的LM调用，记录max差/argmax/首token。
另比较同一小写官方题面的MMSU和MMAU runner token IDs、两音频解码器、prompt IDs/mask、prefix/波形SHA和实际模型审计。
每个4090单卡8CPU/32G作业最多15条自由生成，无benchmark分数；PASS_NUMERICAL_PARITY仅表示数值/runner/音频一致。
新增输出必须不存在，不覆盖或续跑正式结果；生产评测/训练/模型及旧诊断完全不改。

当前base训练目录已改重训LR2e-4新配方，旧34626需要历史匹配源码，严格source hash不放宽。
提交器可--source-revision 8a9f1f76722e77cba4be429a578c63fdcf6ee527，将Git code/RSmol快照解包到共享log子目录；
该revision的26份base .py/.sh与原评测protected_sources一致，真实ckpt合同仍由原loader核验。
--implementation-root选择历史code/RSmol后端，并验证导入路径；不切换当前训练源码/分支，不把登录/tmp交给GPU节点。
以用户此前71427 base的formal_20261119_141531507533438/model--step-34626为例的精确提交命令见新README。

本地联合40项：39通过，1真实CPU小型MeSH测试缺Torch/Transformers/Safetensors/Torchaudio/Soundfile跳过。
新工具12项定向复验11通过/1跳过；原MMSU22项及旧三题诊断6项通过，2880份既有Python/Shell/JSON normalized-LF SHA保持。
真实Bash→Python预览+模拟vc、含空格参数/4090资源、Git历史快照独立进程导入、合成数值错位/首token不符拒绝通过。
Python3.10/Bash语法和git diff --check通过；沙箱临时目录受限后获准沙箱外执行离线测试。
报告audit_reports/mmsu_inference_review_validation.json；模型/音频/GPU在本地诊断测试中明确模拟，真实权重/服务器结果未执行。
未远程提交、commit/push；正式MMSU题面及分数尚未更改。本条仅前置，以下RM原字节完整保留。

## 2026-11-21：原始SmolLM2 + CED-small隔离GBS128正式训练，3090/64CPU

按用户确认计划，从原普通CED-small GBS256完整复制隔离，新增
`code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs128_3epochs_ced_small/`
和入口`code/RSmol/run_mizar_s1_ced_small_gbs128_3090.sh`。
用户补充指定pdgpu-3090、64CPU、256G、8GPU、单节点；原镜像和mellow_c8204d8保持。
原GBS256训练/入口、其他路线、共享模型与全部已有评测代码不改，3166份tracked源码LF SHA保护。

文本仍从`/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2`初始化，原始30个独立物理层、hidden576、
全量LM训练，无递归/MeSH/LoRA。冻结eval CED-small、原官方freq_merge_126 mapper和外置Identity保持；
不加载既有音频checkpoint作为初始化。唯一训练配方变化为micro8→4：8GPU×micro4×GA4=GBS128，
3epochs/每epoch11542更新/总34626，沿用round(5%)得到warmup1731；每epoch消耗1477376行、丢弃116行。
LM max/min LR3e-4/3e-5，mapper6e-4/6e-5；原AdamW/GradScaler/GradNorm/LambdaLR/FP32和TF32关闭保持。
原hybrid1477492行、来源范围/顺序/seed、fixed20中心裁剪、整段mel/BN/时频排列、683布局、
S1 prompt/answer EOS、PAD!/EOS0、attention_mask=None及answer-only shifted CE、rank内token mean→GA4→DDP保持。
21份模型/数据/优化器/staging等文件normalized-LF一致，完整trainer仅描述/报错字符串变化，更新计算完全继承。

保存5771/11542/17313/23084/28855/34626及model--epo-3.ckpt；run.sh完整34626步PASS检查同步。
新route为ced_small_mizar_s1_fixed20_gbs128_v1，独立output/log/job/index-cache/tmpfs和resume身份。
默认输出`outputs/RSmol/mizar_s1_ced_small_gbs128_3090/formal_<timestamp>/formal/`。
CPU index、共享音频立即训练/默认4workers后台复制、optimizer边界8rank MIN一致切换、先reap再owner/token清理保持；
tmpfs保留原守卫前缀`/dev/shm/mizar_s1_gbs256_ced_small_gbs128_<token>`，实际GBS128由合同明确记录。
直接formal，无新增smoke前置；完整恢复保存原模型/optimizer/scheduler/scaler/GradNorm/8rank RNG和数据游标，
只接受本新路线相同源码/资产/数据/依赖/配方checkpoint，沿用原run root，拒绝旧GBS256完整恢复。

本地`python -B tests/validate_mizar_s1_ced_gbs128.py`：56项执行通过、15项真实CPU张量测试因缺依赖明确跳过。
通过实际runtime配置/六保存点、训练geometry拦截、真实checkpoint writer全状态/8rank字段、实际恢复代码rank选择/跨合同拒绝、
16样本rank游标、原正式完成检查、真实Bash模拟vc资源/含空格路径/失败清理、真实tmpfs守卫和原GBS256回归。
Python3.10、2份Bash及4块README命令语法与Git同步清单通过；3166份原源码摘要保持。
官方mapper AST、文本/RNG1344组、tokenizer8组、34627个LR位置误差0、GradNorm10000组对照通过。
Windows沙箱拒绝测试临时目录后获准沙箱外完成同一离线回归；fixture不冒充真实autograd或8GPU恢复。
报告`audit_reports/mizar_s1_ced_gbs128_validation.json`，包内reference_audit为本次审计、parent_reference_audit仅母本历史；
original_files_sha256及两份审计JSON明确放行普通Git同步。完整实现、精确Git提交及恢复命令见包内README。
未读取远程权重/数据、提交真实训练或commit/push；3090显存、生产权重加载及真实训练结果待远程执行。
本条仅前置，原RM字节完整保留。代码经GitHub同步后，远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_gbs128_3090.sh
```

## 2026-11-21：已训练7–14×3–7六槽MeSH接原CED-small/projector，隔离GBS128/3090音频训练

按用户确认计划，从原mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small完整复制隔离，新增
`code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71437_mesh_ced_small/`
及入口`code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced_small_3090.sh`。
默认文本来源为用户指定的
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x3_7_mesh/formal_reinit1200_seed1_20261119_084545_3432609/checkpoint-009244`。
无需重新转换，显式MizarText7_14x3_7MeshForCausalLM加载全部LM/tokenizer及四组已训练routers；
校验完成标记/9244步/四种schedule别名/来源映射，逐张量核对FP32/finite/库存及embedding-head绑定。
28物理层/56执行，0..6、(7..20)×3、21..27；六槽全683位置瞬时memory，八router/16张量/27696参数。
原prefix输出query/value、loop输入query/输出value保持；不重置routers，不读文本training_state.pt。
音频optimizer/scheduler/GradScaler/GradNorm/RNG/游标从step0新建，seed1234及1234+rank。

原CED-small离线strict/冻结eval、whole20 mel/BN分段及正确时频排列、逐样本dB保持；
原freq_merge_126 mapper随机训练，含完整频率MLP/global/位置/audio-type/refinement，dropout.05、resampleFalse、外置Identity。
原hybrid1477492行、fixed20中心裁剪、S1 prompt/answer EOS及原截断/PAD、683布局和answer-only shifted CE保持。
用户确认pdgpu-3090/64CPU/256G/8GPU单节点，micro4×GA4=GBS128，3epochs/34626更新/warmup1731。
LM/routers maxLR3e-4、mapper6e-4；原FP32/无autocast/TF32关闭/GradScaler/AdamW/LambdaLR/GradNorm保持。
loss仍rank-local有效token mean→GA4→DDP均值；不继承文本seed1/BF16或其他音频的统一LR2e-4配方。

独立route/output/log/index/tmpfs；原CPU index、共享音频立即训练/4worker后台复制、optimizer边界8rank MIN集体切换保持。
tmpfs保留mizar_s1_gbs256_安全前缀及text71437独立后缀/token，停止reap producer后再核对owner清理。
直接formal，无新增smoke/人工放行前置；保留真实来源探针，首micro审计三轮输入输出梯度后立即释放引用，检查全部16router梯度。
保存5771/11542/17313/23084/28855/34626及epoch3，完整模型/AdamW/scheduler/Scaler/GradNorm/八rank RNG和下一数据游标。
恢复须本新路线同源码/资产/文本来源/数据/依赖/配方、原外层run-root；恢复前检查完整字段/合同SHA/几何/游标及八rank RNG。
原路线和共享文本实现不改，旧评测器尚未登记本新合同，需后续单独适配。

本地75项执行通过、32项真实ML依赖测试明确跳过，PASS_LOCAL_WITH_SKIPS；3166份原tracked源码LF SHA保持。
包括实际生产forward控制流的独立NumPy三循环/六槽递推、分片16router库存/错误tensor拒绝、恢复实际分支/rank3 RNG、
原完整epoch更新循环/优化器/数据AST、真实Bash+模拟vc、CPU index/后台复制/失败清理及真实tmpfs守卫。
官方mapper AST、文本/RNG1344组、分词8组、34627个LR位置误差0、GradNorm10000标量对照通过。
Python3.10/Bash及README命令语法通过；Windows沙箱阻止临时目录/Bash管道后获准沙箱外执行离线回归。
报告`audit_reports/mizar_s1_text71437_ced_small_validation.json`及包内reference_audit.json；源快照JSON明确Git放行。
真实生产权重加载、3090显存、八卡训练/恢复未执行；未访问远程资产、提交作业或commit/push。
本条仅前置，以下RM既有字节完整保留。完整配置、同run恢复及精确Git同步清单见新包README。
GitHub同步后在远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced_small_3090.sh
```

## 2026-11-21：普通71427 CED音频从step34626仅权重重训3epochs，全局LR2e-4/2e-5

按用户要求直接修改现有`code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced_small_3090.sh`及其配套训练代码，
默认`--init-checkpoint`为：
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_text71427_mesh_3090/formal_20261119_141531507533438/formal/checkpoints/model--step-34626.ckpt`。
这是新一轮训练：完整继承音频projector、全部LM/已训练routers及冻结CED，重新建立optimizer/scheduler/
GradScaler/GradNorm/RNG/数据游标，从step0重新warmup1731并完整训练3epochs、34626更新。
来源34626/epoch3只作权重来源记录，不跳过本轮训练，不恢复其AdamW动量、LR、RNG等训练状态。

所有可训练参数（LM/router/projector）统一max_lr=2e-4、min_lr=2e-5，不再对mapper使用两倍LR。
原8GPU×micro4×GA4=GBS128、pdgpu-3090/64CPU/256G/单节点、镜像和mellow_c8204d8、
FP32/原GradScaler/AdamW/GradNorm/LambdaLR、S1 prompt和answer EOS、fixed20中心裁剪、数据及staging保持。
默认新run目录为`formal_reinit34626_lr2e4_<timestamp>`；保存点仍5771/11542/17313/23084/28855/34626和epoch3。

新增包内audio_initialization.py：登录CPU只读源run的formal/runtime.json/合同/文件metadata，不反序列化权重；
计算节点绑定checkpoint/runtime SHA和来源步数，CPU mmap checkpoint后只使用state_dict，在优化器创建前strict加载。
逐张量校验完整inventory/shape/dtype/finite、冻结CED与配置assets一致、embedding/head源值及绑定、12router及projector完整；
加载后全部模型值精确相等，记录AUDIO_WEIGHTS_INIT_PASS和model_initialization。
来源旧LR和源码hash允许与新配方不同，但checkpoint内部合同SHA必须与其自己同run runtime一致；
缺失/错误架构、分支模型、伪造步数或矛盾别名均拒绝。文本9244仍作结构/tokenizer bootstrap，最终值由音频权重覆盖。
--resume-checkpoint仍为本次新配方同run完整恢复（须相同init来源），不会重新执行权重初始化；
旧配方checkpoint只可用于权重初始化，完整恢复/严格历史评测须用与其保存时匹配的Git源码版本。

本地回归64项通过，33项真实Torch/Transformers/Torchaudio等依赖测试明确跳过。
新增权重初始化/源合同、final checkpoint可作为新起点、统一LR实际分组与起终点、config绑定和Bash参数链验证；
新增两项真实小模型测试覆盖projector/LM/router继承、旧状态不恢复、新optimizer/step0及两步新训练、
缺张量/dtype/nonfinite/tied冲突和源文件保护，本机缺依赖未执行。
3187份本任务范围外tracked源码LF SHA保持，原文本模型/其他训练分支/评测源码不改；Python3.10/Bash语法和git diff --check通过。
原官方函数对照继续通过（mapper AST、文本/RNG1344、tokenize8、LR34627位置及GradNorm10000）；
参考审计明确记录本次统一LR及音频仅权重初始化差异，不再声称优化器LR与母本相同。
报告audit_reports/mizar_s1_text71427_ced_small_validation.json；沙箱外本地离线测试获准执行。
未读取远程资产、提交真实训练或commit/push；实际源权重/3090显存及重训结果仍由服务器任务确认。
本条仅前置，以下RM既有字节完整保留。GitHub同步后正式提交：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced_small_3090.sh
```

## 2026-11-21：修复四模型评测提交的别名错配，新增CPU预检与4090入口

用户报告MMAU/MMAR/MMSU在checkpoint身份校验阶段失败，MMSU generated=0。
用户补充的listen runtime.json为正确的listen/formal路线；其失败报告却记录eval_variant=base、training_route为普通base。
据此确认listen被按base模型提交并拒绝；不据此推定其他所有失败的具体字段或权重损坏。
保留严格合同校验，新增逐字段expected/actual及checkpoint/runtime模型别名诊断；不放宽为接受其他路线。
提交器默认在申请GPU前做CPU预检：runtime/ckpt header/step/合同及训练、共享源码；同一checkpoint四任务只读一次。
CPU通过后仍由原GPU loader严格审计资产与完整张量；CPU不构造模型、不读optimizer张量值、不使用CUDA。
显式错误alias拒绝；可选auto按runtime.json精确注册route/完整formal合同SHA识别，再核验ckpt内部route。
新增run_text71427_ced_eval_4090.sh；旧5090入口兼容，所有任务均默认pdgpu-4090/单卡8CPU/32G。
默认输出改为<run-root>/eval/text71427_ced_v2/<task>/<checkpoint-stem>，保留旧v1失败记录与身份锁。
原推理、预处理、题面、manifest、greedy和评分函数不变，训练目录、共享模型、旧评测源码保持。

本次用户先评测四个model--step-34626.ckpt，固定清单code/RSmol/text71427_ced_eval/rerun_20261121.json：
base formal_20261119_141531507533438；audio7_read2 formal_20261120_111448563391250；
listen formal_20261119_141724655549596；loop_lora formal_20261119_141712678007968。
全部完整绝对路径取自用户；wfuse/ar6不在这次16项重跑清单。产物内容/远程训练与评测PASS未代为核验。
远程仓库根执行：bash code/RSmol/run_text71427_ced_eval_4090.sh batch all --targets-json "$PWD/code/RSmol/text71427_ced_eval/rerun_20261121.json"。
也可将all分别换为reasonaqa5000、mmau、mmar、mmsu，各提交四模型；两种方式择一。
--preflight-only仅做CPU检查；READY_FOR_GPU不是推理通过，SUBMITTED仅代表vc提交成功。
日志/提交诊断在code/RSmol/log/text71427_ced_eval，完整结果在各run的v2 evaluation_report.json。

本地完整验收50项：44通过，6真实CPU张量因缺Torch/Transformers/Safetensors明确跳过。
覆盖本次base/listen错误复现、提交前拦截、auto精确绑定、实际CPU预检控制流、16项清单/4090参数、
失败继续/单checkpoint只预检一次、真实Bash参数链、六模型四任务完整尺寸合成流程与续跑、原函数AST和协议对齐。
3239份原源码normalized-LF SHA保持，Python3.10/Bash语法及git diff --check通过。
Windows沙箱再次阻止测试临时目录后，获准沙箱外完成离线回归；不读取远程资产或提交真实作业。
报告audit_reports/text71427_ced_eval_debug_validation.json；原验收与4090队列记录保留并链接本次报告。
未commit/push；本条仅前置，以下既有RM原字节完整保留。

## 2026-11-21：六模型四项评测全部默认改为pdgpu-4090

按用户要求，将text71427_ced_eval/submit.py统一默认队列改为pdgpu-4090，覆盖六模型的
ReasonAQA5000、MMAU test-mini、MMAR、MMSU全部24种组合及批量提交。
保留原run_text71427_ced_eval_5090.sh兼容入口名，现有命令直接使用，无需额外传--queue。
单卡/8CPU/32G/1node、模型/数据/推理/评分实现保持；显式queue覆盖功能仍保留。
README和原有队列断言同步，新增实际vc参数的4090断言。
12项定向回归全部通过，含24种绑定、真实Bash参数链、模拟vc失败继续、3239份原源码SHA及原协议/AST保护。
Python3.10语法和git diff --check通过；使用已确认的沙箱外临时目录测试，不访问远程或提交真实作业。
原38项完整验收结果保留，队列复验单列audit_reports/text71427_ced_eval_validation.json的queue_4090_followup。
本条仅前置，以下既有RM字节完整保留；未commit/push。

## 2026-11-20：六条7–14×2–7 CED路线接入四项隔离评测，checkpoint显式传入

新增独立包`code/RSmol/text71427_ced_eval/`及入口
`code/RSmol/run_text71427_ced_eval_5090.sh`，不修改六个训练目录、共享模型或任何旧评测源码。
模型别名base、loop_lora、listen、audio7_read2、wfuse、audio7_read2_ar6，分别对应用户指定的
六条GBS128/text71427/3090音频训练路线；全部28物理层/42执行，完整保留各自MeSH、音频分支和两轮LoRA。
复用原ReasonAQA5000同一manifest/完整答案精确匹配、MMAU与MMAR原始输出官方评分、
MMSU官方大小写题面及mmsu_answer_text_contains_v2，不改变原数值推理或评分设置。

FP32/eval/seed1234/TF32off、fixed20/32k、126+1+256 prefix383、PAD!/EOS0、
原EOS后缀分词/截断、attention_mask=None、无KV完整重算、纯argmax300/首EOS停止保持。
后五条每题编码一次并显式逐forward传入(A4,A8)，每forward重建memory；七槽seed5/6只读。
严格加载完整音频state_dict，核对来源/代码/资产/runtime/geometry与张量库存/shape/dtype/finite、
绑定embedding/head；训练后分支fresh=False审计在推理冻结前完成，不重置已训练参数。
只读hooks核对42层轨迹、router/分支边界与两轮各98次LoRA投影，保留196调用/392矩阵区别。

所有checkpoint必须显式--checkpoint，不扫描latest、不固定最终step，也不假设正在训练的run已完成。
标准路径自动推导外层run root及同run formal/runtime.json；step文件核对payload，epoch文件从payload取得step。
接受已完整保存的中间及最终checkpoint；非标准路径须显式runtime-config/output-dir。
评测仅使用模型状态，不恢复训练optimizer等状态，不使用训练resume的“已完成不可恢复”门槛阻止最终评测。
独立输出为<run-root>/eval/text71427_ced_v1/<任务>/<checkpoint-stem>/，身份/进程锁/坏行检查及剩余题续跑保留。
支持单项、单模型all四项及显式JSON清单；失败继续其他任务并整体非零，SUBMITTED只表示提交成功。
默认ReasonAQA5000为4090，其余5090，均1GPU/8CPU/32G/1node、原镜像/mellow_c8204d8；支持显式queue覆盖。
完整四项及六模型提交、参数预览、续跑/CPU比较和精确Git同步命令见新包README。

最终本地`python -B tests/validate_text71427_ced_eval.py`：38项中32项执行通过、6项真实CPU模型
因缺Torch/Transformers/Safetensors明确跳过，PASS_LOCAL_WITH_SKIPS；3239份原tracked源码LF SHA保持。
通过24种绑定与完整尺寸合成流程、原生成/预处理/评分输入函数AST及协议对照、严格加载顺序/错误张量拒绝、
42层/两轮LoRA hooks、同题context、5000与1000完整分母、13题中断续跑、固定官方MMSU scorer及汇总错误拒绝。
实际Bash→Python参数链与模拟vc通过，job完整名小于60位，Python3.10语法与Git可同步清单通过。
模型/音频/Parquet/CUDA在合成流程中明确模拟，bulk fsync加速模拟，不冒充真实autograd或GPU验收。
Windows沙箱临时目录与Git index.lock受限后获准沙箱外完成离线检查；报告audit_reports/text71427_ced_eval_validation.json。
未访问远程生产权重/数据、提交真实作业或commit/push；实际权重加载、显存与正式分数仍待服务器执行。
本条仅前置，以下RM既有历史字节完整保留。

## 2026-11-20：ar6音频训练用户报告step5771后失败，给出同run完整恢复命令

用户提供checkpoint：
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced48_audio7_read2_ar6_lora_r8_gbs128_text71427_3090/formal_20261120_124034240740261/formal/checkpoints/model--step-5771.ckpt`。
用户报告任务到此失败；仅据用户信息记录推进到5771及该产物路径，失败原因/文件完整性未远程核验，不登记完整训练PASS。
恢复沿用原run root `formal_20261120_124034240740261`（不追加内层formal），使用原训练源码/资产/配置，
原ar6入口传`--output-dir <原run root> --resume-checkpoint <上述ckpt>`，恢复模型、优化器等全状态及8rank RNG/游标，继续至总计34626步。
本次仅核对本地入口/README并提供命令，未代为提交远程任务；训练实现及原历史字节保持。

## 2026-11-20：7–14×2–7 CED4/8 原write value拼接融合，恒等初始化＋两轮独立LoRA r8

按用户最终确认，从listen+LoRA母本完整复制隔离，新增
`code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced48_wfuse_lora_r8/`
及入口`code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_wfuse_lora_r8_3090.sh`。
母本、共享模型、其他路线不改；142份已纳入Git的母本/参考/共享文件normalized-LF SHA保护。
新快照及历史参考JSON明确放行Git同步；旧路线两份未跟踪snapshot不作为新路线验收依赖。

默认仍严格继承
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/formal_reinit1000_lr6e4_20261118_112042_3963870/checkpoint-009244`。
完成标记/config/全部FP32 safetensors及原六router严格核验后再注册新模块，复核继承值/Parameter对象和embedding-head绑定。
只加载文本权重/config/tokenizer，音频optimizer等从step0新建，不读文本training_state.pt。

28物理层/42执行与五槽全683位置memory保持；原六router/12张量/17310参数保持。
两套独立无bias Linear1152→576，输入顺序[边界音频hidden;对应CED MLP输出]，权重精确[I576,0]，整个矩阵训练。
逻辑边界7：prefix输出前126位置与A4融合，作为write0 value；边界21：第一轮输出前126位置与A8融合，作为write1 value。
write/read query完整沿用原规则：prefix用prefix output，循环用该轮input；第三次write value仍第二轮output。
SEP/文本value不变，非原地构造value，不覆盖query/循环边界hidden；不保留额外listen router/gate或额外memory写入。
初始融合等价原value，LoRA B零，数学上等价同文本权重/主mapper/输入的原五槽CED+MeSH；更新后不要求训练轨迹相同。

CED冻结/eval，一次编码提取final与完整block4/8 norm前输出；正确时频排列、官方freq_merge_126主mapper、两独立1536-576-576 MLP保持。
原392矩阵/2279424参数的两轮LoRA r8保持，基础W共享并全量训练；无KV，拒绝decoder gradient checkpointing。
融合矩阵1327104、两MLP2443776，加LoRA相对普通71427 CED母本新增6050304；比listen母本多1320180。
首步Encoder半矩阵可有任务梯度，因该半块初值零，两MLP首步任务梯度零是预期；零B时LoRA A梯度零同理。
分别审计融合矩阵两半、原router及两轮LoRA；保留母本32次真实更新累计解锁并随checkpoint保存。
metrics记录融合前后/变化RMS、Encoder半矩阵L2及hidden半矩阵偏离I的L2，不把WD变化当任务梯度。

原hybrid1477492行、fixed20中心裁剪、EOS/PAD、683布局、answer-only shifted CE、rank-local token mean→GA4→DDP保持。
pdgpu-3090/64CPU/256G/单节点8GPU，micro4×GA4=GBS128，3epochs/34626更新/warmup1731。
LM/原router maxLR3e-4，mapper/MLP/fusion/LoRA6e-4；原FP32/GradScaler/AdamW/LambdaLR/GradNorm保持。
原数据、CED/mapper、LoRA和staging源码逐字节继承；完整epoch更新循环仅分支诊断不同。
独立output/log/index/tmpfs，保留CPU index、共享立即训练/后台复制/optimizer边界8rank集体切换和owner/token安全清理。
直接formal，无新增smoke前置；每5771更新及epoch3最终保存。
完整resume保存全部模型/分支、具名optimizer库存与顺序、scheduler/Scaler/GradNorm、8rank RNG及下一数据/审计游标，
严格检查路线/代码/资产/数据/配方、LR/WD/步数、moment和模型张量shape/dtype/finite，不重置已训练融合矩阵。

本地便携验收103项通过、21项真实张量测试缺Torch/Transformers/Torchaudio/Soundfile/Safetensors明确跳过。
覆盖实际forward NumPy独立递推/初始等价/正确query与value范围、有限差分、初始化RNG、真实checkpoint writer的8rank字段、
具名恢复拒绝、梯度解锁、Bash+模拟vc、CPU index/后台复制/失败清理、原音频和文本回归及Git可同步性。
此前另跑20项需本地历史snapshot的旧路线回归通过，不计入最终便携验收103项。
官方文本/RNG1344次、tokenize8组、34627个LR位置误差0、10000次GradNorm标量与mapper AST对照通过；142份原源码摘要保持。
Python3.10/Bash语法通过；Windows沙箱阻止临时目录后，获准在沙箱外完成本地离线测试。
报告`audit_reports/mizar_s1_text71427_ced48_wfuse_lora_r8_validation.json`及包内`reference_audit.json`。
真实CPU autograd/生产checkpoint和数据、3090显存及8GPU训练未验证；未访问远程、提交真实作业或commit/push。
本条仅前置，以下既有RM字节完整保留；模型、输出、完整恢复和精确Git同步命令见新包README。
GitHub同步后远程执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_wfuse_lora_r8_3090.sh
```

## 2026-11-20：audio7 read2隔离六个独立音频routers，五槽write/read2与七槽read0/1，全部LR6e-4

按用户确认计划完整复制母本audio7_read2，新增独立包
`code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_ar6_lora_r8/`
及入口`code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_ar6_lora_r8_3090.sh`。
ar6表示六个独立audio routers。母本、原入口及共享文本模型不改，144份原文件normalized-LF SHA保护。

仍严格继承文本来源
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/formal_reinit1000_lr6e4_20261118_112042_3963870/checkpoint-009244`。
先核验完成标记/config/tokenizer、全部FP32 safetensors及原六router，再安装音频副本与LoRA；
复核原tensor值、Parameter对象及embedding/head绑定。只继承权重，音频optimizer等从step0新建，不读文本training_state.pt。
三个audio write分别完整复制原write0/1/2，audio read2完整复制原read2的weight/bias；
audio read0/1仍按母本扩展七槽：W[:5]=W_text，b[:5]=b_text-log(2)，两新行均取W_text[0]/b_text[0]。
十二个router模块/24张量存储完全独立，CPU fork_rng保持外部RNG；实际loader逐张量检查六个音频分支初值。

每forward prefix前仍初始化audio7 `[E,0,0,0,0,A4,A8]`、SEP/text5 `[E,0,0,0,0]`。
前126位置的三个write均为独立五槽，只写前五槽；read维度7/7/5；位置126起由原六个五槽text routers处理。
A4/A8槽5/6整次forward只读；prefix query/value均为prefix输出，循环query取该轮输入、write value取该轮输出。
用户明确选择全部六个audio router最大LR6e-4，文本router/基础LM仍3e-4；
因此相对母本同时改变四处参数共享与其音频副本的LR。audio routers共19,618参数，全部routers36,928；
较母本额外11,540；加两MLP/两轮LoRA共新增4,742,818，主mapper另计。

CED/mapper/两MLP/392矩阵LoRA及28物理层42执行保持；完整trainer AST只允许router审计及报告计数变化。
原hybrid1,477,492行、fixed20、683布局、EOS/PAD、answer-only CE、rank-local token mean→GA4→DDP保持。
FP32/GradScaler/AdamW/LambdaLR/GradNorm保持；pdgpu-3090/64CPU/256G/8GPU，micro4×GA4=GBS128，
3epochs/34626更新/warmup1731、每5771更新及epoch3最终保存保持。直接formal，无额外smoke前置。
独立route/output/log/index/tmpfs；保留CPU index、共享音频立即训练/后台复制/optimizer边界8rank共同切换及owner/token安全清理。
完整resume保存全部audio routers及原模型/分支、具名optimizer库存、scheduler/Scaler/GradNorm、8rank RNG和数据/审计游标；
拒绝母本或其他路线完整恢复，严格检查新增六角色/6e-4、24个router梯度及模型库存，恢复后不重置初值。

本地`python -B tests/validate_mizar_s1_text71427_ced48_audio7_read2_ar6.py`：142项执行通过、20项真实张量缺依赖明确跳过。
其中新路线41项通过/11项跳过，包含原母本及相关路线回归；实际源码NumPy初始前向对齐、
十二router局部分支独立扰动、六audio router有限差分、正确query/只读seed、实际优化器分组、
24张量梯度审计、全状态writer/八rank RNG/恢复拒绝、真实Bash模拟vc、CPU index/后台复制/失败清理通过。
144份原源码摘要保持，Python3.10及Bash语法通过；Windows沙箱阻止临时目录后获准沙箱外离线验证。
报告`audit_reports/mizar_s1_text71427_ced48_audio7_read2_ar6_validation.json`及包内`reference_audit.json`。
真实CPU用例另覆盖HF分片加载/初值/独立存储、683位置反传/局部梯度隔离、具名优化器/GradNorm/RNG下一步恢复，
本机缺Torch/Transformers/Torchaudio/Soundfile/Safetensors未执行，不能记为真实GPU PASS。
未访问生产权重/数据、提交真实作业或commit/push。RM仅前置，以下旧内容字节完整保留。
本地Git提交、模型配置、输出与同run完整恢复命令见新包README。GitHub同步后远程执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_ar6_lora_r8_3090.sh
```

## 2026-11-20：7–14×2–7 / CED4、8 七槽只读初始化 / 两个循环前音频read / 独立LoRA r8

按本会话用户最终确认，新增独立正式路线：
`code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_lora_r8/`
及入口`code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_lora_r8_3090.sh`。
复制text71427 CED-small母本，沿用已有CED4/8特征与14层两轮LoRA实现；原路线和共享模型不改。
104份母本/参考/入口/共享文件normalized-LF摘要见包内`protected_sources.sha256`。

默认文本权重/config/tokenizer来源为
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/formal_reinit1000_lr6e4_20261118_112042_3963870/checkpoint-009244`。
先校验完成标记/config/全部FP32 safetensors/原六router及绑定embedding/head，再安装新分支；
复核继承值及Parameter对象不变。只继承权重，音频optimizer等从step0开始，不读取文本training_state.pt。

仍为28物理层/42次执行，schedule为0..6、7..20、7..20、21..27。
每完整forward在prefix前初始化audio7 `[E,0,0,0,0,A4,A8]` 与SEP/text5 `[E,0,0,0,0]`。
仅前126音频位置、prefix后/第一轮后（第二轮前）的read0/read1扩为七槽；文本read0/read1及全部read2保持原五槽。
三个write均用原五槽参数并仅更新前五槽，A4/A8槽5/6整次forward只读，不重复注入/无gate或额外交互。
prefix query/value均为prefix输出；循环交互query为该轮输入，write value为该轮输出。
两个独立新read严格执行最终公式：W_new[:5]=W_text，b_new[:5]=b_text-log(2)，
W_new[5/6]=W_text[0]，b_new[5/6]=b_text[0]（用户最后将原讨论中的bias[2]改为bias[0]）。
同query原五槽内部相对分布保持，两新槽初始等权；组质量随query变化，用户明确接受不固定1/2、1/4、1/4。
新增read共8,078参数；全部router共8模块/16张量/25,388参数。

CED-small离线/冻结/eval，一次编码取得最终及完整block4/8输出（final norm前）；主mapper仍官方freq_merge_126。
两个独立1536-576-576 MLP产生A4/A8，时频reshape正确，保留反向梯度；主mapper/MLP在no_grad外。
仅物理层7..20的q/k/v/o/gate/up/down安装两轮独立LoRA，r8/alpha16/scale2/dropout0，A Kaiming/B零。
原主干共享且全量训练，392矩阵/2,279,424参数；加两MLP2,443,776与两read8,078，总新增4,731,278。
逐调用选择loop并finally复位，拒绝decoder gradient checkpointing与KV cache。

资源pdgpu-3090/单节点8GPU/64CPU/256G；micro4×GA4=GBS128，hybrid1,477,492行，3epochs/34,626更新/warmup1,731。
基础LM/原router最大LR3e-4；mapper/新MLP/read/LoRA6e-4。原FP32/GradScaler/AdamW/LambdaLR/GradNorm、
683布局/EOS/PAD/answer-only shifted CE及rank-local token mean→GA4→DDP保持。
独立output/log/index/tmpfs，CPU index及共享音频立即训练/后台复制/optimizer边界8rank集体切换、安全清理保持。
直接formal，不附加smoke前置。保存每5771更新及epoch3最终文件。
完整恢复绑定新路线代码/资产/数据/配方，保存所有分支、AdamW/scheduler/Scaler/GradNorm、8rank RNG和数据/审计游标，
检查具名参数库存/顺序/LR/WD/步数和模型张量shape/dtype/finite，恢复后不重置新read或LoRA。
前两步及每25步检查missing/nonfinite和LoRA轨迹；零梯度作诊断，B零时A任务梯度为零符合预期；无gate解锁期限。

本地101项回归通过、9项真实张量检查因缺Torch/Torchaudio/Transformers/Soundfile/Safetensors明确跳过。
实际源码NumPy前向核对三个交互、只读seed、完整重建、指定概率公式与RNG；
实际checkpoint writer/八rank RNG字段、具名库存恢复拒绝、梯度审计、真实Bash+模拟vc提交、
CPU index/后台复制/失败清理及原音频/文本回归通过，104份原源码摘要保持；Python3.10及Bash语法通过。
Windows沙箱阻止临时目录/Bash后获准在沙箱外进行离线验证。
报告`audit_reports/mizar_s1_text71427_ced48_audio7_read2_validation.json`。
真实CPU autograd、生产checkpoint/数据、显存和8GPU训练未验证；未访问远程或提交作业，未commit/push。
以下已有RM字节保留。完整模型/恢复说明见新包README；代码同步到GitHub后在远程执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_lora_r8_3090.sh
```


## 2026-11-19：7–14×2–7 CED4/8 listen + 两轮独立LoRA r8，隔离GBS128正式音频路线

按用户确认，复制3090/GBS128的text71427 CED-small母本，新增隔离包
`code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced48_listen_lora_r8/`
及入口`code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_listen_lora_r8_3090.sh`。
指定checkpoint-009244模型/config/tokenizer及六个原MeSH router的12张量完整继承，音频optimizer从step0新训练。
不读取文本training_state.pt。严格loading_info/全部safetensors逐张量精确继承检查后才注册新分支，
再检查继承值及Parameter对象、embedding/head绑定保持。

本包模型源独立复制共享71427模型，保留28物理层/42执行、五槽全683位置memory及原write/read查询规则。
一次冻结/eval CED-small forward取得最终特征和第4/8完整block输出（final norm前）；
最终特征保留原freq_merge_126主mapper，第4/8层各有独立1536-576-576 projector。
边界7/21执行“原write → listen写memory音频位置 → 原read”，分别接CED4/8；
gate/listen router查询前段输出/第一轮输出，原循环router继续查询该轮输入，write value为该轮输出。
非原地写入原五槽，仅前126音频位置，无第三次listen、额外gamma或除以4。
gate全零；listen router正态std=sqrt(2/(5*576))后clamp±3std、bias零，非trunc_normal重采样。

仅物理层7..20七投影q/k/v/o/gate/up/down加入两轮独立LoRA，基础W共享且全量训练；
rank8/alpha16/scale2/dropout0，A Kaiming/B零，392矩阵、2,279,424参数。
两个projector/gate/listen新增2,450,700参数，合计新增4,730,124。CPU fork_rng保持外部RNG，拒绝decoder gradient checkpointing。
原数据/中心fixed20/EOS/PAD/683布局/answer-only CE/原损失归约、FP32/AdamW/LambdaLR/GradNorm/GradScaler保持。
pdgpu-3090/64CPU/256G/8GPU，micro4×GA4=GBS128，3epoch/34626步/warmup1731；
基础LM/原routers最大LR3e-4，其余新分支与主mapper6e-4，原保存节奏保持。
独立output/log/index/tmpfs，保留CPU index及共享音频立即训练/后台复制/optimizer边界集体切换和安全清理。
完整音频resume保存全部新增分支、优化器等全状态、8rank RNG/数据游标及累计梯度解锁状态，严格核对身份/库存/LR/游标。
首次microbatch审计两轮边界后释放；首步允许零gate/B的预期零任务梯度，全部新角色需32次真实更新内全rank累计解锁。

本地回归87项通过、19项缺Torch/Transformers/Torchaudio等真实ML依赖明确跳过。
通过实际Bash+模拟vc、CPU index/后台复制/失败清理、生产listen/LoRA NumPy数值/有限差分、
12角色梯度审计、完整恢复拒绝条件及原音频/文本回归；103份母本/参考/共享文件normalized-LF SHA保持。
官方对照1344次文本/RNG、8组tokenize、34627个LR位置误差0、10000次GradNorm标量及mapper AST一致。
新增真实CPU测试覆盖CED hooks/梯度、严格继承、零分支等价、392独立矩阵、42调用/683/两轮反传、
answer shift和完整恢复后下一步一致；本机缺依赖未执行，不记作GPU PASS。
报告`audit_reports/mizar_s1_text71427_ced48_listen_lora_r8_validation.json`及包内`reference_audit.json`。
Windows沙箱阻止临时目录/Bash后获准沙箱外离线回归；未访问远程权重/数据、提交GPU作业或commit/push。
本条仅前置，以下已有RM字节完整保留；提交与完整恢复命令见新包README。GitHub同步后远程执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_listen_lora_r8_3090.sh
```

## 2026-11-19：7–14×2–7 MeSH接CED4→8注入与两轮独立LoRA r8，GBS128/3090

按用户批准方案，完整复制隔离现有text71427 CED-small音频包，新增
`code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced48_loop_lora_r8/`
及入口`code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_loop_lora_r8_3090.sh`。
用户再次确认GBS128（8GPU×micro4×GA4）、3epoch/34626步，并选择最新RM配置pdgpu-3090/64CPU/256G/单节点8GPU。

严格从指定文本来源
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/formal_reinit1000_lr6e4_20261118_112042_3962870/checkpoint-009244`
继承全部文本/routers，先逐张量核对safetensors，再加入新模块并核验原值不变。只继承权重，音频step0新训，
不读取文本training_state.pt，不重做抽层或初始化routers。保留28物理层/42执行、五槽、六router模块/12张量/17310参数及绑定embedding/head。

冻结/eval CED-small单次编码获得最终特征及block4/8完整输出（final norm之前）；主输入保持官方freq_merge_126 mapper、dropout0.05、外置Identity。
两套独立LN1536→Linear1536→576→GELU→Dropout0→Linear576→576→LN576 MLP；两套零初始化tanh逐token gate，
在MeSH read0/read1之后、逻辑7/21处、两轮14层之前分别注入A4/A8，仅直接写前126个音频位置。
循环router查询各自注入后的输入、写入循环输出，原memory规则保留。

LoRA仅覆盖物理7..20的q/k/v/o与gate/up/down七projection，rank8/alpha16/scale2/dropout0；
每层每projection每轮A/B独立，基础W共享并全量训练，前后各7层无LoRA。392个矩阵/2279424参数；
音频MLP/gate共2444930参数，较无注入母本新增4724354参数。主mapper/新分支/LoRA最大LR6e-4，LM/router3e-4。
新模块隔离CPU RNG初始化；use_cache=False，不支持decoder gradient checkpointing。A4/A8显式随每次forward传递。

保留原hybrid1477492行、S1 prompt/answer EOS、中心裁剪fixed20、整段mel/BN/逐样本dB、683位置、answer-only shifted CE、
rank内microbatch token mean→GA4→DDP均值、FP32/GradScaler/无autocast无TF32、原AdamW/LambdaLR/GradNorm及1731步warmup。
保留CPU index、共享立即训练/4worker后台复制、完成后optimizer边界8rank集体切换、停止/reap producer后owner/token清理。
tmpfs前缀`/dev/shm/mizar_s1_gbs256_text71427_mesh_ced48_loop_lora_r8_`满足原安全守卫，实际GBS128。
route/output/index/job/log/resume隔离；输出根`outputs/RSmol/mizar_s1_ced48_loop_lora_r8_gbs128_text71427_mesh_3090/formal_<timestamp>`，
实际训练在其formal/。保存5771/11542/17313/23084/28855/34626及model--epo-3.ckpt，完整恢复模型/新模块/优化器等状态、8rank RNG和下一数据游标。

保留母本42调用/683位置memory、两轮边界及12router梯度审计，加入首步零B/零gate与第二步梯度打开、两轮LoRA trace及gate统计。
63份母本/共享依赖normalized-LF SHA未变；剥离明确新增分支后完整骨干及train函数AST一致，公共训练语义合同一致。
本地验证`python -B tests/validate_mizar_s1_text71427_ced48_loop_lora.py`：109项已执行通过，33项真实ML张量测试因缺Torch等明确跳过。
报告`audit_reports/mizar_s1_text71427_ced48_loop_lora_validation.json`及包内reference_audit.json。
包含实际Bash+模拟vc参数、后台复制/失败清理、CPU index、非零LoRA和read后注入的NumPy行为检查、原音频/文本回归。
Windows沙箱阻止临时目录/Git Bash后获准在沙箱外运行离线测试；未读取远程资产、未真实提交训练、未commit/push。
远程实际权重加载、3090峰值显存和正式训练结果尚待计算节点执行；formal直接提交，无额外smoke前置。

GitHub同步后在远程仓库根执行：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_loop_lora_r8_3090.sh
```
完整恢复命令与合同见新增包README。本条仅前置，以下既有RM字节保留。


## 2026-11-19：7–14×2–7 CED-small音频训练切换3090队列/64CPU

按用户补充要求，本次新增音频路线改为pdgpu-3090、64CPU、256G、8GPU、单节点。
提交入口重命名为`code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced_small_3090.sh`；
默认output/index namespace同步改为mizar_s1_ced_small_gbs128_text71427_mesh_3090。
指定文本checkpoint-009244、GBS128（micro4/GA4）、模型/音频/优化器/34626步等合同保持。
只同步新入口、runtime提示、README、测试及参考审计源码身份；旧GBS256路线不改。
58项本地回归通过，31项缺真实ML依赖明确跳过，实际Bash+模拟vc确认-p pdgpu-3090和-c 64。
未实际提交训练或commit/push；本条仅前置，以下历史字节保留。
GitHub同步后远程执行：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced_small_3090.sh
```

## 2026-11-19：CED-small接已训练7–14×2–7 MeSH，隔离GBS128正式音频路线

用户指定文本权重来源：
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/formal_reinit1000_lr6e4_20261118_112042_3963870/checkpoint-009244`。
新增隔离包`code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced_small/`
及入口`code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced_small_5090.sh`。
母本为既有GBS256 recursive_mesh_ced_small；原路线、旧入口和共享文本模型源码均不修改。

只读调研用户提供的本地Mizar源码，确认S1训练prompt/answer共用追加EOS函数、fixed20中心裁剪；
公开推理入口才是prompt无EOS/单音频前20秒。用户最终明确选择严格保留S1训练EOS和中心裁剪。
原text_pair/模板/CoT规则、截断后不补回EOS、PAD已有!、SEP/EOS0、固定683序列及answer-only CE均保持。
原single录音路径恢复规则保持，历史processed filepath2不视为第二条原录音。

显式MizarText7_14x2_7MeshForCausalLM加载28物理层/42执行、五槽/三组write-read（六模块、12张量、17310参数）；
完整继承checkpoint-009244文本权重及已训练routers。不读取training_state.pt，不恢复文本优化器等状态，
音频optimizer从step0新训练，不需再转换或重新初始化routers。校验完成标记/config/全部schedule与来源映射、
严格loading_info、全部safetensors逐张量精确一致及router SHA；只允许省略一个tied embedding/head别名。
共享模型、其contract及release manifest均纳入音频完整resume源码身份。

CED-small原资产严格离线加载、冻结/eval，保持原FP32在线编码、整段mel/BN/1012帧分段与逐样本dB。
新随机训练官方freq_merge_126 mapper（384×4→126×576、dropout0.05、resampleFalse、外置Identity）；
全LM/routers训练。用户因显存顾虑指定8GPU×micro4×GA4=GBS128；3epoch保持，1477492行得到
每epoch11542步/共34626步、warmup1731、每epoch丢弃116行。
LM/router最大LR3e-4、mapper6e-4，原AdamW/LambdaLR/GradNorm/GradScaler/FP32无autocast无TF32保持。
保存5771/11542/17313/23084/28855/34626及model--epo-3；完整状态、8rank RNG和数据游标恢复保持。

沿用pdgpu-5090/8GPU/32CPU/256G/原镜像/mellow_c8204d8，通过vc submit正式提交，无独立smoke前置。
保留CPU index、共享音频立即训练/默认4线程后台复制、完成后optimizer边界8rank MIN集体切换。
新tmpfs使用mizar_s1_gbs256_text71427_mesh_ced_small_前缀以满足原安全守卫（实际GBS128），
保留停止/reap producer后owner/token校验清理，job完整名含后缀小于60字符。
正式任务内部审计实际42次前向/683位置memory及首次两轮边界和12router梯度，及时释放保留张量。

本地回归58项通过，31项真实Torch/Transformers/Torchaudio等依赖检查明确跳过。
通过Python3.10/Bash语法、真实Bash+模拟vc提交参数、CPU index、后台复制/失败清理及原音频/文本回归；
35份母本/共享文件normalized-LF SHA保持。与本地官方源码比较1344次文本/RNG、8组tokenize、
34627个LR位置（误差0）、10000次GradNorm标量及mapper类AST一致。
报告`audit_reports/mizar_s1_text71427_ced_small_validation.json`及包内`reference_audit.json`。
新增真实小模型测试覆盖权重/router继承与缺失拒绝、42调用/683位置、两轮梯度、answer shift及完整恢复；
因本机缺ML依赖未执行，未记为GPU PASS。Windows沙箱阻止测试临时目录/Git Bash后获准沙箱外离线回归。
未读取远程权重/数据，未提交真实训练，未commit/push；远程加载、显存及正式训练结果待服务器执行。

本次RM只前置，以下既有历史字节完整保留。GitHub同步后，在远程仓库根正式提交：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced_small_5090.sh
```
完整配置、输出位置及同run root完整resume命令见新包README。

## 2026-11-19：三循环从checkpoint-001200权重重新训练9244步，seed1换序

用户指定初始化来源：
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x3_7_mesh/formal_20261118_182750_1375514/checkpoint-001200`。
原三循环提交入口、runtime和trainer默认改为该路径，仅加载模型/config/tokenizer，
完整继承LM与四组routers；新建AdamW/scheduler/RNG/数据游标，optimizer step0起完整9244步。
不读取旧training_state.pt，不恢复旧optimizer/RNG/cursor，不需重新转换，不加LayerNorm。
三循环28物理层/56执行/六槽、LR6e-4/6e-5、warmup463、GBS1024、每400步保存保持。

默认seed由0改为1，新增提交入口--seed选项并传到计算节点；原数据管线不改，
seed同时控制分片打乱/分配和片内buffer4096打乱，seed+rank控制训练RNG。
日志与run_manifest记录来源step1200、新起点0、seed及不恢复旧训练状态。
默认新目录formal_reinit1200_seed1_<timestamp_pid>；完整resume仍需匹配旧seed和训练配置。
提交命令显式micro4/GA32及--seed 1并清空可能遗留的RESUME_FROM；脚本micro8/GA16默认仍可选。
正式入口无新增smoke/审查报告/人工放行条件；保持正常模型身份/完成标记及防覆盖检查。
换seed不能保证消除此前nonfinite gradient norm；本次不宣称已解决数值稳定性根因。

二、三循环回归共55项：35通过、20缺Torch等真实张量依赖明确跳过。
通过项含不可反序列化旧training_state的仅权重输入、错误架构/不完整标记拒绝、
分片及片内换序可复现、Bash语法/default与seed覆盖/dry-run/模拟vc参数传递和隔离。
新增真实小模型测试验证已训练routers逐张量继承、新optimizer空状态、起点0和两步新训练，
本机缺依赖未执行。Windows沙箱阻止临时目录/Bash后获准沙箱外执行本地回归。
未修改二循环、三循环模型/转换器/数据代码，未访问远程资产、实际提交训练或commit/push。
本条仅前置，保留既有RM历史字节；完整用法见本路线README。

## 2026-11-18：三循环文本checkpoint改为每400步保存

用户报告三循环在打印step410后再次出现nonfinite gradient norm，暂不加LayerNorm，
计划重新运行取得step400 checkpoint，再以其模型权重重新启动完整9244步。
本次按要求仅把三循环保存间隔由500改为400，同步runtime shell、trainer默认值及
FORMAL配置限制、recipe、既有测试和README。首次保存checkpoint-000400，最终仍保存9244。
最近三个checkpoint保留策略保持，完整运行最后保留8800/9200/9244；长期使用400权重需另存。
本次没有改变模型结构、初始化来源、LR6e-4/6e-5、warmup463或总步数。
从400权重重新训练的默认路径尚未配置；不能把完整resume当作从零重新训练。
三循环回归25项：16通过、9缺Torch等真实张量依赖跳过；模拟vc及Bash检查通过。
未执行远程训练，实际checkpoint-000400产出仍待用户重新提交后的结果。

## 2026-11-18：MMSU补充独立CPU细分类报告

用户询问为何MMSU只有Perception/Reasoning×Linguistics/Paralinguistics四组，
是否能输出更细分类。核对本地固定官方scorer及官方GitHub：评分脚本仅读取
category和sub-category聚合；官方README描述三层分类、47个任务，数据及我们
原始输出已保留sub-sub-category、linguistics_sub_discipline和task_name。

新增code/RSmol/mizar_mmsu_detailed_report/，直接读取任意架构完整5000条旧结果，
优先逐行哈希校验raw，或在raw不存在时读取原始predictions_official.jsonl。
不读权重/音频/Parquet，不启动GPU，复用原mmsu_answer_text_contains_v2和固定
官方scorer。独立子目录detailed_mmsu_answer_text_v2输出JSON/CSV/Markdown及评分副本。
七种视图：category、原四组、三层taxonomy、完整路径discipline、task_name、
完整路径task、榜单六维category×Semantics/Phonology/Paralinguistics。
每组输出正确数/错误数/样本数/准确率，各view必须汇总为5000和相同总正确数，
原四组和总分核对官方scorer；历史成功v2总分也核对，失配拒绝成功报告。
空分类不丢题、未知榜单标签列Unmapped并警告；任务数按实际字段统计，不硬凑47。
分类依据官方字段，判分仍为本项目既有v2文本包含协议，不宣称更换为榜单字母协议。

入口bash code/RSmol/mizar_mmsu_detailed_report/run.sh --source-output-dir <原MMSU输出目录>；
可重复传入source批量处理，不需要vc提交。默认各自独立子目录，支持新output-dir，
同身份可重复运行、错误身份/完整性损坏/忙锁拒绝，发布前复核来源。生产评测代码、
旧原始回答/报告及续跑身份均未改；RM本条仅前置，原历史字节保留。

15项本地测试全部通过、无跳过，包含真实固定官方scorer对合成5000条结果评分、
所有分组/47任务/六维/计数守恒、原v2边界行为、CSV/Markdown/JSON、重复运行、
历史总分和源文件保护、损坏/部分/重复/外部修改拒绝、多源失败继续、无推理依赖导入、
真实Bash入口+模拟conda。Python3.10语法通过；1768份原源码raw/LF SHA及此前三题
诊断5份文件LF SHA保持。报告audit_reports/mizar_mmsu_detailed_report_validation.json。
Windows沙箱此前阻止临时目录和Bash，本次获准沙箱外运行本地测试；无远程资产、
生产结果处理、GPU、真实作业提交或commit/push。真实结果需用户在Linux服务器执行。

## 2026-11-18：7–14×3–7文本训练LR同步降至6e-4/6e-5

用户确认三循环OOM发生在正式训练的`loss_sum.backward()`，显存23.57GiB、
额外申请1.50GiB失败；转换脚本不涉及该OOM。现按用户要求修改原三循环入口及
配套runtime shell、trainer默认值/参数限制和recipe为max_lr=6e-4、min_lr=6e-5。
保持初始化产物、GBS1024、9244步、warmup463；建议显式micro4/GA32提交以降低显存。
正式入口无新增smoke/审查前置条件。转换产物可直接复用，无需重新转换；
新训练从三循环转换产物开始，完整resume仍要求LR及其余训练合同一致。
三循环回归25项：16通过、9因缺Torch等真实张量依赖明确跳过；Bash语法、默认与
micro4/GA32 dry-run均确认新LR，模拟vc通过。未提交真实GPU任务，二循环本次未修改。

## 2026-11-18：7–14×2–7文本从step1000权重重新训练，LR降至6e-4

用户报告二循环文本run `formal_20261117_202454_1706253`已产出
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/formal_20261117_202454_1706253/checkpoint-001000`，
随后在打印step1070的有限loss/gradient日志后出现`RuntimeError('nonfinite gradient norm')`。
片段不能确定失败的精确step及非有限值来源；NCCL Destroy是退出清理，不登记为通信根因。

按用户要求直接修改原入口`code/RSmol/run_stage4_mizar_text_7_14x2_7_mesh_training_5090.sh`：
默认从上述checkpoint仅加载完整模型及tokenizer（保留已训练routers），不读取旧training_state.pt；
新AdamW/scheduler/RNG/数据游标、optimizer step0起，完整训练9244次更新。
max_lr=6e-4、min_lr=6e-5，warmup463、GBS1024、BF16 forward及原数据策略保持；
输出到新的`formal_reinit1000_lr6e4_<timestamp_pid>`目录，不覆盖或清理旧run。
不需重新转换，不增加smoke/审查报告/人工放行前置条件，非有限值失败机制保留。
本次仅模型初始化允许旧8e-4/8e-5 checkpoint；完整`--resume-from`仍严格匹配当前训练超参。
三循环模型、转换、训练及提交代码保持不变，仍使用8e-4/8e-5；只更新其历史复制快照测试说明，
允许二循环后续独立演进，而不再要求二循环源码永远匹配复制日SHA。

本地二、三循环回归51项：32通过，19因缺Torch等真实张量依赖明确跳过。
覆盖checkpoint仅权重初始化（不可读取旧pickle不影响）、新LR调度、Bash语法/dry-run、模拟vc及隔离。
新增真实小模型测试覆盖已训练routers继承、新optimizer空状态、两步新训练及不读取旧训练状态，
因本地缺依赖尚未执行。远程权重加载及新GPU训练待用户运行，不记作远程PASS。
提交命令和完整语义见`code/RSmol/mizar_text_7_14x2_7_mesh/README.md`。

## 2026-11-17：MeSH四循环MMSU异常的独立三题诊断

用户反馈recursive_mesh_5_10x4_5的MMAU/MMAR正常、MMSU约1%。其提供的三条
accent_identification原始response出现naive开头、重复短语/无关词及300token截断，
属于生成退化，不能仅以评分器不识别字母解释；尚未确认根因，不登记修复完成。
用户提供服务器forward audit：649923次检查通过，最后一次完整50次层调用，
0..4,(5..14)×4,15..19，5/15/25/35/45五组write/read，无cache，4090。
空branch事件符合本路线；片段未附checkpoint/runtime/源码身份，仍需核对。

新增隔离包code/RSmol/mizar_mmsu_meshx4_diagnostics/，精确绑定mesh_x4的
formal_20261103_093224843011222/step34626及同run runtime、训练geometry8,4,4。
选定用户报告的三题，比较原MMSU题面/小写/原MMAU紧凑小写及两runner同输入；
复用原严格loader、全部四循环与router审计、FP32/greedy300/无cache等生成设置。
另比较原MMAU音频bytes解码结果，保存原始回答、tokens、音频与模型/来源审计。
诊断输出必须是新目录，不评分、不覆盖原结果，PASS只代表runner及音频一致。
pdgpu-4090单卡/8CPU/32G/1node提交命令见包README，job包含时间/PID后缀小于60位。

6项新增本地测试全部通过，覆盖精确绑定/题面对照、runner差异、三题覆盖、旧目录
拒绝覆盖、错误数据拒绝加载及真实CLI无资产预览；Python3.10与Bash语法通过。
1768份既有源码raw/LF SHA保持，原MMSU/MMAU/MMAR/训练/模型及已有入口均未改。
报告audit_reports/mizar_meshx4_mmsu_diagnostics_validation.json。真实张量/GPU明确模拟；
Windows临时目录权限阻止测试后，获准沙箱外执行本地合成验证。
未访问远程权重、提交真实作业或commit/push，真实三题对照仍待远程运行。
本条仅前置，原RM历史字节完整保留。

## 2026-11-16：MMSU27项完整job名缩短至52位

用户要求包含后缀的完整job参数名严格小于60位。检查发现原最长一项恰好60位，
现仅将新MMSU27项提交器job别名截断从40改为32：mmsu27-别名-12位摘要，
含前缀/分隔符/摘要后缀完整最多52个ASCII字符。时间与PID继续只参与摘要，
摘要仍绑定完整目标名与批次，长别名截断后不混淆；4090资源/模型/推理/评分配置不变。
更新本包两组测试及README；22项本地测试全部通过，无跳过，真实Bash+模拟vc通过。
核对27项实际-j和4种批次长度共108个名称，全部最长52且互异；Python3.10语法通过。
后续结果单列audit_reports/mizar_mmsu_remaining_eval_validation.json的job_name_followup，
此前195项联合验收计数保留。未提交真实作业，原提交命令继续使用。
本条仅前置，原RM历史字节完整保留。

## 2026-11-16：剩余27项MMSU新增pdgpu-4090单卡入口

用户确认此前11项MMSU均已完成并排除，授权为剩余所有已适配版本新增入口，统一pdgpu-4090。
当前MMSU支持38项，固定38减11为27项；精确别名/variant/run/训练route/step/geometry
见code/RSmol/mizar_s1_ced_mmsu_remaining_eval/targets.json，排除清单见excluded.json。
新增入口code/RSmol/run_mizar_mmsu_remaining_4090.sh和独立包，不修改训练/模型、原MMSU/
MMAU/MMAR或ReasonAQA源码，新包不进入原MMSU的源码身份集合。

每项目标一个vc submit作业：pdgpu-4090/1GPU/8CPU/32G/1node，同镜像、mellow_c8204d8
和离线环境，job名最多60字符。24项step17313/训练geometry8,8,4；mesh_x4、
official_serial84_r8和official_input_only为step34626/8,4,4。精确读取model--step-N.ckpt
和同run的formal/runtime.json，不扫描latest，中间checkpoint不干扰；推理始终batch1。
mesh_inject48沿用候选run，提交前原validator做CPU合同预检，失败继续其余目标；
可显式更正该候选run，CPU通过不代替计算节点strict全模型加载。

计算节点直接执行原mizar_s1_ced_mmsu_eval/run.sh，复用全部原架构context和严格loader，
保留音频中间层注入、crossattention/query32、MeSH memory/router及LoRA分支。
原完整5000题顺序与官方大小写题面、mono32k/fixed20、prompt256/prefix383、FP32、
seed1234/TF32off、无cache完整重算、greedy300/首次EOS不变。
评分继续mmsu_answer_text_contains_v2、原固定官方scorer与完整5000分母，原raw不变。
结果沿用各run的eval/mmsu_step17313或eval/mmsu_step34626。

批次先只读核验既有产物：完整v2跳过；完整raw但缺有效v2分数走CPU离线重评，
同身份partial交原runner续跑；忙锁跳过，错误身份/坏完整行保留源文件并失败。
新ledger在code/RSmol/log/mmsu_remaining4090_admissions，SUBMITTING/SUBMITTED
阻止同绑定重复入队，不声称查询集群；旧job已停后单项--retry-submitted显式重提。
inspect仅写独立状态报告，不改评测源产物；compare核对全5000记录/协议/共有代码并输出27行。

本地验证tests/validate_mizar_mmsu_remaining_eval.py，报告
 audit_reports/mizar_mmsu_remaining_eval_validation.json：195项中181通过、14项缺真实
Torch/Transformers/音频/Parquet等依赖明确跳过，PASS_LOCAL_WITH_SKIPS；新增20项全部通过。
覆盖实际Bash+模拟vc、27项目标/资源/排除、完整5000评分/重评/恢复、只读/锁/身份/
失败继续/重复提交和原架构回归。1768份原源码raw/LF SHA保持，Python3.10与两份Bash语法通过。
Windows沙箱拒绝临时目录/Bash，获准沙箱外执行本地合成验证；真实模型/GPU明确模拟。
未读取远程生产权重、提交真实作业或commit/push，不登记远程GPU PASS。

用户经GitHub同步后，在远程仓库根执行：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_mmsu_remaining_4090.sh all --check-args
bash code/RSmol/run_mizar_mmsu_remaining_4090.sh all
```
完成后bash code/RSmol/mizar_s1_ced_mmsu_remaining_eval/run.sh compare，默认
outputs/RSmol/mmsu_remaining27_4090_v2.csv和同名JSON；未完成项分数留空，全部完整才PASS。
单项可用ca36912等别名；详细范围、纠正/续跑/汇总及Git文件清单见新包README.md。
本条仅前置，全部RM历史字节完整保留。

## 2026-11-16：原七项ReasonAQA5000新增pdgpu-4090单卡入口

用户询问七项能否以4090单卡提交，并授权必要修改。现有七项入口硬编码8卡3090，
因此新增独立包code/RSmol/mizar_s1_ced_reasonaqa5000_selected7_single_eval/及入口
code/RSmol/run_reasonaqa5000_selected7_1gpu_4090.sh，不修改已有8卡或原30项源码。
all只提交原七项，继续排除official_mizar；每项一个vc submit作业：
pdgpu-4090/1GPU/8CPU/32G/1node，同镜像、conda mellow_c8204d8与离线配置。
job采用raqa7s-短别名-12位摘要，完整长度最多60个ASCII字符。

直接复用原七项精确checkpoint/run/route、step17313、runtime与训练geometry=8,8,4，
实际推理单卡batch1，不启动torchrun。沿用原严格loader和FP32/seed1234/TF32off/
fixed20/prefix383/greedy300/无cache及逐样本context，不改变完整答案规范化精确匹配。
复用同一5000条manifest，不重新抽样；独立输出目录
eval/reasonaqa_close_ended_sample5000_seed1234_step17313_1gpu4090_v1/，
逐题fsync、同身份续跑、完整5000后计分及无模型重评分，保留原8卡结果。

本地25项全部通过，无跳过，PASS_LOCAL；1120份原文件SHA保持，Python3.10与两份Bash语法通过。
验证入口tests/validate_reasonaqa5000_selected7_single_eval.py，报告
audit_reports/reasonaqa5000_selected7_single_eval_validation.json。
验证包含精确七项绑定、单卡资源/短job名、原单卡生成循环AST一致、
完整5000合成评测/计分/比较/重评分、中断13后续跑4987、身份变化拒绝、
真实Bash入口与模拟vc批次及失败继续，并复核原8卡/数据/计分测试和旧源码SHA。
Windows沙箱拒绝临时目录读写，获准在沙箱外重跑本地合成验证；真实模型/GPU明确模拟。
本次未访问远程生产资产、提交真实作业或commit/push；不登记远程GPU已通过。

代码经GitHub同步后，在远程仓库根执行：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_reasonaqa5000_selected7_1gpu_4090.sh all --check-args
bash code/RSmol/run_reasonaqa5000_selected7_1gpu_4090.sh all
```

完成后bash code/RSmol/mizar_s1_ced_reasonaqa5000_selected7_single_eval/run.sh compare，
默认outputs/RSmol/reasonaqa5000_selected7_seed1234_1gpu4090_v1.csv及同名JSON。
详细说明见新包README.md。本条仅前置，全部RM历史字节完整保留。

## 2026-11-16：ReasonAQA sample5000补充原七项、每目标8卡3090

用户确认official_mizar仍不测，原先已在完整集合测过的七项需要用同一5000条版本重测。
新增独立包code/RSmol/mizar_s1_ced_reasonaqa5000_selected7_8gpu_eval/及入口
code/RSmol/run_reasonaqa5000_selected7_8gpu_3090.sh；all只补交这七项，
与原30项合计37项，不重新提交原30项。原30项可能正在运行，因此原评测源码及入口、
完整ReasonAQA、MMAU/MMAR/MMSU、训练/模型源码均保持原字节，避免破坏运行身份。

七项固定别名：base、recursive_5_10_5、serial84、serial48_lora_r8、
serial84_lora_r8、recursive_mesh、mesh48_lora_r8。
targets.json完整绑定最初七项的精确run/训练route，均step17313、geometry=8,8,4，
runtime固定同run的formal/runtime.json；不自动选latest、不允许模型身份覆盖。
各路线复用原严格loader和原逐样本context，数值策略与既有sample5000版本保持一致。
复用data/reasonaqa/test_close_ended_sample5000_seed1234_v1/manifest.json，
不重新抽样；唯一指标仍是完整答案规范化精确匹配。

每项一个vc submit作业：pdgpu-3090/8GPU/64CPU/256G/1node。
复用八进程分片协议，导入Torch前屏蔽GPU、manifest索引mod8，每rank625题，
逐题fsync、断点恢复，精确完整5000题后才合并评分；任何rank失败不输出部分分母。
job为raqa7g-<最多40字符别名>-<12位摘要>，整体不超过60个ASCII字符。
输出在各自run的eval/reasonaqa_close_ended_sample5000_seed1234_step17313_8gpu_v1/，
与原30项run及旧完整七项评测目录不冲突。新增独立七项compare入口及CSV/JSON。

本地联合验证236项中221通过、15项因缺真实依赖跳过，PASS_LOCAL_WITH_SKIPS。
验证器tests/validate_reasonaqa5000_selected7_8gpu_eval.py及报告
audit_reports/reasonaqa5000_selected7_8gpu_eval_validation.json。
新16项回归覆盖七项精确绑定、与原30项互斥合计37项、原推理函数AST/协议一致、
8个真实OS进程运行实际worker（Torch/CUDA/模型明确模拟）、完整5000题合并/评分/比较、
rank中断13后只补612、坏raw/错覆盖/重复GPU/身份变化拒绝、
真实Bash入口与模拟vc七项提交及失败继续；同时运行既有评测回归并检查1104份旧源码SHA、
Python3.10与7份Bash语法。真实依赖缺失的原测试明确跳过，不能视为生产GPU通过。
本次仅在本地实现和合成验证，未读取远程生产资产、提交真实作业或commit/push。
完整目标表、同步和操作说明见新增包README.md。

代码经GitHub同步后，在远程仓库根补交这七项：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_reasonaqa5000_selected7_8gpu_3090.sh all --check-args
bash code/RSmol/run_reasonaqa5000_selected7_8gpu_3090.sh all
```

无需重新生成已有manifest。完成后：
bash code/RSmol/mizar_s1_ced_reasonaqa5000_selected7_8gpu_eval/run.sh compare
默认输出outputs/RSmol/reasonaqa5000_selected7_seed1234_8gpu_v1.csv和同名JSON。
原30项仍按原入口和比较命令运行。补交单项用上面的七项别名替换all。
本条仅前置，全部RM历史字节完整保留。

## 2026-11-16：ReasonAQA sample5000每目标8卡、pdgpu-3090独立评测

用户已撤下上一轮作业，批准按8卡计划实现并重新提供提交命令。
新增隔离包code/RSmol/mizar_s1_ced_reasonaqa5000_8gpu_eval/及入口
code/RSmol/run_reasonaqa5000_eval_8gpu_3090.sh；每目标一个pdgpu-3090作业，
8GPU/64CPU/256G/1node，原镜像、mellow_c8204d8、离线配置不变；all提交30项。
复用原固定targets.json和同一seed1234的5000条manifest，继续排除official_mizar与旧七项。
候选mesh_inject48继续CPU合同校验，失败记录并继续其余29项；可按原参数明确更正run。
原单卡/完整ReasonAQA、MMAU/MMAR/MMSU、训练和模型代码保持原字节。

父进程预检/认领独立输出后，以同一Python的torch.distributed.run启动单节点8进程，
standalone、nproc-per-node=8、max-restarts=0。推理不包装训练梯度DDP，不做逐题collective。
worker导入Torch前按LOCAL_RANK屏蔽为一张继承可见GPU，原cuda:0严格loader映射到各自卡；
同时保留原CUBLAS配置、seed1234、FP32、eval、TF32off、每卡batch1、fixed20/prefix383、
原题小写、greedy300/EOS、无cache完整重算、逐题真实context和唯一全答案精确匹配。

按manifest索引mod8无shuffle/补齐分片，每rank恰625题，独立raw/progress/模型审计/锁。
8rank均PASS后校验625各覆盖、完整记录/生成SHA、不同GPU与同一可见卡映射，再恢复原顺序；
必须覆盖原5000个ID且无重漏才评分。父进程合并前复核源码/模型/manifest/源JSON/音频元信息。
任何rank错误或torchrun非零均整体FAILED，不以部分分母输出正确率。
各rank逐题fsync；断点重提仅生成剩余题，已完成rank不加载模型；全部rank完成后重新汇总
不启动torchrun。完整坏行/错误身份拒绝，仅截断未完成末尾写。
默认新输出eval/reasonaqa_close_ended_sample5000_seed1234_step<step>_8gpu_v1/，
rank00至rank07在ranks/下，旧单卡目录不复用、不删除。
job采用raqa8g-<最多40字符别名>-<12位摘要>，整体至多60字符。
--check-args打印30项完整绑定及vc_args，确认3090/8GPU资源且不读资产或提交。

本地python -B tests/validate_reasonaqa5000_8gpu_eval.py：220项中205通过、15项因缺
真实Torch/音频/Parquet等依赖明确跳过，PASS_LOCAL_WITH_SKIPS。
14项新增回归覆盖实际8个OS进程运行worker（Torch/CUDA/模型明确模拟）、5000合并/精确计分/
比较/无模型重评分、rank3中断13后仅补612、坏raw/错覆盖/重复GPU/身份变化拒绝、
30项资源/别名及完整variant解析、60字符job、真实Bash入口和模拟vc及候选失败继续。
1090份原源码SHA保持，Python3.10、5份Bash语法和diff检查通过；完整说明见新包README。
报告audit_reports/reasonaqa5000_8gpu_eval_validation.json。
Windows沙箱阻止临时目录/Bash，获准在沙箱外运行本地合成回归；未读取生产资产、
提交真实GPU作业或commit/push，不登记远程8卡PASS。

代码经GitHub同步后，在远程仓库根重新完整提交：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_reasonaqa5000_eval_8gpu_3090.sh all --check-args
bash code/RSmol/run_reasonaqa5000_eval_8gpu_3090.sh all
```

已有5000条manifest直接复用，无需重新生成。完成后运行：
bash code/RSmol/mizar_s1_ced_reasonaqa5000_8gpu_eval/run.sh compare
默认输出outputs/RSmol/reasonaqa5000_remaining30_seed1234_8gpu_v1.csv和同名JSON。
本条仅前置，全部RM历史字节完整保留。

## 2026-11-15：修复ReasonAQA sample5000提交job名超60字符

用户已删除上一轮提交作业，报告部分vc submit的job参数超过60位。
此前仅截目标别名到36字符，完整时间戳/PID仍拼入job名，导致整体可能超限。
仅修改本隔离提交器的-j命名为raqa5k-<最多40字符别名>-<12位SHA256摘要>，
完整名称最多60个ASCII字符；摘要绑定完整批次tag与完整目标别名，
下划线转连字符；日志保留原完整批次标记/别名，资源继续pdgpu-4090。
同步README及已有提交断言，新增2项回归：30项实际命令在不同长度PID/tag下
均不超过60字符且互异；长目标相同可见前缀及不同批次仍可区分。
两项本地回归通过，Python3.10语法和diff检查通过；未提交真实作业或commit/push。
同步到远程后用bash code/RSmol/run_reasonaqa5000_eval_5090.sh all重新完整提交。
本条仅前置，全部RM历史字节完整保留。

## 2026-11-15：ReasonAQA sample5000评测改用pdgpu-4090

按用户要求，仅将mizar_s1_ced_reasonaqa5000_eval/submit.py中的vc submit分区改为
pdgpu-4090，同步README和既有提交测试期望。兼容原命令，入口文件名仍为
code/RSmol/run_reasonaqa5000_eval_5090.sh；实际30项目标均提交到4090。
核对30项实际command()参数均为4090，1GPU/8CPU/32G/1node及原镜像保持；
两份改动Python文件通过Python3.10语法检查。未提交真实作业或commit/push。
本条仅前置，全部RM历史字节保留。

## 2026-11-15：ReasonAQA test 固定随机5000条及其余30项checkpoint独立评测

新增隔离包 code/RSmol/mizar_s1_ced_reasonaqa5000_eval/，以及数据入口
generate_reasonaqa_test5000.sh、提交入口 run_reasonaqa5000_eval_5090.sh。
本地Windows仅改代码，GitHub中转；原始test.json、音频、权重及GPU推理仍在远程Linux。
不改原完整ReasonAQA七项及MMAU/MMAR/MMSU实现，也不改训练/模型代码。

候选池按用户确认严格核对55379条：AudioCaps-MCQ23672、Clotho-MCQ26031、
ClothoAQA标准答案yes/no5676；独立random.Random(1234)全池均匀无放回抽5000条，
再按原始行号排序，不分层、不去重题目，不改变原记录。所有目标共用保存的manifest，
保留源test SHA、算法、原行号/ID、原始记录及各类别实际入选数；每次启动复核源与抽样。
默认目录 data/reasonaqa/test_close_ended_sample5000_seed1234_v1/，生成manifest.json、
sampled_test.json、selected_records.jsonl、dataset_audit.json；幂等，拒绝异源覆盖。

目标固定为当前MMAU已适配38路减official_mizar和之前七项，共30项。
完整别名/variant/run/route/step/geometry见targets.json与README表格。
official_input_only、official_serial84_r8、mesh_x4使用34626步、8/4/4训练geometry，
其余17313步、8/8/4；评测始终单卡batch1，不自动找latest。
recursive_mesh_layer4_8_inject采用用户暂定run：
mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_5090/formal_20261030_172359752928500，
checkpoint为formal/checkpoints/model--step-17313.ckpt，runtime推定同run的formal/runtime.json。
提交该候选项前CPU读取并调用原训练route validator核对源码/合同/step/geometry/runtime；
错误则明确FAILED并继续其余29项，批量最终返回非零，不声称该路径已获远程核实。
提供inspect_checkpoint独立核对与候选run显式更正；CPU通过不代替GPU strict load。

沿用原fixed20、mono/32k、单文件20秒分半或双文件各10秒、原input小写/256token，
prefix383、FP32、eval、seed1234、TF32off、greedy300/EOS、无cache完整重算；
各架构通过原runtime传入真实音频/context和所需prompt mask，GT/caption不进入模型。
唯一指标保持完整答案规范化精确匹配（去空白/一个末尾EOS/casefold/合并空白）；
保留字母/正文/括号/标点，完整5000条后才评分，不跳题缩小分母。
原始生成逐题fsync，锁/身份保护、坏末尾恢复、进度重建、剩余题续跑与完整raw重评分；
逐题保存实际prompt tokens/mask/音频波形指纹，汇总严格比对共同数据/协议/代码/输入。
每run输出eval/reasonaqa_close_ended_sample5000_seed1234_step<step>_v1/。
提交器用原pdgpu-5090/1GPU/8CPU/32G/1node和镜像/conda，逐项保存submission报告。
新增脚本名避开原完整ReasonAQA的身份glob，不使旧结果仅因新增入口而失效。

远程同步代码后，仓库根执行：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/generate_reasonaqa_test5000.sh
bash code/RSmol/run_reasonaqa5000_eval_5090.sh all --check-args
bash code/RSmol/run_reasonaqa5000_eval_5090.sh all
```

30项完成后bash code/RSmol/mizar_s1_ced_reasonaqa5000_eval/run.sh compare，
默认CSV为outputs/RSmol/reasonaqa5000_remaining30_seed1234_v1.csv及同名JSON。
若改了候选run，比较时显式列出真实report-dir；失败单项按别名补交，避免重提整个all。

本地python -B tests/validate_reasonaqa5000_eval.py：204项中189通过，
15项因缺真实Torch/音频/Parquet等依赖明确跳过，PASS_LOCAL_WITH_SKIPS；
包含20项新回归：55379行池与固定5000抽样、完整5000流程/精确计分、
中断13条后续跑4987、清单/身份/坏记录拒绝、30路context及真实Bash模拟提交和失败继续。
1070份原源码SHA不变，旧数据/评分字节与数值策略AST保持，Python3.10/Bash检查通过。
报告audit_reports/reasonaqa5000_eval_validation.json，完整操作见隔离包README.md。
Windows临时目录受限，经批准在沙箱外跑本地合成验证；未读取生产权重、
生成真实采样集、提交GPU作业或commit/push，不登记远程PASS。
本条仅前置，全部RM历史字节完整保留。

## 2026-11-15：init0904 串行 CED8→4 双循环 LoRA r8 三项评测扩展

已将 mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8 接入共享 multilayer
评测注册及现有 MMAU/MMAR 提交器；MMSU 通过原动态注册表复用同一模型加载和解码。
variant=recursive_5_10_5_init0904_layer8_4_audio_reinject_loop_lora_r8，固定 run 为
mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_gbs256_5090/
formal_20261113_221451337688842，精确选择 model--step-17313.ckpt、同 run 的 runtime.json，
训练 geometry=8,8,4；其他中间 checkpoint 不参与选择。
实际导入隔离训练模型，保留 (A4,A8) 接口、input→CED8 / input→CED4 顺序、30次层执行、
双循环七投影 LoRA r8 和完整训练 state_dict strict=True 加载；不重置已训练 gate/LoRA。
生成/输入/评分函数不变：固定20秒、prefix383、FP32 batch1、无cache完整重算、argmax、
300token/EOS；MMAU test-mini与MMAR各1000条沿用小写题面和原官方scorer，MMSU5000条
沿用官方原大小写题面及现行 mmsu_answer_text_contains_v2 评分，原始输出独立保留。
历史SHA快照不改；新扩展引用记录精确新增块和前后SHA，内存剥离后继续原源码审计，
额外保护31份本路线训练源码/启动器；训练/模型/转换源码及生产合同SHA校验未修改。
本地回归175项：161通过、14项因缺少真实tensor/音频/Parquet依赖明确跳过；新增12项覆盖
三入口真实Bash模拟提交、严格加载/错误身份拒绝、注入/双LoRA执行hooks、1000+1000+5000
合成流程和续跑。5项新增真实小模型logits/最终权重测试已提供但因缺Torch/Transformers跳过。
报告 audit_reports/mizar_init0904_lora_eval_validation.json；完整说明及三条远程提交命令见
code/RSmol/scripts/README_mizar_init0904_lora_eval.md。Windows临时目录受沙箱限制，
经批准在沙箱外运行本地合成回归；未访问生产权重、提交GPU作业或commit/push。
本条仅前置，RM历史字节完整保留。

## 2026-11-13：纯三循环音频启动兼容文本checkpoint遗留的双循环冗余schedule字段

用户重提mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8音频训练后，write_config.py
报recursive_logical_to_physical_schedule contradicts the fixed schedule。原始5-10-5转换器写入四个
schedule字段，纯x3转换器deep-copy后只更新三个，漏掉recursive_logical_to_physical_schedule；
该冗余别名继承30次x2序列，真正执行使用的logical_to_physical及40/20/3模型/metadata均为三循环。
此前音频入口把全部别名一律要求相同，阻断了这个有效的已训练文本checkpoint。

仅修改本条隔离音频路线contract.py和models/recursive_text.py：
只允许recursive_logical_to_physical_schedule恰为原始0..4,(5..14)×2,15..19历史序列，且主字段
logical_to_physical必须完整为0..4,(5..14)×3,15..19；其他别名/架构/40逻辑20物理3循环/源层映射/
非MeSH/完整9244步来源检查保留。runtime.json的contract.recursive_text_checkpoint.
schedule_metadata_compatibility记录原值/规范值/原因及source_checkpoint_modified=false。
严格加载完整权重并校验实际20个独立物理模块和40次执行后，只在内存中将冗余别名规范到40次。
不改写远程源checkpoint/config/权重/训练状态，不重转或重训文本；shared文本模型和转换器保持原字节。
音频mapper/三组LoRA、input5/15/25和CED4/8入口5/15、数据/损失/更新/配方保持原样。

本地回归python -B tests/validate_mizar_s1_ced_text_x3_loop_lora.py：48项通过，7项真实tensor因缺少
Torch/Transformers明确跳过，报告PASS_LOCAL_WITH_SKIPS。新增4项身份回归，执行实际旧转换器→x3转换器
重现残留字段、只兼容已知别名/仍拒绝真实x2和其他错别名、源文件不修改、实际模型审核后仅内存规范；
原shared-start的write_config合成链路改用真实历史形态，runtime.json生成并记录兼容审计。
可选真实strict loader tensor检查追加历史别名/全部原权重一致/源config字节不变/内存规范断言。
36份母本/共享SHA和16份继承源码保持，完整train() AST对照通过；Python3.10/shell/README语法通过。
更新reference_audit.json及audit_reports/mizar_s1_ced_text_x3_loop_lora_validation.json。
Windows临时目录/Bash受沙箱限制，经批准沙箱外运行本地合成回归；未访问生产权重、提交GPU作业或commit/push。

失败在write_config.py、torchrun前，因此没有该run的音频更新或checkpoint；同步修复后直接重提，
沿用原文本checkpoint和原训练配方，使用默认新时间戳目录，不能将失败run作为resume或复用非空output-dir：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8_5090.sh --stage-workers 4
```

完整说明和精确Git提交命令见该隔离路线README。本条仅前置，全部RM旧字节保留。

## 2026-11-13：init0904 CED8→4双LoRA音频作业中断及fresh重启目录保护诊断

用户报告run_mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_5090.sh
日志仅有fresh formal directory is not empty，目标run为
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_gbs256_5090/formal_20261113_144012025494116。
只读检查本地launcher/run.sh/train.py后，请用户提供远程目录和metrics证据。

用户实际目录包含runtime/config_audit、全部8rank的first/opened_gate梯度审计文件、metrics和staging报告，
说明曾进入真实八卡训练，而不是首次启动即失败。用户22:11:14和22:11:44两次读取均为step250，
loss1.00848484、gradient_l2_before_clip0.88760376、gradient_common_scale1、GradScaler65536，
staging_complete=true、audio_read_mode=local、local_switch_step17，
实际串行轨迹input→CED8/input→CED4及双LoRA轮次符合预期。checkpoints目录total0。
以上仅确认训练曾推进到250步，不表示完成17313步或整体远程PASS。

用户补充集群部分卡掉线、训练运行约40分钟后回到Pending。最可能是平台重启了原fresh命令，
复用同一个run root，随后计算入口的非空目录保护拒绝再次从step0覆盖既有产物。
这是结合用户平台现象作出的推断，尚无平台重启事件记录，不登记为已验证根因。
原代码首次保存点为2886，当前无完整checkpoint，metrics不能代替训练状态，无法从250完整恢复。

建议先在平台终止旧Pending/重试作业，再沿用原入口提交新的自动时间戳run，不传旧--output-dir或--resume-checkpoint：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/run_mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_5090.sh --stage-workers 4
```

重新严格继承原指定formal-maxlr8e-4-resume4500-20260904_154711/checkpoint-step-009244文本权重，
音频从step0新建所有训练状态；保持全部模型/训练配方及源码身份。
本次不修改训练实现、不放宽空目录保护、不删除远程旧目录；未代为取消或提交作业、commit/push。
本条仅前置，全部RM历史字节完整保留。

## 2026-11-13：六槽三循环MeSH改用指定双循环已训练routers，干净SmolLM2主干重新训练

用户报告run_stage4_5_10x3_5_mesh_6slot_formal_3090.sh首步nonfinite gradient norm，批准更换初始化方案。
只扩展本路线转换.py/.sh及训练入口的初始化来源校验/实际加载核验/报告；模型结构、MeSH记忆与query、
三循环执行、数据/损失/梯度归一化/裁剪/优化器/调度/保存恢复计算均保持原样。
本地Windows维护核心代码、GitHub中转，权重/数据/输出和GPU训练仍在远程Linux。

文本主干/tokenizer仍来自干净SmolLM2，默认models/SmolLM2-5-10-5原始权重抽层产物，也支持原始30层
models/SmolLM2固定抽层。独立router来源默认固定为：
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh_6slot/formal_1epoch_fp32_20261031_124955519529617/checkpoint-009244。
新增--router-checkpoint和RSMOL_5_10X3_5_MESH_6SLOT_ROUTER_CHECKPOINT，与主干来源分开校验。
源x2仍为六槽/三组/6模块/12参数张量，目标x3为六槽/四组/8模块/16参数张量/27696参数。
按执行顺序复制为w_pre,r_pre,w0,r_pre,w0,r0,w1,r1；write索引[0,1,1,2]，read索引[0,0,1,2]。
weight/bias一起copy到独立目标Parameters，重复来源不绑定/冻结，后续全部训练；memory每次forward新建。

新增code/RSmol/text_5_10x3_5_mesh_6slot/router_transfer.py：验证源架构/FP32训练合同、9244步完整
metadata/manifest/complete文件；只从safetensors materialize所需12个router，不加载源LM/tokenizer或
training_state.pt。检查形状/dtype/finite/缺失重复额外键；无随机回退。
转换产物固定FP32，避免BF16主干源导致FP32 router降精度；主干继承逐元素核验、router迁移逐元素与SHA核验、
目标存储独立、40次执行、保存重载tensor/logits和tokenizer一致后原子发布。config与mesh_conversion_metadata
记录双来源、文件与router SHA及精确映射；不再声称fresh_router_tensors随机初始化。
训练fresh校验拒绝旧随机产物，严格from_pretrained后再次核对16个实际router值、FP32、独立存储及可训练性。
完整resume沿用checkpoint实际参数，不用初始donor覆盖更新后的router；报告保留初始化来源。

本地提交推送后，远程登录节点重新转换并提交新训练（旧三循环转换目录已由用户删除）：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/scripts/convert_stepwise_5_10x3_5_mesh_6slot.sh &&
  RSMOL_5_10X3_5_MESH_6SLOT_FORMAL_RESUME_FROM='' bash code/RSmol/run_stage4_5_10x3_5_mesh_6slot_formal_3090.sh
```

转换CPU8线程/离线；产物仍models/SmolLM2-5-10x3-5-mesh-6slot，不覆盖已有目录。
新时间戳训练输出保持outputs/RSmol/stage4_5_10x3_5_mesh_6slot/formal_1epoch_bf16_<timestamp>/，
从step0新建optimizer/scheduler/RNG/数据游标；GBS1024=8×micro4×GA32，context1024，1epoch/9244步、
463warmup、max/minLR8e-4/8e-5，FP32参数/状态、BF16 autocast、FP32 CE、无GradScaler、TF32关闭，
原pdgpu-3090/8GPU/64CPU/256G/1node及rsmol环境不变；不新增smoke前置。
完整转换/训练/恢复及精确Git提交命令见code/RSmol/text_5_10x3_5_mesh_6slot/README.md。

python -B tests/validate_text_x3_routes.py --output audit_reports/text_x3_mesh6_router_transfer_validation.json：
46项中36项本地合成回归通过，10项真实Torch/Transformers/safetensors/tokenizers CPU检查因缺依赖明确跳过，
报告PASS_LOCAL_WITH_SKIPS。包含10项新增router迁移回归及原两条路线回归：映射与数值、独立存储/别名和冻结
拒绝、仅获取12个tensor/不读训练状态、错误来源/缺失重复/非有限/dtype拒绝、Python默认与覆盖、
Bash双来源参数/空格路径和提交几何、实际初始化元数据、原stream/LR/梯度更新/优化器/恢复合同。
28份母本/模型/纯三循环/双循环/提交源码LF摘要不变，记录router_transfer_source_guard.json；
Python3.10语法、相关shell及README命令语法通过。Windows沙箱临时目录/Bash限制，经批准沙箱外验证。
未改动工作区其他Huginn路线已有编辑，Git命令只纳入本次文件；未代为commit/push、读取生产权重或提交GPU作业。
保留nonfinite gradient norm硬检查；不宣称新router初始化已解决真实首步报错或完成远程训练。
本条仅前置，全部RM旧字节保留。

## 2026-11-13：原地更新 Huginn 音频训练，增加循环前后共享 RMSNorm 与失败诊断

用户报告音频作业在train.py的nonfinite/emergency gradient norm保护处退出。
该检查同时涵盖非有限范数、有限范数达到1e6、异常裁剪系数，现有日志不足以证明梯度本身为inf。
用户明确要求不再复制隔离，直接修改现有Huginn音频实现；本轮保留原入口、route、输出命名与文本来源。
本地Windows只维护代码，GitHub同步，生产权重/数据及GPU执行均在远程Linux。

原入口仍为code/RSmol/run_mizar_s1_ced_small_huginn_text_epoch1_5090.sh，
实现/运行/提交/恢复命令见
code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_huginn_text_epoch1/README.md。
默认仍严格从
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_huginn_gbs1024/epoch1/checkpoint-step-009244
开始全新音频训练，不读文本training_state.pt，不使用旧音频checkpoint恢复。

先显式from_pretrained严格加载完整原文本LM含训练后的A，此时不注册新Norm槽位，
missing/unexpected/mismatched/error仍全部拒绝；校验后新增两个LlamaRMSNorm(576)，再校验新模块。
recurrent_input_norm位于每次共享A后、Core十层前；recurrent_output_norm位于每次Core后、下一次A或Coda前。
两个Norm相互独立且各自在两轮共享，沿hidden维作用于全部683位置，weight全1、无bias、
epsilon沿用文本rms_norm_eps，新增1152参数。注册不消耗RNG、不改变已加载基础LM/A/绑定embedding/head；
新增Norm使用LM maxLR3e-4及低维WD0，全部LM/A/新Norm均全量训练。
E=Prelude(X)，S0仍每forward一次±3σ截断高斯/σ√(2/5)，
Si=Norm_out(Core(Norm_in(A(cat([Si-1,E])))))，i=1,2；S2直接进入原Coda/FinalNorm/head。
20物理层/30执行、同一A两次、相同E、不detach完整反传均保持。
全1Norm并非恒等映射，音频阶段从文本权重适配新增归一化；原共享文本模型及转换/训练路线不改。

contract/runtime/config_audit登记audio_boundary_normalization；完整音频state_dict保存新Norm，
HF normalized config也支持重建槽位。源码/合同哈希改变，严格拒绝旧的无边界Norm音频checkpoint完整恢复；
README恢复命令只适用于本次修改后、相同源码/资产/数据/配方的完整音频checkpoint。

新增gradient_diagnostics.py及失败前诊断，不改变原GradNormTracker算术/裁剪结果/1e6阈值或GradScaler。
正常步不做FP64扫描；异常时在unscale后、裁剪前检查逐参数有限性/FP64 L2/最大值，
保留前20项及非有限参数清单，写formal/gradient_failure_rank<N>_step<XXXXXX>.json，
记录rank/待更新步/已完成步/loss/LR/scaler/样本，区分真实非有限梯度、FP32范数溢出及有限超阈值。
首次梯度审计失败也落诊断；仍在optimizer更新前集体退出。JSON用字符串表示inf/nan，不输出非标准数值。

原CED/mapper/decoder及全部data/index/staging逐字节不变。原train.py完整AST去除新增审计/诊断后
与母本一致；optim.py去除失败诊断后保持原算术、裁剪状态更新和优化器分组；
音频Huginn去除三个Norm挂载/边界扩展后与原共享文本模型AST一致，audit_reference.py验证以上受限差异。
35份原始参考源码SHA继续通过。保持此前修好的mizar_s1_gbs256_临时目录前缀。
全部原音频配方不改：hybrid1477492行/fixed20/冻结eval CED/新mapper/683位置，
answer-only rank-local token mean→GA4→DDP均值，8×micro8×GA4=GBS256，3epochs/17313步/866warmup，
LM3e-4/mapper6e-4、AdamW(.9,.999)/eps1e-8/WD.01、LambdaLR floor.1、GradNormTracker、
FP32无autocast/TF32、保留GradScaler、seed1234及rank1234+rank；原资源/环境/共享音频后台复制均保持。

本地提交推送后，远程登录节点执行：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_huginn_text_epoch1_5090.sh --stage-workers 4
```
使用默认新时间戳输出，不传旧音频--resume-checkpoint，原CED资产及CPU index准备后直接formal。

python -B tests/validate_mizar_s1_ced_huginn_text_epoch1.py：51项本地通过、9项真实CPU tensor缺依赖跳过；
报告audit_reports/mizar_s1_ced_huginn_text_epoch1_validation.json。
新增NumPy执行真实模型循环AST/全部683位置/两轮Norm顺序/归一化状态回注，
真实安装/严格加载顺序/新旧恢复身份/诊断写盘CPU fixture，以及真实GradNormTracker受控数值对照。
可选Torch用例补齐新Norm反传及LR/WD/更新、完整checkpoint下一随机步往返、
HF原文本全部tensor继承与训练A保留/缺A拒绝、Norm注册RNG不变/幂等、normalized HF保存重载。
NumPy/fixture不代表Torch autograd验证；Windows沙箱临时目录及Bash限制下经授权沙箱外执行同一检查。
未代为commit/push、访问远程资产或提交真实GPU作业，不登记稳定训练或8rank恢复PASS。
本条前置，原RM字节完整保留；其他并发路线改动不触碰。

## 2026-11-13：纯三循环文本 checkpoint 初始化 CED4→8／三轮 input 重注入／三组 LoRA r8 音频训练

按用户批准计划，完整复制双循环 layer4_8_inject_audio_reinject_loop_lora_r8 音频母本后隔离扩展。
新入口 code/RSmol/run_mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8_5090.sh；
实现、Git提交推送、正式训练及完整恢复命令见
code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8/README.md。
继续本地Windows维护核心代码、GitHub中转；生产权重/数据/输出及GPU执行均在远程Linux。

默认文本来源固定为：
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x3_5/formal_1epoch_bf16_20261109_153922858684473/checkpoint-009244。
无需重新转换；严格加载纯非MeSH的全部LM权重和同目录tokenizer，20物理层/40次执行，
0..4,(5..14)×3,15..19，保留原抽层映射、hidden576及embedding/head绑定。
校验config和metadata/manifest/complete的纯三循环身份及最终9244步；不反序列化文本training_state.pt。
音频mapper、注入模块、三组LoRA及optimizer/scheduler/GradScaler/GradNorm/RNG/数据游标从step0新建。

CED-Small冻结eval，每批一次encoder提取final与完整block4/8 norm前输出，原freq_merge_126 mapper和
两个独立CED MLP/gate保持。原mapped E作为126音频token输入；每次完整LM forward一次共享audio_linear(E)，
不detach。逻辑索引5：共享input残差→CED4；索引15：共享input残差→CED8；索引25：仅共享input残差。
两个CED gate读取post-input hidden；所有音频残差仅直接作用前126位置。
仅物理层5..14七投影q/k/v/o及gate/up/down增加三套独立A/B，按(logical_index-5)//10选组；
r8/alpha16/scale2/dropout0，独立CPU RNG、A Kaiming/B零、基础W/bias共享且全量训练；前后五层无LoRA。
三组LoRA为420张量/2442240参数，串行音频分支保持2777859，两者合计5220099，mapper另计。
无KV，关闭decoder gradient checkpointing，保留每次调用/异常后的轮次清理，梯度审计覆盖三组A/B。

原hybrid1477492行及fixed20，126audio+SEP1+prompt256+answer300=683，PAD!/17、EOS/SEP0，maskNone；
answer-only rank-local microbatch token mean→GA4均值→DDP均值。8GPU×micro8×GA4=GBS256，
3epochs/每epoch5771/总17313/866warmup；LM maxLR3e-4，mapper/音频分支/三组LoRA maxLR6e-4；
原AdamW(.9,.999)/eps1e-8、矩阵WD.01/bias低维WD0、LambdaLR floor.1、GradNormTracker保持。
FP32无autocast/TF32、保留GradScaler，seed1234和1234+rank；共享音频立即训练、4workers后台复制，
optimizer边界全rank共同切换，producer停止reap后清理。原mellow_c8204d8/镜像、pdgpu-5090、
32CPU/256G/8GPU/1node；原CED资产检查和CPU索引后直接formal，无新增smoke前置。

独立route ced_small_mizar_s1_fixed20_gbs256_text_x3_layer4_8_audio_reinject_loop_lora_r8_v1；
默认输出outputs/RSmol/mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8_gbs256_5090/formal_<timestamp>/，
实际训练再追加formal/，最终formal/checkpoints/model--step-17313.ckpt及model--epo-3.ckpt。
日志/job/index-cache/tmpfs及full-resume身份隔离；tmpfs保留原mizar_s1_gbs256_守卫前缀。
完整音频恢复保留全部LM/三组LoRA/音频参数与训练状态、八rank RNG和数据游标，只接受本新路线
同源码/文本来源/资产/数据/依赖/配方并沿用原run root。

本地提交推送GitHub后，远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8_5090.sh --stage-workers 4
```

python -B tests/validate_mizar_s1_ced_text_x3_loop_lora.py：44项本地检查通过，7项真实CPU tensor因缺少
Torch/Transformers明确跳过，报告PASS_LOCAL_WITH_SKIPS。覆盖36份母本/共享SHA、16份逐字节继承源码、
完整train() AST和loss/更新/保存恢复语义、17314个LR位置、实际纯三循环转换config兼容、最终checkpoint
身份/错误来源拒绝、step0新状态和完整恢复rank3 RNG/游标、非零三组LoRA数值与第三轮错误轨迹拒绝、
真实Bash配合假vc及tmpfs守卫、并发复制/集体切换/失败清理。Python按远程3.10语法检查，新2个shell及
README Bash命令语法通过。报告audit_reports/mizar_s1_ced_text_x3_loop_lora_validation.json。
source_guard.json保护原文件，reference_audit.json为本路线当前审计，parent_reference_audit.json仅为母本历史。
Windows沙箱拒绝临时目录/Bash signal pipe后，经批准沙箱外运行同一合成验证；可手动加--require-tensors，
不读生产资产/不用CUDA/不提交作业，不作formal前置。未代为commit/push、访问远程资产或提交真实GPU训练，
不声明生产权重加载/八卡训练/八rank恢复PASS。现有评测尚未登记本新路线合同，后续另行适配。
本条仅前置，全部旧RM字节保留。

## 2026-11-13：init0904 文本 checkpoint 初始化 CED8→4／input重注入／双LoRA r8 音频训练

按用户确认计划完整复制指定非MeSH CED8→4串行音频注入和双LoRA r8母本。
新入口 code/RSmol/run_mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_5090.sh；
实现、Git同步、正式提交及完整恢复命令见
code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8/README.md。
继续本地Windows维护核心代码、GitHub中转，生产权重/数据/输出/GPU执行均在远程Linux。

默认文本来源固定为：
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-maxlr8e-4-resume4500-20260904_154711/checkpoint-step-009244。
无需重新转换；显式RecursiveLlamaForCausalLM严格加载完整非MeSH LM和同目录tokenizer，
保留hidden576、20物理层/30次执行、0..14,5..14,15..19及原抽层映射和embedding/head绑定。
先加载文本权重再注册新音频模块/LoRA，不反序列化文本training_state.pt；
音频optimizer/scheduler/GradScaler/GradNormTracker/各rank RNG/数据游标从step0新建。

CED-small由原models/CED_small严格加载、冻结eval，每批一次编码取final与完整block4/8 norm前输出。
原freq_merge_126 mapper新初始化；E仍作为126音频token输入，每完整LM forward一次共享audio_linear(E)，不detach。
逻辑索引5：共享input残差→CED8残差；索引15：同一input残差→CED4残差；CED gate读取post-input hidden。
残差直接作用前126位置，原两个独立CED MLP/gate和共享input投影/gate保持，三个gate零初始化。
仅物理层5..14七投影q/k/v/o、gate/up/down新增两套独立A/B，r8/alpha16/scale2/dropout0，
A Kaiming/B零；基础W/bias共享且全量训练，前后五层无LoRA。保留独立CPU RNG初始化、
每层/异常清理轮次、无KV和禁用decoder gradient checkpointing；首步B和更新后A/B任务梯度审计保持。
LoRA新增1628160参数，串行音频分支2777859，相对普通CED母本合计4406019。

train.py、optim.py、write_config.py及全部model/LoRA/CED/data/index/staging共24份运行源码逐字节继承。
contract.py只改route/默认文本路径，run.sh只改入口提示/tmpfs名称，launcher只改来源及隔离命名。
独立route ced_small_mizar_s1_fixed20_gbs256_init0904_layer8_4_audio_reinject_loop_lora_r8_v1；
代码/output/log/job/index-cache/tmpfs/full-resume身份隔离。source_guard.json保护69份母本、
共享模型和原普通init0904路线文件；本次reference_audit.json为新路线源码审计，
parent_reference_audit.json仅为母本历史审计；目录.gitignore明确放行三份JSON。
tmpfs使用/dev/shm/mizar_s1_gbs256_ced_init0904_in84_loop_lora_r8_<token>，
保留原守卫前缀，回归直接执行shell实际赋值及原validate_stage_root/cleanup和owner/token检查。

原hybrid1477492行、fixed20、126audio+SEP1+prompt256+answer300=683token、PAD!/17、EOS/SEP0、
attention_mask=None及answer-only rank-local token mean→GA4均值→DDP均值保持。
8GPU×micro8×GA4=GBS256；3epochs/5771updates每epoch/17313total/866warmup；
LM maxLR3e-4，mapper/全部音频分支/双LoRA maxLR6e-4；原AdamW(.9,.999)/eps1e-8、
矩阵WD.01/bias低维WD0、LambdaLR floor.1、GradNormTracker、FP32无autocast/TF32及GradScaler保持。
共享音频立即训练、默认4workers后台复制、optimizer边界全rank共同切换、先reap再清理保持。
原mellow_c8204d8、原镜像、pdgpu-5090/32CPU/256G/8GPU/1node，原CED资产检查和CPU索引后直接formal，无新增smoke前置。

本地提交推送GitHub后，远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_5090.sh --stage-workers 4
```

默认输出outputs/RSmol/mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8_gbs256_5090/formal_<timestamp>/，
沿母本再追加formal/；最终formal/checkpoints/model--step-17313.ckpt及model--epo-3.ckpt。
完整音频恢复包括LM、双LoRA、音频分支及所有训练状态/八rank RNG/游标，不重置已训练参数，
只接受本路线同源码/资产/数据/依赖/配方并沿用原run root；拒绝其他路线的full resume。

python -B tests/validate_mizar_s1_ced_init0904_loop_lora.py：
46项本地检查通过，7项真实CPU tensor因缺Torch/Transformers明确跳过，报告PASS_LOCAL_WITH_SKIPS。
覆盖69份参考SHA、完整源码差异白名单和semantic配方、17314个LR位置、实际step0/全状态恢复与rank3 RNG选择、
严格加载和错身份拒绝、非零双LoRA和post-input串行数值/有限差分/梯度失败审计、
实际并发复制/共享切换/失败清理、真实Bash配合假vc/torchrun及真实路径守卫、Git审计文件可提交性。
源码按远程Python3.10语法检查；报告audit_reports/mizar_s1_ced_init0904_loop_lora_validation.json。
Windows沙箱限制临时目录及Bash signal pipe后，经批准在沙箱外运行同一合成验证通过。
可手动加--require-tensors，合成CPU模型、不读生产资产/不跑CUDA/不提交作业，不作formal前置。
未代为commit/push、访问远程资产或提交真实GPU作业，不登记真实权重加载/八卡训练/真实8rank恢复PASS。
现有评测尚未登记本新路线合同，后续单独接入；原训练及评测均未修改。本条仅前置，全部旧RM字节保留。

## 2026-11-13：指定MeSH 4→8注入/Loop-LoRA r8的MMSU固定目标入口

按用户给定checkpoint扩展MMSU提交，精确绑定
mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_loop_lora_r8_5090/
formal_20261104_161727489909082/formal/checkpoints/model--step-17313.ckpt，
配套同run的formal/runtime.json，expected step17313和训练geometry8,8,4。
核对原MMAU/MMAR分支与共享multilayer registry，variant为
recursive_mesh_layer4_8_inject_loop_lora_r8，原MeSH五槽/两循环/4→8注入与Loop-LoRA r8
适配已完整存在；MMSU通用入口原本已支持，无需改loader、模型或评测核心。

新增code/RSmol/run_mizar_mmsu_mesh48_lora_r8_5090.sh，固定模型身份，仅允许数据/scorer/
输出路径及Parquet I/O batch-size覆盖；支持--check-args纯参数检查。
提交原通用MMSU入口，完整5000题，pdgpu-5090/1GPU/8CPU/32G/1node，原镜像和conda，
保留音频fixed20/32k、FP32、prompt256、greedy300、无KV及官方题面。
沿用已确认mmsu_answer_text_contains_v2，经原官方scorer汇总，完整分母5000。
默认结果写本run的eval/mmsu_step17313/，原始生成/评分副本/解析审计均按现有MMSU链路保存。
原五目标快捷入口和离线all保持原样。说明、远程提交与本地Git同步见
code/RSmol/mizar_s1_ced_mmsu_eval/MESH48_LORA_R8.md。

python -B -m unittest discover -s tests -p test_mizar_mmsu_mesh48_lora_r8.py -v：
7项全部通过，无跳过，执行真实新入口/原通用提交/计算shell和实际参数解析，
仅vc/conda为显式fixture，核验精确身份、资源、空格路径、只解析不提交、错误参数拦截。
132份现有评测文件摘要与105份训练源码LF摘要保持；新shell及Python3.10语法通过。
报告audit_reports/mizar_mmsu_mesh48_lora_r8_validation.json。
经授权在Windows沙箱外运行本地合成检查，未访问远程权重或提交真实GPU作业，
未代为commit/push，不声明远程推理PASS。本条仅前置，全部RM历史字节完整保留。

## 2026-11-13：MMSU完整答案文本包含评分及五份已有生成的CPU离线重评

用户确认忽略生成的选项字母，只检查归一化后生成中是否包含完整answer_gt文本。
评分版本mmsu_answer_text_contains_v2：去首尾空白、末尾一个EOS和开头选项格式标记，
对生成/GT做Unicode casefold及连续空白合并，保留标点，并对答案两端词字符检查边界。
完整文本命中才从原始选项元数据找到精确等于GT的第一个字母，作为官方格式编码；
生成字母不参与判分。未命中/空回答/只有字母但无答案文本，向官方scorer写空字符串计错。
解释、否定句或多候选中出现GT也会命中，这是用户选定的文本包含指标，不是选项字母准确率。

新增MMSU parsing与rescore CPU链路；后续evaluate自动使用同一转换，原始生成/token IDs及
predictions_official.jsonl不改，另存predictions_normalized.jsonl、response_parsing.jsonl、
parsing_report.json。官方scoring.py与外部mmsu_evaluation.py原样，固定LF SHA
b6d5d35d2755244019a78e601495ea74e722681bf6f7e1fd0179bab7cabca3eb；
执行官方CLI和原函数并核对分母5000、总体正确率等于文本命中数/5000。
音频/官方题面/FP32/prompt256/greedy300及已有模型loader、训练、MMAU/MMAR设置均保持。

离线入口code/RSmol/rescore_mizar_mmsu_selected5.sh支持official_mizar、base、
recursive_5_10_5、serial84_lora_r8、recursive_mesh和all。五个源目录精确复用原提交预设，
all会尝试全部目标，某份失败仍处理其余份。直接CPU运行，不用vc/GPU，不读权重/音频/Parquet。
优先校验完整raw的5000条连续行号、唯一ID、元数据与record/generation SHA，
有原预测时逐行交叉核验；仅raw不存在时允许预测文件回退，损坏raw不绕过。
不截断/修补源文件，不覆盖旧报告；新结果分别写各源目录的rescore_mmsu_answer_text_v2/。
保留生成进程锁与严格源码/来源/协议身份。同版本可重复评分；旧v1完整结果用离线入口，
未完成旧生成先沿用原代码完成再重评，不放宽断点身份。完整操作与Git同步命令见
code/RSmol/mizar_s1_ced_mmsu_eval/RESCORE.md。

本地验证python -B tests/validate_mizar_s1_ced_reasonaqa_eval.py --report
audit_reports/mizar_mmsu_answer_text_v2_validation.json：180项中170通过、10项因缺少
Torch/音频/Parquet依赖明确跳过，报告PASS。含16项新回归：真实官方scorer/5000条合成
结果、文本包含/错误字母忽略/分母、离线重复/源文件保留/损坏拒绝、五路径/真实Bash CPU
入口/无模型导入及未来完整生成流程；原MMAU/MMAR/MMSU/ReasonAQA回归通过。
10份相关Python源码符合Python3.10语法，新2个shell语法通过。保留ReasonAQA原132份
历史snapshot，normalization_reference.json单列4份授权MMSU文件前后SHA，其余128份
字节一致。旧MMSU validator恢复原字节，新测试接入综合validator，不改历史基线。
Windows沙箱外经授权运行本地合成测试；未读取远程资产、代为commit/push或提交真实作业，
不声明五个远程模型评分已完成。本条仅前置，原RM全部历史字节保留。

## 2026-11-13：独立 ReasonAQA test close-ended 评测及七目标提交

按用户确认范围，仅纳入 AudioCaps-MCQ.json、Clotho-MCQ.json 全部行，以及 ClothoAQA.json
中 strip/casefold 后标准答案严格为 yes/no 的行。其余行统计排除数量，保留原 JSON 顺序和
原始行号，不去重、不跳过失败样本。唯一指标为完整答案规范化精确匹配：去首尾空白、
移除末尾一个 <|endoftext|>、Unicode casefold、连续空白合并；保留字母、括号、正文和标点。
不抽取字母、不作子串/语义匹配，空回答算错且保留分母。输出 overall 微平均及三类别统计。

新增隔离包 code/RSmol/mizar_s1_ced_reasonaqa_eval/，默认读取
/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test.json，音频按项目原有 AudioCaps/Clotho/
ClothoAQA 根映射，支持反斜杠、空格、ClothoAQA 唯一规范化 stem；缺失/歧义/越界均失败。
只把原 input 小写后送入训练 tokenizer，不追加指令、不把 caption/GT/task 元数据送入模型。
使用原训练 mono/resample32k/fixed20，单文件整段20秒后分10秒，双文件分别10秒；保存实际
音频波形SHA与prompt token IDs指纹。FP32/eval、seed1234、TF32关闭、prompt256/prefix383、
纯greedy300/完整重算/无KV/LM mask=None与原CED协议一致；只调用已有严格loader和decoder，
保留递归/MeSH/注入/reinjection/Loop-LoRA每样本context，不修改训练及模型源码。

七个固定目标为 base、recursive_5_10_5、serial84、serial48_lora_r8、serial84_lora_r8、
recursive_mesh、mesh48_lora_r8；精确绑定用户所列七个run的model--step-17313.ckpt及配套
formal/runtime.json，训练geometry8,8,4，不自动选择latest。提交入口
code/RSmol/run_mizar_reasonaqa_selected7_5090.sh支持单目标和all七个独立完整作业；通用入口
run_mizar_s1_ced_small_reasonaqa_eval_5090.sh。每作业pdgpu-5090/1GPU/8CPU/32G/1node，
沿用原镜像、mellow_c8204d8及离线环境，不加自动smoke。--check-args只解析、不提交或读资产。

默认各自run下eval/reasonaqa_close_ended_step17313_v1/，输出完整raw generations/token IDs、
筛选/路径/模型审计、逐题正确性、accuracy及evaluation_report。checkpoint/runtime/test/
音频stat/源码/协议严格绑定；进程锁、append+fsync、raw权威恢复，只截断未完成末尾写入，
完整损坏记录硬失败，身份改变不覆盖原报告。完整raw重评分不加载模型。
compare模块对七个完成结果验证PASS、相同test/筛选/协议/评测源码、完整覆盖、原始hash、
重算分数及逐题实际模型输入指纹后输出CSV/JSON。详细协议、七条/一键提交、Git同步和
续跑/汇总命令见code/RSmol/mizar_s1_ced_reasonaqa_eval/README.md。

python -B tests/validate_mizar_s1_ced_reasonaqa_eval.py：164项中154通过、10项因本地缺少
Torch/音频/Parquet等依赖明确跳过，报告PASS；含22项新ReasonAQA测试（21通过/1跳过）
及原MMAU/MMAR/MMSU回归。验证真实Bash入口+模拟vc/conda、七目标精确路径/资源/引用、
筛选/完整答案计分/数值预处理fixture/题面隔离/断点恢复/损坏拦截/比较一致性；3个shell语法
与Python语法通过。132份原评测文件字节SHA保持，snapshot记录在新包reference_audit.json。
报告audit_reports/mizar_s1_ced_reasonaqa_eval_validation.json。Windows沙箱临时目录受限，
经授权在沙箱外运行本地合成回归；未访问远程生产权重或数据，未代为commit/push或提交GPU作业，
不声明远程推理PASS。本条仅前置，全部RM历史字节保留。

## 2026-11-13：四个指定新 checkpoint 的 MMAU / MMAR / MMSU 评测适配

按用户批准计划扩展 init0904、普通 text_epoch1、text_epoch1 CED4→8/audio reinjection/双 LoRA r8、
以及继承文本 epoch1 已训练双 LoRA 的 ced48 四个明确音频 checkpoint，全部 step17313、训练 geometry8,8,4。
普通两条加入 mizar_s1_ced_recursive_eval_runtime.py；后两条加入 multilayer runtime，分别绑定自己的
训练 route、目录、run 和文本合同。第四条使用其原训练构造器严格继承已训练 A/B，最终仍以完整音频
checkpoint state_dict 为准，不重置 LoRA、不用文本初始化覆盖最终权重；报告区分2777859个新音频参数
与1628160个继承的 LoRA 参数。严格合同、源码/资产摘要、实际类及执行顺序校验全部保留。

扩展原 MMAU/MMAR 5090 提交入口的新分支；MMSU 自动复用共享 registry，题面/音频处理/FP32/
prompt256/greedy300/原始response/官方scorer和完整1000/1000/5000题口径保持。
新增 code/RSmol/run_mizar_selected4_benchmarks_5090.sh，四个 TARGET 为 init0904、text_epoch1、
text_epoch1_serial48_lora_r8、text_lora_epoch1_serial48_r8；BENCHMARK 为 mmau/mmar/mmsu/all。
all 各提交三个独立完整作业，共12项，某次提交失败仍尝试剩余项；模型身份固定并覆盖残留模型环境变量。
--check-args 仅打印有效命令而不提交或读取生产资产；单项允许数据/评分器/输出路径覆盖。
输出分别为各自 run 下 eval/mmau_test_mini_step17313、eval/mmar_step17313、eval/mmsu_step17313。
原五目标 MMSU 预设保留。完整映射、四条批量/十二条单项提交、本地Git同步命令见
code/RSmol/mizar_selected4_eval/README.md。

保存105份训练源码摘要，训练目录和模型源码未修改。原28份MMSU历史母本SHA基线保持原样，
新增 reference_audit 显式登记4份获准评测扩展的前后摘要；原默认 shell 分支、registry/constants、
17个runtime函数及原模型审计主体/字段另做源码/AST保护。旧测试的路线数量和获准摘要断言同步更新。
共享 adapter 摘要随扩展改变，已有未完成旧评测应沿用原代码版本续跑，不放宽断点身份或混合预测。

python -B tests/validate_mizar_s1_ced_mmsu_eval.py --report audit_reports/mizar_selected4_eval_validation.json：
142项中133通过、9项因本地缺少Torch/音频/Parquet依赖明确跳过，报告PASS。
含十二项真实Bash入口/模拟vc及实际参数解析、错误路线/合同拦截、完整state严格加载先于冻结、
继承参数报告、物理schedule/串行4→8/双LoRA hooks、旧路线和官方scorer/断点回归；3个shell语法通过。
新增真实tiny CPU final-audio LoRA加载及logits等值测试，因缺Torch等依赖明确跳过，不声明已执行。
Windows沙箱临时目录限制下经授权在沙箱外执行本地合成回归。未访问远程权重、代为commit/push或提交GPU作业。
本条仅前置，RM全部历史字节完整保留。

## 2026-11-13：为用户指定的五个 Mizar 目标准备 MMSU 提交预设

用户选择官方 Mizar 本身，以及 base / 普通 recursive_5_10_5 /
recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8 / 原始 recursive_mesh
四个明确的 model--step-17313.ckpt，要求复用已存在的 MMAU/MMAR 适配并提供提交命令。
已逐一核对旧提交分支、runtime 注册、训练 contract 与官方 release manifest。
官方 Mizar 本身对应 /hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M/
checkpoints/final/stage3_q4_seed20260905.ckpt，使用既有严格发布包 loader；
其余四条 total_step=17313、geometry=8,8,4，沿用各自 run 下 formal/runtime.json。

新增 code/RSmol/run_mizar_mmsu_selected5_5090.sh，五个 TARGET 为：
official_mizar、base、recursive_5_10_5、serial84_lora_r8、recursive_mesh。
其固定训练 run 分别对应用户所列 20261026_193830617147305、20261028_160823559722647、
20261103_165410438279511、20261027_081123150934256；checkpoint 路径、route 目录、
step、geometry 和独立输出均显式绑定，最终调用原通用 MMSU 提交入口。
仅允许数据路径、scorer 路径、output-dir、Parquet I/O batch-size 与 --check-args；
目标模型身份参数固定，其余架构继续使用通用入口。

五条均为完整5000题、pdgpu-5090、1GPU/8CPU/32G/1node、原镜像/conda/离线协议。
官方输出为 outputs/RSmol/mizar_official_stage3_q4_seed20260905_matched_ced_v1/mmsu；
四条训练目标分别写入各自 run 的 eval/mmsu_step17313。
说明与可直接复制的五条远程提交、本地Git同步命令：
code/RSmol/mizar_s1_ced_mmsu_eval/SELECTED5.md。
旧官方Mizar MMAU/MMAR的4090提交分支保持；本次MMSU按已有通用入口走5090。

tests/test_mizar_mmsu_selected5.py执行真实新入口、通用提交和计算shell与参数解析，
仅vc/conda为显式fixture，核验全部目标的精确路径/geometry、资源、独立输出、
路径空格、只检查不提交和错误参数拦截，并与原官方发布manifest核对。
联合已有MMSU shell回归与28份母本SHA检查，8项全部通过，接入原完整MMSU validator。
报告 audit_reports/mizar_mmsu_selected5_validation.json；Windows沙箱外运行本地合成检查。
原MMSU推理/评分实现、通用提交脚本、MMAU/MMAR、训练及模型文件均未修改。
仅进行本地代码准备，未访问远程生产权重、代为commit/push或提交真实GPU作业。
本条仅前置，全部RM旧内容字节保留。

## 2026-11-13：新增独立 MMSU 完整测评链路

按用户确认方案实现 MMSU 官方大小写题面，保留项目现有 CED 的数值与推理参数。
正式入口：code/RSmol/run_mizar_s1_ced_small_mmsu_eval_5090.sh。
实现与详细参数模板：code/RSmol/mizar_s1_ced_mmsu_eval/README.md。
本地 Windows 只维护代码，经 GitHub 同步；生产权重和 GPU 推理仍在远程 Linux。
未预选架构、训练 run、checkpoint、step 或 geometry，全部由后续目标显式指定。

新增隔离 dataset/runtime/storage/scoring/evaluate 与计算入口；导入原有严格模型 loader、
greedy decoder 和每样本交互 context，支持现有 base、mesh、简单递归、multilayer 注册路线，
以及既有官方最终 release loader；尚未被原 runtime 支持的新架构不自动推断。
原 MMAU/MMAR 的入口、runner、runtime、训练和模型文件均不修改，28 份原评测文件字节 SHA 一致。

默认读取 /hpc_stor03/sjtu_home/jinwei.zhang/data/MMSU 下 MMSU data 和 MMSU code，
保持含空格路径的完整引用，依次遍历三份 train parquet 的 1667/1667/1666 行。
核验 5000 个唯一 ID、schema、原始选项/null 和文本 GT；GT/分类不进入模型题面。
音频优先嵌入 bytes，按 SoundFile 实际解码采样率/长度，先 mono 后 torchaudio 32k resample，
训练 fixed20 整段居中裁剪/右补零后分两个10秒；FP32、prompt256、prefix383、300 token
纯 greedy/完整重算/无 KV/无 attention mask、原始 EOS 解码等保持已有口径。

官方 mmsu_evaluation.py 原样外部执行，CLI 使用位置参数 JSONL，LF SHA 固定为
b6d5d35d2755244019a78e601495ea74e722681bf6f7e1fd0179bab7cabca3eb。
response 不转大写、不抽取字母、不删 EOS；保留官方对部分格式错误回答排除分母的行为。
仅满 5000 条唯一生成后评分；分开记录推理覆盖、scorer 执行、实际分母。
scorer 正常退出但实际分母不足5000时保留官方分数/输出，整轮仍 FAILED、返回1。

输出包含 run_config、progress、raw_generations、model_audit、predictions_official.jsonl、
official_evaluation.txt、evaluation_report.json。严格绑定 checkpoint/runtime/data/scorer/code/protocol；
OS进程锁防并发写，raw 为恢复依据，追加并 fsync；只恢复未完成末尾写，不吞完整损坏记录。
出错停止、不跳题，匹配身份后续跑剩余条目，完整 raw 重评分不再加载模型。
提交默认完整评测，不增加自动 smoke 或额外 GPU 测试；沿用5090的1卡/8CPU/32G/原镜像/原conda。

python -B tests/validate_mizar_s1_ced_mmsu_eval.py --assets-root C:/Xlance/assets/MMSU：
122项中119通过，3项真实Torch/torchaudio检查因本地缺依赖明确跳过。
含完整5000条fixture控制流、中断续跑、原官方scorer、真实Bash模拟vc、原有MMAU/MMAR相关回归。
独立临时依赖目录提供PyArrow/SoundFile，未修改原Python环境；实际核验5000条/47任务元数据，
并真实解码全部5000音频：8种采样率、450段双声道，时长0.302至35.915秒，
145段会被fixed20截断，4855段需补零；298段头部帧数与实际解码不同，以实际解码为准。
修正验证入口选用Git Bash以避免WindowsApps WSL别名；两个新增脚本bash -n通过。
报告：audit_reports/mizar_s1_ced_mmsu_eval_validation.json。
不声明真实模型/远程推理通过；未访问远程生产权重、提交真实作业或代为commit/push。
本条仅前置，原RM全部历史字节完整保留。

## 2026-11-12：已训练文本双LoRA epoch1音频路线的tmpfs前缀自查与修复

用户要求自查run_mizar_s1_ced48_text_epoch1_loop_lora_r8_5090.sh是否存在隔离目录前缀问题。
已确认其调用的mizar_s1_ced48_text_epoch1_loop_lora_r8/run.sh原STAGE_ROOT为
/dev/shm/mizar_s1_ced48_text_epoch1_loop_lora_r8_<token>，与未改的stage_data.py
validate_stage_root要求的mizar_s1_gbs256_前缀不符；启动和EXIT cleanup均会先被同一校验拒绝。
仅将实际运行路径修正为/dev/shm/mizar_s1_gbs256_ced48_text_epoch1_loop_lora_r8_<token>。
保留路线后缀、每作业token及/dev/shm直属/非软链接/解析路径一致/owner检查；
stage_data.py、文本checkpoint及已训练A/B继承、音频模型和训练配方均不改。

新增tests/test_mizar_s1_ced48_text_epoch1_loop_lora_r8_staging_roots.py的4项回归：
执行run.sh真实Bash赋值，再执行原validate_stage_root/cleanup函数；
修复前复现旧目录启动/清理失败，修复后通过，验证不同token的独立目录、
不存在目录的安全清理、匹配owner才允许rmtree、越界/嵌套/软链接/解析别名及错误owner/token拒绝。
POSIX文件查询及rmtree为显式fixture，不创建或删除真实/dev/shm目录。
此前假staging进程遗漏了此接口校验，本次补齐；validator接入新suite，原路径断言同步。
README补充故障原因、修正路径和重提说明，reference_audit.json按新源码重新生成。

python -B tests/validate_mizar_s1_ced48_text_epoch1_loop_lora_r8.py：42项本地检查通过、6项真实CPU tensor缺依赖跳过。
报告audit_reports/mizar_s1_ced48_text_epoch1_loop_lora_r8_validation.json；37份母本SHA及音频配方对照通过。
Windows沙箱阻止Bash信号管道，经授权在沙箱外运行同一本地回归及整套验证。
未读取远程权重/数据、提交真实作业或代为commit/push，不声明远程训练PASS。
若旧作业只在该校验处失败且未进入torchrun，同步后用原入口重新提交到新输出目录即可。
仅前置本条，原RM所有内容逐字节保留；其他并行路线修改保持。

## 2026-11-12：自查并修复 Huginn 文本 epoch1 音频路线 tmpfs 前缀

用户提醒其他路线出现staging目录前缀不兼容后，自查确认Huginn入口也有同样问题。
入口run_mizar_s1_ced_small_huginn_text_epoch1_5090.sh调用独立目录run.sh；
旧STAGE_ROOT=/dev/shm/mizar_s1_ced_small_huginn_text_epoch1_<token>，
而stage_data.py的validate_stage_root及cleanup只允许mizar_s1_gbs256_前缀。
此前模拟staging进程的检查未覆盖真实路径守卫，导致该启动/清理兼容问题漏检。

仅将本路线run.sh的生产代码一行改为：
/dev/shm/mizar_s1_gbs256_ced_small_huginn_text_epoch1_<token>。
保留Huginn专属后缀及唯一token，stage_data.py及所有路径/软链接/所有权保护不改；
指定Huginn checkpoint、模型、训练设置、staging/清理时序、输出目录与作业入口均保持。

新增tests/test_mizar_s1_ced_huginn_text_epoch1_staging_roots.py，
从真实run.sh执行Bash路径赋值，直接执行stage_data.py中实际validate_stage_root/cleanup函数。
修复前重现启动及清理拒绝，修复后验证新路径及独立token、错误token/root、越界/嵌套/软链接/解析别名拒绝。
Windows仅POSIX文件查询和rmtree为显式fixture，无真实/dev/shm目录创建或删除。
Huginn validator新增staging_roots四项回归；原目录差异断言同步，README更新说明和Git文件清单。
python -B tests/validate_mizar_s1_ced_huginn_text_epoch1.py：46项本地通过、7项真实CPU tensor缺依赖跳过；
报告audit_reports/mizar_s1_ced_huginn_text_epoch1_validation.json。
Windows沙箱阻止Bash信号管道后，经授权在沙箱外运行同一检查；未访问生产资产或提交真实训练。

本地提交推送并在远程git pull --ff-only后，沿用原入口：
bash code/RSmol/run_mizar_s1_ced_small_huginn_text_epoch1_5090.sh --stage-workers 4。
本轮只修复Huginn路线，不改工作区其他路线的已有修改；仅前置本条，旧RM字节完整保留。

## 2026-11-12：修复普通文本 epoch1 两条音频路线的 tmpfs staging 前缀

用户提供普通CED及CED4→8串行音频注入/双LoRA两条正式作业的启动日志。
直接原因：复制隔离时run.sh改为mizar_s1_ced_*临时目录名，而未修改的stage_data.py
只允许/dev/shm直属、非软链接、解析路径不变且名称以mizar_s1_gbs256_开头的目录。
producer在创建目录/发布TRAINING_READY前退出，随后wait_for_start与EXIT清理连锁报错；
日志未进入torchrun，重新提交为全新音频训练，不使用resume。

仅修正以下两条运行脚本的STAGE_ROOT命名：
code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_text_epoch1/run.sh：
/dev/shm/mizar_s1_gbs256_ced_small_text_epoch1_<token>
code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8/run.sh：
/dev/shm/mizar_s1_gbs256_ced_text_e1_in48_loop_lora_r8_<token>
两路线的后缀及token仍独立。stage_data.py及路径/软链接/所有权保护保持母本原样，
checkpoint来源、模型、optimizer、GBS、epochs、学习率、音频处理及数据/staging实现均不修改。

新增tests/test_mizar_s1_ced_text_epoch1_staging_roots.py：
执行真实run.sh中的Bash赋值，将实际输出交给原validate_stage_root及cleanup函数；
Windows仅POSIX路径文件查询/rmtree使用显式fixture，不创建/删除真实/dev/shm目录。
修复前重现两条路径在启动/清理校验失败，修复后通过；
同时验证目录隔离、越界/嵌套/软链接/解析别名拒绝、错误token/root拒绝。
原先模拟staging进程而漏检目录前缀的覆盖缺口现由该回归覆盖。
两个validator已接入此四项用例，路径断言与LoRA源码审计同步。
普通版49项本地通过/3项真实tensor缺依赖跳过；LoRA版44项本地通过/7项真实tensor缺依赖跳过。
Windows沙箱阻止Bash管道及测试临时目录后，经授权在沙箱外执行同一本地合成验证。
报告为audit_reports/mizar_s1_ced_text_epoch1_validation.json及mizar_s1_ced_text_epoch1_loop_lora_validation.json。
两个README同步路径、失败原因及联合Git提交命令，LoRA reference_audit.json已按实际新源码重新生成。

本地修复提交推送后，远程仓库根分别重新提交，自动使用新的时间戳输出目录：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_text_epoch1_5090.sh --stage-workers 4
bash code/RSmol/run_mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_5090.sh --stage-workers 4
```
此轮未修改其他隔离路线，未访问远程生产权重/数据、未提交真实GPU作业或代为commit/push。
仅前置本条，之前RM内容字节完整保留。

## 2026-11-11：Huginn 文本 epoch1 checkpoint 初始化 CED-Small 音频 S1

按用户确认方案复制隔离原CED-Small音频S1完整路线；正式入口：
code/RSmol/run_mizar_s1_ced_small_huginn_text_epoch1_5090.sh。
实现及正式提交/完整恢复/Git命令见
code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_huginn_text_epoch1/README.md。
本地Windows维护核心代码，经GitHub同步；权重、数据和GPU执行保留在远程Linux。

默认直接使用：
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_huginn_gbs1024/epoch1/checkpoint-step-009244。
显式MizarHuginnForCausalLM严格加载全部HF权重及训练后的共享A，不重新初始化A；
无需重新转换或先训文本epoch2。只读HF权重/tokenizer/JSON，不反序列化文本training_state.pt；
音频optimizer/scheduler/scaler/GradNormTracker/各rank RNG及游标从step0新建。
20物理层/30次执行，E=Prelude(完整683位置多模态embedding)，S0逐元素±3σ截断高斯，
σ=√(2/5)，每forward只采样一次；S1=Core(A([S0,E]))，S2=Core(A([S1,E]))，
S2直接进入后五层/final norm/head。同一无bias1152→576投影两次，663552参数，
拼接顺序[state,prelude]；E和S1不detach，固定两轮完整反传，全部LM含A全量训练。
模型副本models/huginn.py仅调整合同导入路径，避免Torch已加载stdlib code导致包名冲突；
模型计算/参数名/checkpoint格式与既有文本Huginn实现一致，原共享文本模型保持不变。

母本为run_mizar_s1_gbs256_recursive_5_10_5_ced_small_5090.sh及对应完整目录。
train.py、optim.py、原decoder/CED/mapper和全部data/index/staging源码逐字节继承；
更改集中于专用loader/合同、配置说明与隔离命名，35份参考源码LF SHA记录于source_guard.json。
.gitignore放行source_guard.json和parent_reference_audit.json；后者仅为母本历史记录。
新route ced_small_mizar_s1_fixed20_gbs256_huginn_text_epoch1_v1；
代码/output/log/job/index-cache/tmpfs/完整恢复身份隔离，其他既有路线不改。

严格保持hybrid1477492行、fixed20、冻结eval CED-Small及新初始化的freq_merge_126 mapper。
126audio+1SEP+256prompt+300answer=683，PAD为已有!、EOS/SEP0，attention_mask=None、无KV。
answer-only rank-local有效token均值→GA4均值→DDP均值；保留原seed1234及每rank1234+rank。
8GPU×micro8×GA4=GBS256，3epochs、5771updates/epoch、17313total、866warmup。
LM含A maxLR3e-4，mapper6e-4；原LambdaLR floor.1、AdamW(.9,.999)/eps1e-8、
矩阵WD.01/bias低维WD0、GradNormTracker、FP32前向/无autocast/GradScaler/TF32关闭保持。
共享音频立即训练、后台默认4workers复制、optimizer边界全rank共同切换、先reap再清理保持。
完整音频恢复含模型/A/mapper和所有训练状态、8rank RNG及游标，沿用同run输出根，拒绝其他路线。
保留mellow_c8204d8、原镜像、pdgpu-5090/32CPU/256G/8GPU/1node；
原CED资产检查及CPU索引准备后直接formal，无新增自动smoke或tensor前置。

代码提交推送后远程登录节点执行：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_huginn_text_epoch1_5090.sh --stage-workers 4
```
默认outputs/RSmol/mizar_s1_ced_small_huginn_text_epoch1_gbs256_5090/formal_<timestamp>/，
沿母本追加formal/；最终formal/checkpoints/model--step-17313.ckpt和model--epo-3.ckpt。
README含显式来源/输出、同run完整恢复及精确Git命令；RM按块仅选本条。

python -B tests/validate_mizar_s1_ced_huginn_text_epoch1.py：42项本地检查通过，7项真实CPU tensor缺依赖跳过。
覆盖35份参考SHA、逐文件差异、完整semantic配方/17314个LR位置、严格加载/新音频step0、
A形状/训练分组/绑定权重、错误checkpoint及跨路线恢复拒绝、导入兼容、共享音频/后台复制/切换/清理、
Bash模拟vc资源和参数、Git审计清单可提交性；报告audit_reports/mizar_s1_ced_huginn_text_epoch1_validation.json。
可在远程mellow_c8204d8可选加--require-tensors检查合成CPU模型梯度、A继承/更新和随机恢复；不作formal前置。
Windows沙箱阻止临时目录及Bash管道后，经授权在沙箱外运行同一本地验证。
未代为commit/push、访问远程生产资产或提交真实作业；不登记真实权重加载/GPU训练/8rank恢复PASS。
仅前置本条，之前RM所有内容字节完整保留。

## 2026-11-11：已训练文本双LoRA epoch1 → CED4→8 音频 S1（A/B原值继承）

按用户确认方案复制隔离指定CED4→8双LoRA母本，直接继承已训练文本LoRA进入音频阶段。
本地Windows仅维护核心代码，通过GitHub同步；权重、数据、输出及GPU执行均在远程Linux。
新入口 code/RSmol/run_mizar_s1_ced48_text_epoch1_loop_lora_r8_5090.sh；
实现/提交/恢复/Git命令见
code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced48_text_epoch1_loop_lora_r8/README.md。

母本为run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8_5090.sh及对应完整目录。
默认文本来源严格为：
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_loop_lora_r8_gbs1024/epoch1/checkpoint-step-009244。
无需转换或先完成文本epoch2。此路线继承文本checkpoint中的全部基础LM及已训练双LoRA，
区别于下方从普通文本epoch1再新建LoRA的另一条隔离路线。

20物理层/30次执行、前5→中10第0轮→中10第1轮→后5；中间物理层5..14七投影
q/k/v/o与gate/up/down使用r8/alpha16/scale2/dropout0两套独立A/B；前后五层无LoRA。
280个A/B张量、1,628,160参数与全部基础LM、embedding/head绑定直接继承并继续全量训练。
模型构造时先注册LoRA槽位，再由HF加载基础权重及完整A/B；加载后不重新安装/初始化LoRA。
HF missing/unexpected/mismatched/error均拒绝，添加音频模块前后两次逐项核验所有源tensor
名称/形状/FP32原值/有限性、绑定关系及全部280个LoRA；保存base/LoRA SHA审计。
来源文本lr_policy=same_as_base保留为元数据，音频optimizer采用母本分组学习率。

新初始化原freq_merge_126 mapper、两套CED4/8 MLP和gate、共享input投影及gate，三个gate零初始化。
冻结eval CED从原models/CED_small加载；每批一次编码取得final及完整block4/8输出。
逻辑索引5：共享input残差后CED4；索引15：同一input残差后CED8，CED门读取post-input hidden。
共享input投影每forward只计算一次，不detach；音频残差直接作用前126位置，LoRA作用完整序列。
保留逐层/异常清理轮次、无KV、禁用decoder gradient checkpointing。
首步保留训练后的非零B，相应两轮A/B均须有效任务梯度；新音频gate仍须从零开始，保留打开gate后审计。
只读HF权重/tokenizer和JSON元数据，不反序列化文本training_state.pt；
音频optimizer/scheduler/scaler/GradNormTracker/各rank RNG/数据游标从step0新建。

严格保留母本整个音频前向和训练更新循环；train.py仅增加继承审计、取消初始B必须全零的旧断言。
数据/CED/mapper/optimizer/staging等18份运行源码逐字节不变，37份原始参考文件LF SHA保护。
source_guard.json与parent_reference_audit.json通过本目录.gitignore例外同步；
reference_audit.json为本次源码对照，parent_reference_audit.json仅为母本历史记录。
新route ced_small_mizar_s1_ced48_text_epoch1_loop_lora_r8_v1，代码/output/log/job/index-cache/tmpfs及恢复身份隔离。

原hybrid1,477,492行、fixed20、126audio+1SEP+256prompt+300answer=683token、
PAD!/17、EOS/SEP0、attention_mask=None、answer-only rank-local token mean→GA4→DDP均值保持。
8GPU×micro8×GA4=GBS256，3epochs/5771updates每epoch/17313total/866warmup；
基础LM maxLR3e-4，双LoRA及全部音频可训练模块maxLR6e-4；原AdamW(.9,.999)/eps1e-8、
矩阵WD.01/bias低维WD0、LambdaLR floor.1、GradNormTracker、FP32无autocast/TF32、GradScaler保持。
共享音频立即训练，后台默认4workers复制，optimizer边界全rank共同切换，先stop/reap再清理。
完整音频恢复保留LM/双LoRA/音频模块及全部训练状态、8rank RNG和游标；拒绝其他路线恢复。
保持mellow_c8204d8、原镜像、pdgpu-5090/32CPU/256G/8GPU/1node；原CED资产及CPU索引准备后直接formal，无新增smoke前置。

本地提交推送后，远程登录节点执行：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced48_text_epoch1_loop_lora_r8_5090.sh --stage-workers 4
```
输出根outputs/RSmol/mizar_s1_ced48_text_epoch1_loop_lora_r8_5090/formal_<timestamp>/，
沿母本追加formal/；最终formal/checkpoints/model--step-17313.ckpt及model--epo-3.ckpt。
README含显式来源/固定输出和同run完整恢复命令，Git命令只暂存本路线新文件；RM按块只选本条。

python -B tests/validate_mizar_s1_ced48_text_epoch1_loop_lora_r8.py：38项通过、6项真实CPU tensor因缺依赖跳过。
涵盖完整母本源码/配方、17314个LR位置、非零A/B继承数值检查、严格加载顺序/失败、step0不读文本状态、
初始梯度及跨路线恢复拒绝、共享读取/并发复制/切换/失败清理、Bash假vc资源与参数。
真实tensor用例补齐非零双A/B文本类保存→音频公开loader加载（含shards/tied）、零音频gate logits对齐、
缺A/B拒绝、音频注册保留RNG/预训练值且禁止重置、实际反传/optimizer及CPU完整checkpoint下一步往返。
报告audit_reports/mizar_s1_ced48_text_epoch1_loop_lora_r8_validation.json；远程有依赖时可选--require-tensors，
仅合成CPU模型，无生产资产/CUDA/真实作业，不作为formal前置；collective/scaler为明确CPU fixture。
Windows沙箱曾阻止临时目录及Bash管道，经授权在沙箱外运行同一本地验证。
未代为commit/push、访问远程资产或提交GPU作业，不登记真实权重加载、GPU训练或8rank恢复PASS。
仅前置本条，工作区其他未提交路线记录及全部原RM内容字节保留。

## 2026-11-11：文本 epoch1 初始化 CED4→8 串行音频注入＋两轮独立 LoRA r8

按用户确认方案复制隔离指定完整LoRA母本；正式入口：
code/RSmol/run_mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_5090.sh。
实现及提交/恢复/Git命令见
code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8/README.md。
本地Windows维护核心代码、GitHub同步，模型/数据/输出及GPU执行位于远程Linux。

母本为run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8_5090.sh及对应训练目录。
默认直接继承/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_gbs1024/epoch1/checkpoint-step-009244，
显式严格加载完整非MeSH文本LM及tokenizer后，再注册CED层注入、共享input音频再注入及LoRA。
20物理层/30次执行、前5→中10→中10→后5、全部基础权重及embedding/head绑定保持；
只读HF权重/tokenizer，不读文本training_state.pt，音频optimizer/scheduler/scaler/tracker/RNG/游标从step0新建。

CED-Small冻结eval，每批一次编码取final及完整block4/8的final norm前输出；
原freq_merge_126 mapper生成E [B,126,576]，A_input=audio_linear(E)每次完整forward只算一次不detach。
零基逻辑索引5：共享input残差→CED4残差；索引15：同一input残差→CED8残差。
CED门控读取post-input hidden，直接残差限前126位置；两个CED MLP/gate独立，三个gate零初始化。
仅物理层5..14的q/k/v/o、gate/up/down七投影新增两轮独立LoRA A/B，
r8/alpha16/scale2/dropout0，A Kaiming/B零，基础W/bias共享且全量训练，前后五层全量训练无LoRA。
LoRA新增1,628,160参数，保留音频分支2,777,859参数；独立CPU RNG、
每层及异常时轮次清理、无KV、关闭decoder gradient checkpointing、首步/打开门控梯度审计保持。

严格保留母本train.py、optim.py及全部models/data/staging/index/CED配置等24份运行源码逐字节一致。
contract.py只改ROUTE和默认文本路径，run.sh只改入口提示和tmpfs前缀；
提交器只改目录/来源/输出/log/job/index-cache及提示。
audit_reference.py为此次直接母本的精确差异检查，不参与训练；35份原始文件LF SHA记录于source_guard.json。
.gitignore仅放行新增source_guard.json，确保普通Git提交包含该清单。
母本历史reference_audit.json不复制，本目录新reference_audit.json仅记录本地精确源码/配方验证。
新route为ced_small_mizar_s1_fixed20_gbs256_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_v1；
与旧LoRA母本、普通CED epoch1版本分别隔离，原入口及共享模型均保持。

原hybrid1,477,492行、fixed20、126audio+1SEP+256prompt+300answer=683token、
PAD!/17、EOS/SEP0、attention_mask=None、answer-only rank-local token mean→GA4→DDP平均保持。
8GPU×micro8×GA4=GBS256，3epochs/5771updates每epoch/17313total/866warmup；
基础LM maxLR3e-4，mapper、音频分支及两套LoRA maxLR6e-4；
原AdamW(.9,.999)/eps1e-8/矩阵WD.01/bias低维不衰减、LambdaLR floor.1、GradNormTracker、
FP32/无autocast/保留GradScaler/关闭TF32、seed1234和rank+seed、原数据顺序均保持。
共享音频立即训练、后台默认4workers复制、optimizer边界全rank共同切换、producer先reap再清理保持。
完整音频恢复含基础模型/音频分支/两套LoRA、所有训练状态、8rank RNG及游标；
严格拒绝其他路线或不同源码/资产/数据/配方恢复。

远程仓库根同步后：
```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_5090.sh --stage-workers 4
```
原CED资产/CPU索引准备后直接vc submit，无额外smoke/tensor前置；
mellow_c8204d8/原镜像/pdgpu-5090/32CPU/256G/8GPU/1node保持。
默认输出outputs/RSmol/mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_gbs256_5090/formal_<timestamp>/，
沿母本追加formal/；最终formal/checkpoints/model--step-17313.ckpt和model--epo-3.ckpt。
README含显式初始化/输出路径及同run完整恢复命令，Git命令只暂存此次新路线；
RM.md已有其他记录，需git add -p仅选本条。

python -B tests/validate_mizar_s1_ced_text_epoch1_loop_lora.py：40项本地检查通过、7项真实CPU tensor缺依赖跳过。
覆盖35份母本SHA、完整源码差异白名单、整个semantic recipe及17314个LR位置、
文本epoch1配置兼容/实际step0初始化、严格加载后注册、恢复拒绝、
非零按轮LoRA/scale2/post-input gate/梯度审计、共享读取/复制/切换/失败清理、Bash模拟提交资源与参数。
报告audit_reports/mizar_s1_ced_text_epoch1_loop_lora_validation.json。
Windows沙箱拒绝测试临时目录及Bash信号管道后，经授权在沙箱外运行同一本地检查。
可在远程mellow_c8204d8可选加--require-tensors执行合成CPU用例，不依赖生产资产，不作为训练前置。
未代为git commit/push或提交真实vc作业，未读取远程权重/数据；不登记真实GPU训练或八rank恢复PASS。
仅前置本条，之前RM内容字节完整保留。

## 2026-11-11：SmolLM2 非MeSH 文本 init0904 初始化 CED-small 音频 S1

按用户确认计划复制隔离，说明与提交/恢复命令见：
code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_init0904/README.md。
本地Windows维护核心代码、GitHub中转，权重/数据/生产执行均在远程Linux。

完整复制 run_mizar_s1_gbs256_recursive_5_10_5_ced_small_5090.sh 对应训练路线。
新入口 run_mizar_s1_gbs256_recursive_5_10_5_ced_small_init0904_5090.sh 默认文本来源为：
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-maxlr8e-4-resume4500-20260904_154711/checkpoint-step-009244。
无需重新转换；显式非MeSH RecursiveLlamaForCausalLM 严格加载完整LM和同目录tokenizer，
保留hidden576、20物理层/30次执行、固定5+10×2+5和原抽层映射，无MeSH/router/LoRA/音频重注入。
只继承文本权重/tokenizer，不读取文本training_state.pt；音频训练step0重新建立optimizer/scheduler/
GradScaler/GradNormTracker/各rank RNG/数据游标。CED-small从原资产冻结eval加载，原freq_merge_126
mapper按母本顺序随机初始化并训练，完整LM继续全量训练。

train.py、optim.py、write_config.py、全部model/data/索引/staging保持母本原实现；
contract.py只改ROUTE及默认文本路径，run.sh只改入口提示和tmpfs前缀。
新route ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_init0904_v1；入口/output/log/job/
index-cache/tmpfs及完整恢复身份隔离，原母本/共享模型/现有评测不改。
38份母本/参考文件归一化LF SHA见source_guard.json；parent_reference_audit.json为原路线历史审计，
其PASS和源码身份仅指原路线，不冒充本次新路线生产验证。

保持hybrid1,477,492条、fixed20、audio126+SEP1+prompt256+answer300=683token，
原模板/PAD!/EOS与SEP0/attention_mask=None/无KV/answer-only loss保持。
loss为rank-local microbatch有效token均值→GA4均值→DDP均值，不改成全局token归一化。
8×micro8×GA4=GBS256、3epochs/5771updates每epoch/17313总步/866warmup，
LM maxLR3e-4及mapper6e-4、原LambdaLR floor.1、AdamW(.9,.999)/eps1e-8/WD.01、
bias/低维不衰减、原GradNormTracker、FP32前向/无autocast/GradScaler/TF32关闭保持。
共享原始音频立即训练，单producer默认4workers后台复制，完成后optimizer边界全rank MIN切换；
退出先stop/reap再清理本token tmpfs；每次resume重建staging从shared开始。
原mellow_c8204d8/镜像/pdgpu-5090/32CPU/256G/8GPU/1node；直接formal，无自动smoke/tensor前置任务。

本地代码提交推送GitHub后，远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_init0904_5090.sh --stage-workers 4
```

输出根outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_init0904_5090/formal_<timestamp>/，
其下formal/checkpoints/model--step-17313.ckpt及model--epo-3.ckpt为最终产物。
完整恢复沿用原run root，要求本init0904同代码/资产/依赖/数据/配方；拒绝原路线音频checkpoint完整恢复。
README提供显式输出、同run恢复、可选CPU验证及精确git提交命令，RM.md按块只暂存本条记录。

python -B tests/validate_mizar_s1_ced_init0904.py：42项实际本地检查通过，3项真实CPU tensor缺依赖跳过。
覆盖38份母本SHA、整份trainer/loader/数据/优化器及复制受限差异、完整semantic合同和17314个LR位置，
执行真实fresh/resume初始化代码块验证step0不读旧状态、旧route/错身份拒绝及八rank状态选择，
严格加载和tokenizer检查、实际并发后台复制/共享读取/集体切换/失败清理、Bash假vc及torchrun入口验证。
报告audit_reports/mizar_s1_ced_init0904_validation.json；远程依赖完整时可手动加--require-tensors，
仅合成CPU模型、无生产资产/CUDA/作业提交，不作为formal前置。
Windows沙箱阻止临时目录和Bash signal pipe后，经批准在沙箱外运行同一验证通过。
未代为git commit/push、读取远程生产资产或提交GPU作业，不登记真实权重加载/八卡训练/八rank恢复PASS。
工作区另一条Mizar文本epoch1音频路线记录及此前RM内容完整保留，仅前置本条。

## 2026-11-11：Mizar 文本 epoch1 checkpoint 初始化 CED-Small 音频 S1

按用户确认方案复制隔离正式音频路线，说明与提交/恢复命令见：
code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_text_epoch1/README.md。
继续本地 Windows 维护核心代码、GitHub 中转、远程 Linux 保存权重/数据并执行 GPU 作业。

母本为 run_mizar_s1_gbs256_recursive_5_10_5_ced_small_5090.sh 及对应完整训练目录。
新入口 run_mizar_s1_ced_small_text_epoch1_5090.sh，默认直接使用：
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_gbs1024/epoch1/checkpoint-step-009244。
无需再次转换或先训练文本 epoch2；严格显式加载原非MeSH RecursiveLlamaForCausalLM 与对应 tokenizer，
保留20物理层/30次执行、前5→中10→中10→后5、已训练LM全部权重及embedding/head绑定。
只读取HF文本权重/tokenizer，不读取文本training_state.pt；音频optimizer/scheduler/GradScaler/
GradNormTracker/各rank RNG/数据进度从step0重新开始。CED仍由models/CED_small加载冻结eval，
官方freq_merge_126 mapper沿用母本随机初始化，完整mapper和完整LM全量训练。

严格保持原训练实现：train.py、optim.py、全部模型/数据/音频预处理/离线CED配置/索引/staging逐字节一致。
contract.py只修改ROUTE和默认文本路径；run.sh只修改入口提示和tmpfs前缀；
提交器只修改代码目录/文本初始化/输出/log/job/index-cache名称和提示。35份参考文件LF SHA见source_guard.json。
独立身份ced_small_mizar_s1_fixed20_gbs256_text_epoch1_5_10x2_5_v1；原入口、共享模型和文本路线未修改。

原hybrid 1,477,492行、fixed20、126audio+1SEP+256prompt+300answer=683token、
PAD为已有!、EOS/SEP0、attention_mask=None/无KV、answer-only rank-local token mean→GA4→DDP平均保持。
8×micro8×GA4=GBS256、3epochs/5771steps每epoch/17313updates/866warmup；
LM maxLR3e-4、mapper maxLR6e-4、原LambdaLR floor.1、AdamW(.9,.999)/eps1e-8/WD.01、
bias/低维不衰减、GradNormTracker、FP32 forward/无autocast/GradScaler/关闭TF32保持。
共享音频立即训练、后台默认4workers复制、optimizer边界全rank MIN切换及先reap再清理保持。
原完整音频checkpoint/八rank RNG及采样游标恢复保持，拒绝旧路线或不同源资产/代码/数据/配方恢复。
不增加smoke或CPU张量测试前置；保留原CED配置检查和CPU index准备后直接vc submit。
原mellow_c8204d8环境、镜像、pdgpu-5090/32CPU/256G/8GPU/1node保持。

GitHub同步后远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_text_epoch1_5090.sh --stage-workers 4
```

默认输出outputs/RSmol/mizar_s1_ced_small_text_epoch1_gbs256_5090/formal_<timestamp>/，
沿母本再追加formal/；最终formal/checkpoints/model--step-17313.ckpt与model--epo-3.ckpt。
index-cache/log/job/tmpfs命名独立；README含显式文本来源、新输出路径、同run完整恢复及精确git提交命令。

python -B tests/validate_mizar_s1_ced_text_epoch1.py：45项本地检查通过，3项真实CPU tensor缺依赖跳过。
覆盖35份母本SHA、全部未调整文件逐字节一致、整个semantic recipe、17314个LR位置、
GradNorm数值、实际step0初始化不读文本状态、旧音频恢复拒绝、严格LM加载与tokenizer、
共享读取/后台复制/切换/失败清理、Bash正式提交资源和参数及原完整更新循环继承。
报告audit_reports/mizar_s1_ced_text_epoch1_validation.json；可在远程mellow_c8204d8可选加--require-tensors，
仅使用合成CPU模型，不要求生产权重/数据，也不作为训练前置。Windows沙箱拒绝临时目录和Bash管道后，
经授权在沙箱外运行同一本地检查。未代为git commit/push、读取远程生产资产或提交GPU作业，
不登记远程权重加载、真实训练或八rank恢复PASS。仅前置本条，原RM内容字节完整保留。

## 2026-11-09：Mizar 文本 Huginn 类共享状态投影，两次独立单 epoch 训练

按已确认方案复制并隔离新路线，说明与执行/提交命令见 code/RSmol/mizar_text_5_10x2_5_huginn/README.md。
本地 Windows 维护核心代码、GitHub 中转、远程 Linux 保存权重/数据/输出并执行任务。
保留原 Mizar 文本、LoRA 及其他路线，16 份母本文件的归一化 LF SHA256 记录于 source_guard.json。

独立模型 recursive_model_mizar_text_5_10x2_5_huginn.MizarHuginnForCausalLM：
20 物理层/30 次 Transformer 执行，E=前5层输出；S0 为 [B,L,576] 独立逐元素截断高斯，
底层 σ=√(2/5)、±3σ 截断，每次完整 forward 只采样一次；S0 为临时激活，不是模型参数或持久状态。
S1=Core(A(cat([S0,E])))，S2=Core(A(cat([S1,E])))，S2 直接进入后5层/final norm/head。
同一 Linear(1152,576,bias=False) 调用两次，新增663,552参数；拼接顺序固定[state,prelude]。
E 和 S1 不 detach，固定两轮完整反传，所有继承权重及A全量训练；沿用原 norm/attention/RoPE/30槽cache。
A采用 σ=√(2/(5×576))、±3σ 截断正态，在独立CPU seed0 RNG上下文中初始化，不改变调用者RNG。
投影在BF16 autocast后转回原FP32 residual dtype；不引入MeSH、LoRA、sandwich norm、可变深度或截断BPTT。

CPU离线转换8线程，严格从 Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt 的 caption_decoder.lm.* 抽取，
不使用 assets/smollm2-135m/model.safetensors；仅核验最终checkpoint及文本assets的官方release SHA和tensor inventory。
沿用前1..5、中6/8/10/12/14/16/18/20/22/24、后26..30的一基选层；逐项核验继承tensor并严格组装唯一新增A。
验证30次物理trace、A两次调用、保存重载全部tensor含A、绑定权重、固定同一噪声实现的logits和tokenizer。
显式新模型类加载，无全局AutoModel注册；默认模型 models/Mizar159M-text-5-10x2-5-huginn，拒绝覆盖，失败staging保留。

沿用原8×micro8×GA16=GBS1024、9244updates/9244scheduler domain/463warmup、每次单epoch。
epoch1 max/min LR=8e-4/8e-5，epoch2=2e-4/2e-5；用户手动将 --model-dir 指向epoch1完整checkpoint-step-009244。
第二epoch严格继承全部训练权重含A，不读training_state.pt；optimizer/scheduler/RNG/计数/游标重新开始、seed0从头读。
保留85片seed0 shuffle后连续分rank、buffer4096、片尾不足micro8丢弃、context1024/动态padding/不packing。
FP32参数及AdamW状态、BF16仅forward、FP32 CE、单组AdamW(.9,.95)/eps1e-8/WD.1/clip1、全局有效token加权梯度均保留。
每500步及最终保存，保留8500/9000/9244；名义每epoch9,465,856样本，不声明严格遍历全部有效文本恰好一次。

模型Torch CPU/CUDA RNG按seed+1009×rank独立，结构审计保护RNG；完整checkpoint收集每rank的Python/NumPy/CPU/当前CUDA RNG。
同epoch --resume-from要求相同LR、新空输出目录，按rank恢复RNG、optimizer/scheduler/计数/数据游标，累计到9244。
仍不保存bounded-shuffle buffer内容，不声称数据逐样本连续或八rank训练bitwise exact。
正式入口直接vc submit，无数据扫描/Gate B/smoke前置；沿用pdgpu-5090/32CPU/256G/8GPU/1node及原镜像/rsmol环境。
新模型/输出/log/job/环境前缀RSMOL_MIZAR_TEXT_5_10X2_5_HUGINN完全隔离，转换CONVERT_OUTPUT_DIR与训练OUTPUT_DIR分开。

远程仓库根 /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM 同步后：

```bash
bash code/RSmol/scripts/convert_mizar_text_5_10x2_5_huginn.sh
bash code/RSmol/run_stage4_mizar_text_5_10x2_5_huginn_training_5090.sh --max-lr 8e-4 --min-lr 8e-5
```

本地 python -B tests/validate_mizar_text_5_10x2_5_huginn.py：33项通过，13项真实CPU tensor因缺Torch等依赖跳过。
涵盖母本SHA、原更新/数据/损失/attention/cache AST、全9245点LR、第二epoch实际控制流、完整RNG合同拒绝错误恢复、
NumPy执行模型实际循环并对照两轮递推公式、真实Bash模拟vc→torchrun资源/转义/旧环境隔离/失败传递。
NumPy不等于PyTorch autograd验证；可在远程rsmol运行 --require-tensors 的小模型CPU测试，无需生产资产、不作为训练前置。
张量测试覆盖共享A与完整梯度、分布/初始化RNG、固定噪声数值/BF16 residual、训练后A保存重载/缺A严格拒绝、
固定每token噪声的30槽cache、每rank RNG审计保护及完整checkpoint序列化往返恢复。
报告 audit_reports/mizar_text_5_10x2_5_huginn_validation.json；Windows沙箱临时目录受限后在获准的沙箱外运行同一本地检查。
未代为git commit/push、访问远程生产资产、执行真实转换或提交GPU作业，不登记远程转换/训练/八rank恢复PASS。
原RM内容及工作区另一条LoRA记录完整保留；提交命令精确列出Huginn文件，RM.md按块仅选本条记录。

## 2026-11-09：Mizar 文本中间两轮独立 LoRA rank8/alpha16 隔离实现

按用户确认计划，在既有 Mizar 文本 5＋10×2＋5 路线基础上复制隔离。
说明与执行/提交命令：code/RSmol/mizar_text_5_10x2_5_loop_lora_r8/README.md。
保持本地 Windows 维护核心代码、GitHub 中转、远程 Linux 保存全部权重/数据并执行任务。
原模型/转换/trainer/提交脚本不改，18 份参考源文件的归一化 LF SHA256 已记录并验证。

专用模型 recursive_model_mizar_text_5_10x2_5_loop_lora_r8.MizarTextLoopLoRAR8ForCausalLM，
20 物理层执行前5→中10第0轮→中10第1轮→后5，共30次；沿用原选层、attention/RoPE/30逻辑cache。
仅中间物理层5..14（零基）的 q/k/v/o、gate/up/down 七投影新增按轮独立 A/B，
rank8、alpha16、scale2、dropout0、A Kaiming/B零；140对、280 tensor、1,628,160新增参数。
各层/投影/轮次独立，基础W共享且继续训练，前后五层无LoRA；所有参数同LR、单组AdamW WD.1。
原W/bias对象与state_dict名称保留，初始化隔离CPU RNG；显式模型类加载，不调用全局AutoModel注册。
关闭decoder gradient checkpointing，明确拒绝启用，保证按轮作用域与autograd选中参数一致。

CPU离线转换8线程，源为Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt的caption_decoder.lm.*，
不从assets/smollm2-135m/model.safetensors取权重；只核验final checkpoint及文本assets官方release SHA。
严格加载源LM并逐项继承选中权重，检查LoRA范围/数量、30次trace、全量可训练、绑定embedding/head；
零B logits与相同继承权重的原循环模型精确比较，保存重载全部tensor（含A/B）、logits、tokenizer。
拒绝覆盖，失败staging保留；默认模型models/Mizar159M-text-5-10x2-5-loop-lora-r8。

复制母本完整训练更新循环、数据流和优化方法：8×micro8×GA16=GBS1024，
每次单epoch 9244updates/9244scheduler domain/463warmup；epoch1 max/min8e-4/8e-5，epoch2为2e-4/2e-5。
用户手动用 --model-dir 指向epoch1完整checkpoint-step-009244提交epoch2，完整继承基础权重及两套已训练A/B，
不读取training_state.pt；optimizer/scheduler/RNG/计数/游标重置、seed0从头读。
同epoch --resume-from恢复完整状态并要求同LR、新空输出目录，累计至9244；旧路线/缺失A/B拒绝。
保留FP32参数和AdamW状态、BF16仅forward、FP32 CE、全局token归一化、无额外GA除法、clip1。
85片seed0 shuffle后连续分rank（11×5+10×3），buffer4096、片尾不足micro8丢弃、动态padding/context1024/无packing。
每500步及最终保存，保留8500/9000/9244；全恢复保留母本shuffle buffer未保存、仅rank0模型RNG快照限制，
不声称逐样本连续或八rank bitwise exact。

正式入口直接vc submit，无CPU数据扫描/Gate B/smoke前置；沿用pdgpu-5090/32CPU/256G/8GPU/1node、原镜像和rsmol。
模型/输出/log/job/环境变量完全隔离，转换CONVERT_OUTPUT_DIR和训练OUTPUT_DIR分开。
远程仓库根/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM同步后CPU转换和正式提交：

```bash
bash code/RSmol/scripts/convert_mizar_text_5_10x2_5_loop_lora_r8.sh
bash code/RSmol/run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh --max-lr 8e-4 --min-lr 8e-5
```

本地 python -B tests/validate_mizar_text_5_10x2_5_loop_lora_r8.py：34项通过，10项真实CPU tensor因缺Torch等依赖跳过。
包含参考源SHA、完整更新/数据/attention/cache AST、两组各9245点LR、第二epoch实际启动不反序列化旧状态、
NumPy执行真实LoRA前向/有限差分/异常作用域清理、真实Bash vc→torchrun转义/资源/新旧环境隔离/失败传递。
NumPy不等同验证PyTorch autograd。10项tensor测试留给远程rsmol --require-tensors，可运行合成CPU模型，
覆盖任务梯度两轮A/B、初始化RNG与继承权重、训练后非零LoRA保存重载、缺A/B严格拒绝、30槽cache、
完整checkpoint optimizer moments/RNG/指定rank游标恢复；无需生产模型/数据，不作为训练前置门槛。
报告audit_reports/mizar_text_5_10x2_5_loop_lora_r8_validation.json；Windows沙箱拒绝测试临时目录后在获准的沙箱外运行。
未代为git commit/push、访问远程生产资产、执行生产转换或提交GPU作业，不登记远程转换/训练/八rank恢复PASS。
本条仅前置项目记录，原RM内容完整保留。

## 2026-11-09：Mizar 最终文本权重抽层循环与单 epoch 文本训练

按用户确认方案完成独立代码，说明见 code/RSmol/mizar_text_5_10x2_5/README.md。
继续本地 Windows 维护核心代码、GitHub 中转、远程 Linux 保存全部权重/数据和执行任务。
原 5＋10×2＋5 与 5＋20×2＋5 及其他既有路线保持不变；十份母本归一化 LF SHA 记录于新路线 source_guard.json。

CPU 转换从 models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt
提取 caption_decoder.lm.*，不使用 assets/smollm2-135m/model.safetensors 作为训练权重。
只校验最终 checkpoint 和文本配置/tokenizer 的官方 release SHA，不要求音频资产或构造 CED/mapper。
前五/后五层保留，中间原二十层沿用隔层取十层：一基 6,8,10,12,14,16,18,20,22,24；
20 物理层，执行前5→中10→中10→后5共30次，中间参数共享，无 MeSH/router/LoRA/音频分支。
使用原 recursive_model_5_10_5.RecursiveLlamaForCausalLM 显式加载，不调用全局 AutoModel 注册。
转换检查源 LM 严格加载、绑定 embedding/head、全部继承 tensor、30次 trace、保存重载 tensor/logits/tokenizer，
CPU 离线8线程、拒绝覆盖、失败 staging 保留；默认输出 models/Mizar159M-text-5-10x2-5。

独立单 epoch trainer 复制较新无审计文本入口，恢复原 8×micro8×GA16=GBS1024、
9244 updates/9244 scheduler domain/463 warmup，保留原完整更新循环、全局有效 token 加权 CE、
FP32 参数/AdamW 状态、BF16 仅 forward、FP32 CE、单组 AdamW(.9,.95)/eps1e-8/WD.1/clip1。
原85片 seed0 shuffle后连续分给八rank、片内 buffer4096/片尾不足8丢弃、动态padding/context1024/不packing保持。
每次名义额度9,465,856样本，不保证严格完整数据遍历恰好一次；每500步及最终保存，保留8500/9000/9244。
显式 --max-lr/--min-lr，min默认0.1×max；epoch1为8e-4/8e-5，epoch2为2e-4/2e-5。
用户手动以 --model-dir 指向 epoch1 的完整 checkpoint-step-009244 开始 epoch2，
只加载模型，不读 training_state.pt；优化器/scheduler/RNG/计数/数据游标重置，seed0 从头读取。
两个 epoch 各自 min→max warmup，再cosine回min，不合并成18488步scheduler。
同epoch --resume-from 要求本路线完整checkpoint和相同LR，累计到9244，使用新空输出目录。
恢复继承原母本限制：保存八rank游标但无shuffle buffer，模型RNG仅rank0快照，不声称八rank bitwise exact。

正式入口直接 vc submit，无CPU数据扫描/Gate B/smoke前置；沿用 pdgpu-5090/32CPU/256G/8GPU/1node、原容器和rsmol环境。
新路线模型/输出/log/job/环境变量隔离，转换 CONVERT_OUTPUT_DIR 和训练 OUTPUT_DIR 分开。
远程仓库根目录 /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM 同步后：

```bash
bash code/RSmol/scripts/convert_mizar_text_5_10x2_5.sh
bash code/RSmol/run_stage4_mizar_text_5_10x2_5_training_5090.sh --max-lr 8e-4 --min-lr 8e-5
```

README含固定epoch1/epoch2输出、手动第二epoch、完整恢复、执行和精确git add/commit/push命令。
python -B tests/validate_mizar_text_5_10x2_5.py：28项本地检查通过，5项真实CPU tensor因缺Torch等依赖跳过。
覆盖母本SHA、完整更新循环/数据/损失AST、所有9245点LR、最终权重初始化和错路线拒绝、
实际FORMAL启动不加载旧训练状态、原micro8片尾规则、Bash vc→torchrun转义/资源/显式LR/失败传递。
报告 audit_reports/mizar_text_5_10x2_5_validation.json；远程rsmol可选加 --require-tensors 验证合成CPU张量。
Windows沙箱拒绝临时目录后，经授权在沙箱外运行同一本地验证。
未代为git commit/push、读取远程生产资产、执行生产转换或提交GPU作业，不登记远程转换/训练/恢复PASS。
本条仅前置项目记录，原RM内容完整保留。

## 2026-11-09：新增两条隔离的固定三循环文本完整一轮训练

按用户确认方案落代码：纯5＋10×3＋5，以及5＋10×3＋5 MeSH六槽、四组write/read routers（8个独立Linear模块）。
两条均20物理层、40逻辑执行/缓存槽，中间十层共享参数执行三次；MeSH每forward重建[B,6,L,576]激活，
槽0原始embedding、其余五槽零，prelude query/value均用prelude输出，三轮query用各轮输入、value用各轮输出。
8个Linear(576,6,bias=True)共16个tensor/27,696参数；router softmax FP32，诊断非致命，无辅助loss。
继续本地Windows编辑核心代码、GitHub同步、远程Linux保存全部权重/数据和运行任务。

新增模型recursive_model_5_10x3_5.py、recursive_model_5_10x3_5_mesh_6slot.py，
各自独立convert_stepwise、train_stage4_ddp、CPU转换shell、vc FORMAL提交器和完整恢复入口。
默认干净models/SmolLM2-5-10-5二十层直接继承；亦支持原始models/SmolLM2三十层固定20层映射。
不从旧训练checkpoint或已有MeSH/新递归路线初始化；转换CPU离线8线程，拒绝覆盖，验证继承tensor、
40次forward、保存重载tensor/logits/tokenizer后发布，失败staging保留。两条使用专用模型类，无全局AutoModel注册调用。

两条训练均独立复制七槽v2较新完整epoch母本：8×micro4×GA32=GBS1024，9244updates、9244scheduler、463warmup、
epochs1/context1024/seed0/maxLR8e-4/minLR8e-5；FP32参数和AdamW状态、BF16仅forward、FP32 CE、
全局有效token归一化梯度、clip1、无GradScaler/TF32。AdamW(.9,.95)/eps1e-8，矩阵WD.1、bias/norm/低维WD0。
两条统一85片按文件名排序后index%8分rank、第一轮顺序读取、动态padding、不packing；沿用母本shard尾不足batch保留规则。
此读取和decay分组与历史x2纯文本不同，已按用户确认统一两条新路线。
项目一轮名义额度9244×1024=9,465,856，不声明严格遍历全部有效文本恰好一次。

正式入口直接vc submit：pdgpu-3090/8GPU/64CPU/256G/1node，不加CPU数据扫描、Gate B或smoke前置门槛。
模型目录/训练输出/日志/job/env/架构和训练合同完全分开；转换CONVERT_OUTPUT_DIR与训练OUTPUT_DIR分开。
每500步和最终保存，保留8500/9000/9244三个完整checkpoint；完整恢复要求同路线同FORMAL配置，
保存全部rank游标/Python/CPU/CUDA RNG、optimizer、scheduler、manifest，使用新空输出目录累计训练到9244。

纯路线说明：code/RSmol/text_5_10x3_5/README.md；MeSH路线说明：code/RSmol/text_5_10x3_5_mesh_6slot/README.md。
文档包含CPU转换、vc正式提交、同路线恢复、远程CPU验证和精确Git add/commit/push命令。
远程同步后，分别执行（仓库根目录/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM）：

```bash
bash code/RSmol/scripts/convert_stepwise_5_10x3_5.sh
bash code/RSmol/run_stage4_5_10x3_5_formal_3090.sh
bash code/RSmol/scripts/convert_stepwise_5_10x3_5_mesh_6slot.sh
bash code/RSmol/run_stage4_5_10x3_5_mesh_6slot_formal_3090.sh
```

python -B tests/validate_text_x3_routes.py：26项本地合同/游标/Bash模拟提交检查通过，9项真实CPU tensor因缺Torch等依赖跳过。
含母本十文件归一化LF SHA保护、attention/cache/MeSH读写AST继承、两路线读取/更新主体一致性、
全9245点LR曲线、错路线/错配方拒绝、实际row cursor跳过、vc环境转义和互不串线等检查。
报告audit_reports/text_x3_routes_validation.json；远程rsmol可加--require-tensors跑合成CPU转换/梯度/cache/训练/恢复验证。
临时目录/Bash测试因沙箱权限失败后，经授权在沙箱外执行相同本地验证。
未代为git commit/push、读取生产权重/数据或提交真实GPU作业，不登记远程转换/训练/八rank恢复PASS。
原母本不变，工作区既有Mizar评测改动未覆盖；本文仅前置本条记录，原内容和用户既有修改完整保留。

## 2026-11-09：三层音频CA＋input重注入、官方input-only的MMAU/MMAR评测

原两个CED评测入口已扩展至26条交互/MeSH/官方循环路线，见
[精确产物映射、协议和四条提交命令](code/RSmol/scripts/README_mizar_s1_ced_multilayer_eval.md)。
继续本地Windows维护核心代码、GitHub同步，权重/数据与GPU作业均在远程Linux。

新增recursive_5_10_5_layer4812_crossattn_audio_reinject绑定formal_20261109_005039335295104的step17313；
短训练目录in_ca4812，20物理/30执行，CED单次编码提取raw4/8/12，逻辑5/15执行input→音频CA，
完整加载4,334,405参数/32份分支tensor，三个audio banks和input投影每完整LM forward各一次。
新增official_final_s1_5_20x2_5_input_only绑定formal_20261107_220147075105401的step34626；
独立input-only模型、30物理/50执行，逻辑5/25只注入input，完整训练state458份tensor；
GPU可用runtime构建后必须用训练checkpoint覆盖初始化bundle，不加载LoRA/中间层分支。

评测侧规范化两项/三项prefix返回值，增加只读编码/真实交互/投影/物理层调用hooks；
原训练、转换与模型源码及其生产SHA合同不改。精确选择对应step和同run的formal/runtime.json，
同目录中间checkpoint不参与默认选择。训练geometry分别8,8,4与8,4,4；评测仍单GPU/batch1。
原CED协议、数据及选项顺序、32k/fixed20、256prompt/383prefix/首logit382、FP32/seed1234/TF32关闭、
mask=None/无KV/完整重算argmax/max300/EOS、原始输出/官方scorer/1000分母/零skip保持。

python -B tests/validate_mizar_s1_ced_multilayer_eval.py：317项实际本地检查通过，
39项真实CPU tensor检查缺Torch/Transformers跳过（本次新增4项）。覆盖两路线实际合同与加载顺序、
CA真实源码NumPy数值行为、完整52种千条模拟流程、原续跑/评分/旧分支回归及Bash假提交。
报告audit_reports/mizar_s1_ced_multilayer_eval_validation.json；--require-tensors要求无张量跳过。
Windows沙箱临时目录受限后，经授权在沙箱外运行相同本地流程。
未执行git commit/push、远程checkpoint/runtime读取或真实GPU作业提交，不登记远程评测PASS。

## 2026-11-09：修复纯文本5＋20×2＋5启动，取消新增CPU数据审计

用户远程日志已完成模型forward_trace_done，随后被初版新增的
validate_cpu_audit中的“CPU audit dataset directory mismatch”阻断，尚未进入第一步训练。
该判断比较历史报告data_dir与当前容器目录，不能据此认定实际数据不匹配；原5+10×2+5文本母本无此检查。
按用户要求撤掉新增流程，继续本地Windows改核心代码、GitHub同步、远程Linux保存资产和执行训练。

正式launcher现在和原文本提交器一样直接vc submit，不再调用prepare_text_5_20x2_5.py，
不自动运行Gate B、扫描85片footer、分词三个sample shard或比较报告目录/tokenizer文件。
训练默认无需--audit-report；使用原discover_parquet_files/assign_shards，从当前数据目录或其data/子目录
发现85个shard并按seed0分给八卡，然后直接进入原streaming训练。旧报告若显式提供仅存历史记录，
不参与路径/容量匹配、不提供训练文件路径；缺失/损坏说明后继续。当前dataset_audit明确NOT_RUN，
不将历史报告记成本次审计PASS。手动prepare旧命令仅检查模型配置，不再生成或要求CPU报告。

模型/转换文件未改；30物理/50执行、8×micro4×GA16=GBS512、18488updates/925warmup，
FP32参数/BF16前向/FP32 CE、全局有效token加权loss、AdamW及完整更新主体保持。
本次失败无已训练checkpoint需要恢复，既有转换模型直接复用，不必再次转换；同步修复后执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_stage4_5_20x2_5_training_5090.sh
```

python -B tests/validate_text_5_20x2_5.py：50项本地检查通过，8项真实CPU tensor因缺依赖跳过。
新增实际FORMAL启动控制流回归（替换GPU/模型执行），无报告/旧报告异挂载路径均从forward_trace_done进入run_training；
八rank manifest广播、可选报告缺失/损坏、原85片发现/分配、Bash直接vc→torchrun无CPU preflight均通过，
母本SHA和原更新/读取/损失/attention/cache AST对照通过。报告audit_reports/text_5_20x2_5_validation.json。
按既有授权在沙箱外运行临时目录/Bash测试；本次未读取远程资产、提交真实作业或验证GPU训练完成。
以下初版记录的自动CPU审计/匹配流程由本条修复取代，最新用法见[README](code/RSmol/text_5_20x2_5/README.md)。

## 2026-11-09：纯文本 SmolLM2 5＋20×2＋5，GBS512 正式训练

按已确认计划实现独立非MeSH纯文本路线，见[转换、正式训练与恢复说明](code/RSmol/text_5_20x2_5/README.md)。
继续保持本地Windows只改核心代码、GitHub中转、远程Linux保存权重/数据和执行训练。
母本为run_stage4_5_10_5_5090.sh；原training_5090同名入口只是该母本的兼容别名。
原模型/转换/训练/CPU审计等10个文件SHA256记录于text_5_20x2_5/source_guard.json，母本均未修改。

从远程models/SmolLM2原始30层初始化，源层0..29一一继承，不抽层、不平均。
新recursive_model_5_20x2_5.py保存30物理层，执行0..4、5..24、5..24、25..29，共50次。
两轮中间20层复用同一参数；KV分50逻辑槽，前后5层只执行一次。
embedding/final norm/lm_head/权重绑定/tokenizer/RoPE与attention配置继承；全量训练，无MeSH/router/memory/LoRA/音频。
转换器CPU离线8线程，拒绝已有输出和递归/训练源；严格tensor继承、源模型单遍30层一致性、50次trace、
保存重载权重/logits/tokenizer检查后发布。失败staging保留诊断，显式专用模型类加载，不改全局AutoModel注册。

8×micro4×GA16=GBS512，18,488updates/18,488 scheduler domain/925warmup，
保持9,465,856样本预算（原9,244×1,024）。max/minLR8e-4/8e-5，原FP32参数/BF16 autocast/FP32 CE，
context1024、单组AdamW(0.9,0.95)/eps1e-8/WD0.1/amsgrad false、clip1.0、seed0/buffer4096/workers0保持。
原85片分配/片内shuffle/tokenize/padding及完整更新主体AST对照一致；
文本loss仍为跨所有rank与16次累积的有效token总数加权，无额外GA除法；不采用音频microbatch均值损失。
每片末尾不足microbatch仍丢弃；micro4可改变片尾丢弃数，因此保持样本预算与读取规则，不保证逐样本/token数完全一致。
每500步及最终保存，保留最新3个完整checkpoint；正常最终保留17500、18000、18488。

远程同步本地代码后，在登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash scripts/convert_stepwise_5_20x2_5.sh
bash run_stage4_5_20x2_5_training_5090.sh
```

转换默认models/SmolLM2-5-20x2-5；训练默认outputs/RSmol/text_5_20x2_5_gbs512/formal_<timestamp>_<pid>/formal/。
可用--audit-report复用兼容原CPU Gate B报告，否则先运行原CPU审计再提交一个FORMAL；
检查85片路径/大小/行记录、tokenizer当前文件一致性、context与数据目录，重建seed0八卡分配并核对正式raw容量。
采样内容不冒充全量有效行数证明。无自动GPU smoke，显式FORMAL，不会沿旧默认进入十步Gate D。
独立模型/输出/log/job/tmp和RSMOL_5_20X2_5_环境变量；原rsmol、原容器、pdgpu-5090/32CPU/256G/8GPU/1node。
最终formal/checkpoint-step-018488/；--resume-from指定本路线FORMAL完整checkpoint，--output-dir用新的空外部目录。
恢复模型/optimizer/scheduler/计数/manifest/各rank cursor，拒绝旧5+10×2+5、MeSH、Gate D/E和不同配方。
沿用文本母本限制：bounded shuffle不保存buffer，模型Python/Torch/CUDA RNG只存rank0，
并非八rank独立RNG恢复；bounded_shuffle_bitwise_exact=false，不承诺逐样本连续或bitwise exact。

python -B tests/validate_text_5_20x2_5.py：45项实际本地检查通过，8项真实CPU tensor检查缺依赖跳过。
覆盖母本SHA、原更新/损失/数据及模型forward/attention/cache AST、50槽实际helper、全部18,489个LR位置、
配方/恢复拒绝、真实micro4迭代/审计复用，以及实际Bash假vc/conda/torchrun转换/正式/恢复/失败传递。
Windows沙箱临时目录受限后，经授权在沙箱外运行相同测试；报告audit_reports/text_5_20x2_5_validation.json。
手动--require-tensors要求真实tensor用例无skip；未执行git commit/push、远程生产资产读取、模型转换或GPU作业提交，
不登记生产转换、八卡训练或八卡恢复PASS。

## 2026-11-08：非MeSH CED4/8/12，input后串行音频CA正式训练

按用户确认方案从原layer36912_crossattn完整复制隔离，串行input重注入参考8→4路线。
见[架构、正式提交与完整恢复](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_recursive_5_10_5_ced_small_in_ca4812/README.md)。
本地Windows维护核心代码、GitHub中转，权重/数据/正式GPU作业均在远程Linux。
内部in_ca4812短路径兼容Windows；独立route/launcher/output/job/log/index-cache/tmpfs/resume身份。

CED冻结/eval，单次编码提取raw block4/8/12（零基3/7/11，final LN前）。
各自保留affine LN1536→Linear1536→576→affine LN576；三个bank每完整forward计算一次。
原final mapper及20物理/30逻辑、5+10×2+5 schedule保持，无MeSH/memory/LoRA。
原input音频E0经共享Linear576→576每forward只投影一次，不detach。
逻辑5/15分别先用raw pre-input audio hidden的tanh gate注入input，再由post-input hidden
逐音频token生成126组三层softmax，同时间混合三层作K/V；Q为LN(post-input audio hidden)，
CA gate读取raw post-input hidden。MHA576/9heads/dropout0，每Q可看126音频K/V。
两次残差都只写回0..125；SEP126及文本位置保留。删除轮次偏置、CA输出Norm及残差系数。
全部新增模块两处共享；两个残差gate和depth gate零初始化；CA输出weight非零，CPU RNG隔离。
相对普通CED循环母本新增4,334,405参数/32份tensor，全部新增及mapper maxLR6e-4，原LM3e-4。

原hybrid1,477,492行/fixed20/683token/FP32/无autocast/TF32关闭/GradScaler保持；
8×micro8×GA4=GBS256，3epochs/17313updates/866warmup；原AdamW/LambdaLR/GradNorm、
answer-only rank-local token mean→GA→DDP、数据顺序、共享启动/默认4workers后台复制、
optimizer边界全rank MIN切换、先reap后清理、完整checkpoint及八rank RNG恢复合同保持。
首步审计两残差gate，开门后逐分支梯度审计；记录两轮depth权重和input/CA残差RMS。

本地新增实现提交推送GitHub后，远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4812_crossattn_audio_reinject_5090.sh --stage-workers 4
```

直接一个formal，无自动smoke/tensor前置作业；原容器/mellow_c8204d8，pdgpu-5090/32CPU/256G/8GPU/1node。
默认outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer4812_crossattn_audio_reinject_5090/formal_<timestamp>/formal/，
最终checkpoints/model--step-17313.ckpt及model--epo-3.ckpt；完整恢复仅同路线/源码/资产/数据/依赖/配方及原run root。
原36912与串行8→4checkpoint不可full resume；既有MMAU/MMAR evaluator需另行适配新合同/state_dict。

python -B tests/validate_mizar_s1_ced_ca4812_audio_reinject.py：118项实际本地检查通过，
39项真实CPU tensor检查因缺Torch等依赖跳过（新路线12项）。覆盖原文件SHA保护、复制源码inventory、
骨干/mapper/预处理/整个更新循环AST、实际NumPy逐token混合/raw gate输入/音频写回/CA幅度/因果性、
LR与模块/梯度失败审计、真实Bash模拟提交/共享启动/后台复制/集体切换/清理及旧路线回归。
Windows沙箱临时目录受限后，经授权在沙箱外运行同流程；报告audit_reports/mizar_s1_ced_ca4812_audio_reinject_validation.json。
手动--require-tensors要求无skip。未执行git commit/push、生产资产读取、GPU训练、八卡恢复或作业提交，不登记远程PASS。

<!-- CA4812_RECORD_END -->
## 2026-11-08：七条新增训练产物的 MMAU test-mini / MMAR 评测适配

已扩展既有两个CED评测提交入口，详细映射、精确默认run和14项提交命令见
[二十四路线评测说明](code/RSmol/scripts/README_mizar_s1_ced_multilayer_eval.md#2026-11-08新增七条路线共14项评测)。
保持本地Windows改代码、GitHub同步、远程Linux保存权重/数据与执行评测。

新增variant为recursive_mesh_4slot、recursive_mesh_6slot、recursive_mesh_7slot_v2，
非MeSH串行8→4的loop_lora_r4、loop_attn_shared_mlp_modality_lora_r8、
loop_attn_shared_mlp_shared_modality_lora_r8，以及official_final_s1_5_20x2_5。
前三条指向用户提供的3090产物run；前六条精确读取step17313，官方版本step34626，
同run配套formal/runtime.json，不搜索latest或中间checkpoint。

直接复用训练模型，严格加载完整音频checkpoint；4/6/7槽为全序列memory，
七槽v2保留最终9244步文本来源校验。两种Attention共享LoRA保留不同family布局，
MLP共享+模态独立分支各scale1并行相加；rank4为alpha8/scale2。
官方路线独立GPU-safe加载与manifest审计后严格覆盖1038份训练state，
保留30物理层/50次执行、逻辑5/25的串行8→4和两轮r8 LoRA。
训练源码、共享模型和生产身份SHA检查均未修改。

原fixed20、383-token prefix、FP32/seed1234/TF32关闭、无KV cache、完整重算argmax、
最多300新token/首次EOS停止及原始输出直接交官方scorer保持；各benchmark完整1000条。
仍提交pdgpu-5090/8CPU/32G/1GPU/1node、mellow_c8204d8。

python -B tests/validate_mizar_s1_ced_multilayer_eval.py：293项本地检查通过，
35项真实CPU tensor检查因缺Torch/Transformers跳过（新增3项）。
覆盖七条实际合同、全部跨路线拒绝、严格加载/冻结顺序、逐步执行与LoRA family验证、
实际源码NumPy非零LoRA与缩放/基础投影一次、14项Bash假提交及checkpoint覆盖，
全部24个variant双benchmark的1000条合成流程，以及旧路线回归。
Windows沙箱临时目录/Bash受限后，在获准的沙箱外执行完整校验；报告
 audit_reports/mizar_s1_ced_multilayer_eval_validation.json。
--require-tensors要求真实张量检查无跳过。未执行git commit/push、远程权重/runtime读取或评测提交，
未登记远程GPU评测PASS。

## 2026-11-07：非MeSH CED8→4，Attention共享＋MLP共享/模态独立并行LoRA

按用户确认计划从Attention共享、MLP模态独立r8路线完整复制隔离，见[正式训练实现、提交与完整恢复说明](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_recursive_5_10_5_ced_small_in84_loop_attn_shared_mlp_both_lora_r8/README.md)。
本地Windows维护代码，经GitHub同步；权重、数据和真实GPU训练均在远程Linux。
内部in84/mlp_both缩写兼容Windows路径长度，both表示shared＋modality，正式launcher保留完整名称。

仅中间十个物理层5..14的q/k/v/o保持Audio/Text shared rank8/alpha16/scale2；
gate/up/down改为shared＋Audio private＋Text private三条独立LoRA，各rank8/alpha8/scale1。
每个MLP token使用共享与自身private两条残差，从同一投影输入并行相加，基础Linear仅一次。
显式系数之和1＋1=2，不保证独立残差实际幅度或更新量与原scale2单分支相同。
不同层/投影/轮次A/B均独立，原W/bias共享且全量训练，前后五层、mapper/音频MLP/input/gates不新增LoRA。
固定Audio0..125、Text126起含SEP；无模态学习router，原causal attention/RoPE/SwiGLU保持。

Attention614,400；MLP shared/audio/text各1,013,760，LoRA共3,655,680，较直接母版多1,013,760。
260对A/B即520份独立tensor；加原串行音频分支2,777,859，相对普通CED循环母本共新增6,433,539。
合同按family记录alpha/scaling，优化器以target＋branch区分attention_shared与mlp_shared；共24角色，8组LoRA A/B梯度审计。
每完整forward重置调用/形状/分支计数，要求140基础投影、260 LoRA分支和正确scope复位。

保持非MeSH20物理/30逻辑、prelude后input→CED8、loop1后input→CED4、loop2后直接suffix。
每forward一次input projection，CED gate仍读post-input hidden，CED冻结/eval及原mapper保持。
strict原非MeSH文本checkpoint-step-009244初始化；A Kaiming/B零/隔离CPU RNG；基础与音频初值及调用者RNG不被LoRA初始化扰动。
骨干/loader整个AST与母版一致，train更新函数整个AST相等；原hybrid1,477,492、683-token、FP32/无autocast/TF32关闭，
GradScaler、8×micro8×GA4=GBS256、3epochs/17313updates/866warmup，LM maxLR3e-4及LoRA/mapper/音频分支6e-4，
AdamW/LambdaLR/GradNorm、数据顺序、共享启动/后台staging、checkpoint及完整恢复流程保持。

本地代码通过GitHub提交推送后，远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_shared_modality_lora_r8_5090.sh --stage-workers 4
```

独立训练目录/launcher/route/output/job/log/index-cache/tmpfs/resume身份，直接提交一个formal，无自动smoke/tensor前置任务。
沿mellow_c8204d8、原容器、pdgpu-5090/32CPU/256G/8GPU/1node。
最终本run下formal/checkpoints/model--step-17313.ckpt及model--epo-3.ckpt。
full resume仅接受本路线同源码/资产/数据/依赖/配方和原run root，直接母版即使private矩阵形状一致也因缺MLP shared及alpha不同而拒绝。
现有MMAU/MMAR evaluator需为本新合同/state_dict单独适配。

python -B tests/validate_mizar_s1_attn_shared_mlp_both_lora_r8.py：152项实际本地检查通过，45项真实CPU tensor检查缺依赖跳过（新版10项）。
Python/Bash语法、257原文件SHA保护、20复制源码除命名外一致、整个骨干/loader/train更新函数AST及原配方对照通过。
实际源码NumPy并行两分支/scale1/base一次、共享MLP双模态梯度相加、125/126与轮次隔离、有限差分、post-input gate、
24角色/全LR/8组梯度审计失败、分支调用/scale/边界/scope拒绝、Bash模拟提交/参数转发及shared/staging/失败清理接口通过，含直接母版和原r8两顺序回归。
此前已确认Windows沙箱临时目录/Bash限制，本次完整流程经批准在沙箱外执行；报告audit_reports/mizar_s1_attn_shared_mlp_both_lora_r8_validation.json。
手动--require-tensors禁止skip；未执行git commit/push、远程生产资产读取、真实GPU训练或八卡恢复，不登记远程PASS。

## 2026-11-07：非 MeSH CED8→4，两轮独立 LoRA rank4 / alpha8

按用户确认计划，从原rank8/alpha16完整复制隔离，见[正式训练、提交与完整恢复说明](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r4/README.md)。
保持本地Windows维护代码、GitHub中转、远程Linux保存权重/数据并执行真实训练。
新训练目录/launcher/route/output/job/log/index-cache/tmpfs/resume身份独立，原r8/r16及其他路线保持。

仅将中间十个物理层5..14七个投影的LoRA改为rank4、alpha8，scale仍2、dropout仍0。
q/k/v/o及gate/up/down每层每投影各有两轮独立A/B；同轮Audio/Text共享该投影LoRA，基础W/bias共享且全量训练。
140对A/B即280份独立tensor，实际GQA形状下两轮LoRA共814,080参数，较r8少814,080；
原串行音频分支2,777,859保持，相对普通CED循环母本共新增3,591,939。
A Kaiming/B零及隔离CPU RNG保持，严格加载原非MeSH文本checkpoint-step-009244后注册LoRA和原音频模块。

保留20物理/30逻辑、非MeSH无memory/router；prelude后input→CED8、loop1后input→CED4、loop2后直接suffix。
每forward一次共享input projection，CED gate仍读取post-input hidden。train/optim/骨干/文本loader完整AST与r8相等。
原hybrid1,477,492、683-token、FP32/无autocast/TF32关闭、GradScaler、8×micro8×GA4=GBS256，
3epochs/17313updates/866warmup，基础LM maxLR3e-4及LoRA/mapper/音频分支6e-4，
原AdamW/LambdaLR/GradNorm、12类梯度角色、数据顺序、共享启动/后台staging、保存与完整恢复保持。
完整恢复仅接受本r4同源码/资产/数据/依赖/配方checkpoint和原run root，原r8/r16形状不同不能full resume。

本地代码提交并推送GitHub后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r4_5090.sh --stage-workers 4
```

直接提交一个formal，无自动smoke/tensor前置任务；沿mellow_c8204d8、原容器、pdgpu-5090/32CPU/256G/8GPU/1node。
默认outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r4_5090/formal_<timestamp>/，
最终formal/checkpoints/model--step-17313.ckpt及model--epo-3.ckpt。

`python -B tests/validate_mizar_s1_ced_loop_lora_r4.py`：108项实际本地检查通过，33项真实CPU tensor检查缺依赖跳过（新版7项）。
Python/Bash语法、218原文件SHA保护、26复制源码受限差异、22源码除隔离命名外一致、完整骨干/loader/train/optim AST与训练配方对照通过。
实际源码NumPy非零LoRA、轮次/scale2/有限差分/post-input gate、全部12类LR/梯度审计、跨rank恢复拒绝、
真实Bash模拟提交/参数转发、共享启动/后台复制/失败清理/集体切换，以及原r8两顺序和串行母本回归通过。
Windows沙箱临时目录/Bash受限后，获准在沙箱外同流程复验通过；报告audit_reports/mizar_s1_ced_loop_lora_r4_validation.json。
手动加--require-tensors可要求真实CPU tensor用例无skip；不作为formal前置。未执行git commit/push、远程生产资产读取、GPU训练或八卡恢复，不登记远程PASS。
本r4/alpha8新合同的MMAU/MMAR评测器需另行适配。

## 2026-11-07：非 MeSH CED8→4，Attention共享/Transformer MLP模态独立的两轮LoRA

按已确认计划从`run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8_5090.sh`
完整复制隔离，见[正式训练实现、提交与完整恢复说明](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_recursive_5_10_5_ced_small_in84_loop_attn_shared_mlp_modality_lora_r8/README.md)。
保持本地Windows改核心代码、GitHub中转、远程Linux保存权重/数据和执行GPU作业的方式。
内部目录用in84缩写兼容Windows路径长度，正式launcher保留完整架构名称；原训练与评测代码保持。

仅中间十个物理层5..14加LoRA：q/k/v/o每投影每轮Audio/Text共享一对A/B，
gate/up/down每投影每轮Audio/Text分别独立；不同层、投影和轮次的LoRA不共享。
全部rank8/alpha16/scale2/dropout0，基础W/bias仍共享且全量训练，前五/后五层及其他模块不新增LoRA。
MLP位置0..125选audio，126起（含SEP、prompt、answer/PAD/生成）选text；无学习模态router，完整causal attention/RoPE保持。
Attention614,400、MLP audio1,013,760、MLP text1,013,760，LoRA共2,641,920参数，
200对A/B即400份独立tensor，比母本多1,013,760；相对普通CED循环母本加串行分支共新增5,419,779。

保持20物理层/30次执行、非MeSH无memory/router、每forward一次input projection、
prelude后input→CED8、loop1后input→CED4、loop2后直接suffix；CED gate仍读取post-input hidden。
原CED冻结/eval、mapper/音频MLP/gates/完整LM可训练，strict原非MeSH文本checkpoint-step-009244初始化，A Kaiming/B零及隔离CPU RNG。
683-token、hybrid1,477,492、FP32/无autocast/TF32关闭、原GradScaler、8×8×4=GBS256、3epochs/17313updates/866warmup，
LM maxLR3e-4及全部LoRA/mapper/音频新增模块6e-4，原AdamW/LambdaLR/GradNorm/数据顺序/共享启动及后台staging/保存恢复保持。
20类参数角色分别审计两轮Attention shared/MLP audio/MLP text的A/B梯度、参数变化和完整保存；
每forward重置轻量调用元数据，要求140次基础投影/200次LoRA分支、正确模态边界与finally scope复位。

本地代码提交推送GitHub后，远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_modality_lora_r8_5090.sh --stage-workers 4
```

直接提交一个formal，无自动smoke/tensor前置任务；沿mellow_c8204d8、原容器、pdgpu-5090/32CPU/256G/8GPU/1node。
训练目录/route/output/job/log/index-cache/tmpfs/resume身份独立；最终formal/checkpoints/model--step-17313.ckpt及model--epo-3.ckpt。
full resume仅接受本路线同源码/资产/数据/依赖/配方，旧两套/四套LoRA或MeSH/textmesh checkpoint不能直接完整恢复；
现有MMAU/MMAR evaluator需另行适配本合同和新state_dict。

`python -B tests/validate_mizar_s1_attn_shared_mlp_modality_lora_r8.py`：113项实际本地检查通过，35项真实CPU tensor检查缺依赖跳过（新版9项）。
覆盖Attention共享两段梯度相加、MLP125/126边界/有限差分/轮次、真实源码循环及post-input gate、梯度/调用拒绝、
真实Bash假提交/参数转发/共享启动/失败清理/集体切换和原母本回归；Python/Bash语法、146原文件SHA保护、
19复制源文件除命名外一致、完整骨干/train受限AST对照及原数据/更新/保存恢复合同通过。
报告`audit_reports/mizar_s1_attn_shared_mlp_modality_lora_r8_validation.json`；手动加`--require-tensors`禁止skip。
未执行git commit/push、远程生产资产读取、真实GPU训练、八卡恢复或作业提交，不登记远程PASS。

## 2026-11-07：官方 Mizar final，仅 input 重注入，5+20×2+5 CPU转换与GBS128正式训练

按用户已确认计划，从官方完整Mizar循环模型/训练器复制隔离实现；见[CPU审计与转换](code/RSmol/mizar_official_recursive_5_20x2_5_input_only/README.md)及[正式训练与恢复](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_official_final_s1_gbs128_5_20x2_5_input_only/README.md)。
保持本地Windows只改核心代码、GitHub中转、远程Linux保留权重/数据与执行的工作方式。
原serial84+LoRA模型、转换与训练代码不改；新模型/训练目录、route/output/log/job/index-cache/tmpfs和resume身份独立。

来源为完整官方models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt及其assets，
不从旧LoRA转换产物拆权重。保留30物理层、prelude0–4、中段5–24共享权重循环两次、coda25–29，
执行0..24,5..24,25..29共50次。只保留共享Linear(576,576) input projection和Linear(576,1) gate：
原final mapper输出A0只投影一次R，在逻辑5/25各自循环前H_audio += tanh(gate(H_audio))*R，
仅直接更新音频0–125；gate每轮读取当前hidden，R不detach。projection weight沿母本初始化、bias0，gate weight/bias0，
保留input初始化分配顺序及隔离CPU RNG。无LoRA、encoder中间层注入/MLP/gate/capture、memory或router。
CED冻结/eval，每音频批次只提取一次最终特征；原已训练mapper与完整LM继续训练。
新增332,929参数，总独立参数159,632,769；454原state加4份新增，共458份。

转换入口CPU-only/offline/8threads，直接终端运行、不提交GPU作业、不读训练数据；
保留完整来源SHA、454份strict继承、单遍30层对齐、合成fixed20音频683token完整50次前向，
保存后458份tensor精确重载及683token logits精确相等，最后原子发布；含CloudStor发布兼容及publish续发布。
默认新产物models/Mizar159M-5-20x2-5-input-reinject；已有目录拒绝覆盖。

本地提交并推送GitHub后，远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5_input_only.sh convert --seed 0 --threads 8
```

转换成功后正式提交：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_official_final_s1_gbs128_5_20x2_5_input_only_5090.sh --stage-workers 4
```

训练strict加载全部458份state，GPU loader不调用会清空CUDA的CPU转换入口。
循环基础参数（含norm）和input新增参数maxLR6e-4；原mapper、prelude/coda、embedding/最终norm/lm_head为3e-4。
原S1 hybrid1,477,492条、3epochs、8×micro4×GA4=GBS128；11542updates/epoch、34626总步、1731warmup。
原FP32前向、无autocast/TF32、GradScaler/AdamW/LambdaLR/GradNorm、683token、answer-only CE和模板均保持。
loss仍为rank-local microbatch有效token均值→GA均值→DDP均值。首步gate有梯度/投影task梯度零，第二步检查投影梯度解锁。

沿母本initial_steps=0，计算节点只等本次owner/path metadata；共享原始音频立即训练，
单producer/默认4workers后台复制raw bytes，全部文件/字节通过后八rank在optimizer边界、取microbatches前
all_reduce(MIN)统一切到节点/dev/shm。producer异常中止，先停止/reap后只清理本token独立tmpfs。
半epoch保存5771/11542/17313/23084/28855/34626及epoch3final，完整保存八rank RNG/游标/optimizer/scheduler/scaler/GradNorm。
full resume只接受本路线同代码/依赖/资产/数据/配方、原output root；每次恢复重建staging并重新shared启动。
正式入口直接提交一个formal，无自动smoke/tensor前置任务；沿pdgpu-5090/32CPU/256G/8GPU/1node、原容器、mellow_c8204d8。
默认outputs/RSmol/mizar_official_final_s1_gbs128_5_20x2_5_input_only_5090/formal_<timestamp>/formal/。

python -B tests/validate_mizar_input_only.py：63项实际本地检查通过，12项真实CPU tensor检查缺依赖跳过。
含45原文件SHA保护、复制数据/工具字节一致、完整microbatch更新/保存RNG/GradNorm AST、所有34627个LR位置、
resume拒绝、被阻塞真实copier线程共享启动及完成切换、真实Bash假提交/CPU wrapper/计算入口先训练后复制完成/先reap后清理。
Python/Bash/LF检查通过；报告audit_reports/mizar_input_only_validation.json，--require-tensors禁止skip。
Windows沙箱临时目录/Bash受限后，沙箱外同流程复验通过。未执行commit/push、远程生产资产读取/转换、GPU训练、八卡恢复或作业提交，不登记远程PASS。

## 2026-11-07：七槽 v2，两阶段隔离转换、BF16文本训练与CED-small音频训练

按用户确认计划从当前四槽两阶段实现复制隔离，见[七槽v2转换、文本正式训练与音频衔接](code/RSmol/text_5_10x2_5_mesh_7slot_v2/README.md)和[七槽v2音频正式训练与恢复](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_7slot_v2_ced_small/README.md)。
本地仅维护代码，GitHub中转；权重、数据、CPU生产转换和真实GPU作业均在远程Linux。
原四/六槽、历史七槽3081步路线、音频七槽/文本五槽特征槽实验及评测器保持原样。
新类为RecursiveLlama5_10x2_5Mesh7SlotV2ForCausalLM，架构合同logical_30_physical_20_5_10x2_5_mesh_7slot_v2_3router，
新模型/源码/环境变量/训练合同/output/job/log/index-cache/tmpfs及恢复身份均独立。

hidden576、20物理/30逻辑、5+10×2+5、两轮中段共享保持；每forward重建[B,7,L,576]，
slot0原始embeddings，其余六槽零。三组write/read、六个独立Linear(576,7)、12份参数共24,234；
prefix output-query/value和loop input-query/output-value、原读写公式保持，初始化std=sqrt(2/(7×hidden))。
转换默认从干净models/SmolLM2-5-10-5，亦接受原30层SmolLM2；保留30→20映射，20层来源不再次抽层。
逐tensor继承核对、tied alias与tokenizer保持，只新建七槽routers；CPU/offline，默认拒绝已有目标/训练态来源/checkout模型输出。
默认产物models/SmolLM2-5-10x2-5-mesh-7slot-v2。

文本为FP32参数/AdamW状态+BF16 autocast前向，FP32 loss/router softmax，无GradScaler、TF32关闭，
CE/全局token归一化/backward在autocast外；必须CUDA BF16，CPU只允许Gate A诊断。
沿四槽8×micro8×GA16=GBS1024、context1024、完整一轮9244步/463warmup、LR8e-4→8e-5、原AdamW/clip1。
原Parquet/shard/行序和全局有效token归一化保持，每500步及最终保存、保留三份；完整各rank游标/RNG恢复。
正式入口直接提交，提供独立可选10+2诊断，不作为正式前置门槛。

音频入口必须显式--text-model-dir指向本次最终checkpoint-009244。登录节点CPU预检最终9244步、
本v2架构/训练合同、完整一轮BF16 precision policy、完整manifest/marker/文件；计算节点再查并绑定模型/tokenizer/config SHA及provenance。
strict/offline FP32加载LM和已训练routers，仅作新音频初始化，不恢复文本optimizer/scheduler/RNG/游标。
CED-small冻结/eval，官方freq_merge_126 mapper随机初始化训练；全LM/router训练。
原hybrid1477492、fixed20、audio126+SEP1+prompt256+answer300、answer-only CE、PAD/EOS、attention_mask=None及use_cache=False保持。
音频仍FP32前向、无autocast、原GradScaler/AdamW/LambdaLR/GradNorm/TF32关闭；文本BF16不改变音频配方。
8×micro8×GA4=GBS256，3epochs/17313步/866warmup，LM/router maxLR3e-4及mapper6e-4；
loss保持rank-local microbatch有效token均值→GA均值→DDP均值。七槽memory覆盖完整683个多模态位置。
沿母本initial_steps=0共享路径立即训练、后台单producer默认4workers复制raw bytes，全部文件/字节检查完成后，
八rank在optimizer边界、读取四个microbatch前all_reduce(MIN)一致切换本地；producer失败/死亡/心跳异常中止，先reap再清理本token tmpfs。
每次resume重新staging并从shared开始；音频完整恢复只接受本v2同代码/资产/数据/配方，使用原output root和原文本绝对路径。

本地提交推送GitHub后，远程登录节点执行转换：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/scripts/convert_stepwise_5_10x2_5_mesh_7slot_v2.sh
```

转换成功后提交文本正式训练，并保存TEXT_RUN：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
TEXT_RUN="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh_7slot_v2/formal_1epoch_bf16_$(date +%Y%m%d_%H%M%S%N)"
RSMOL_5_10X2_5_MESH_7SLOT_V2_OUTPUT_DIR="$TEXT_RUN" \
bash run_stage4_5_10x2_5_mesh_7slot_v2_formal_3090.sh
printf 'Audio text initialization: %s/checkpoint-009244\n' "$TEXT_RUN"
```

确认文本实际完成后，使用同一TEXT_RUN提交音频（新shell需手动恢复实际路径）：

```bash
bash run_mizar_s1_gbs256_recursive_mesh_7slot_v2_ced_small_3090.sh \
  --text-model-dir "$TEXT_RUN/checkpoint-009244" \
  --stage-workers 4
```

两阶段均pdgpu-3090/64CPU/256G/8GPU/1node、原容器；文本rsmol，音频mellow_c8204d8。
音频默认outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_7slot_v2_3090/formal_<timestamp>/，
最终formal/checkpoints/model--step-17313.ckpt和model--epo-3.ckpt；恢复命令见README。

python -B tests/validate_mesh_7slot_v2.py：116项实际本地检查通过，52项真实tensor检查缺依赖跳过（新路线26、四槽回归26）。
覆盖实际文本checkpoint writer→音频CPU preflight/runtime、错槽/历史七槽/短训练/精度/不完整产物拒绝、
完整model/converter/text trainer与音频更新循环受限AST对照、206原文件SHA保护、22复制文件一致、
真实Bash假提交/资源/参数转发/共享启动/失败清理/集体切换、四槽与历史七槽回归；Python/Bash/LF及文档命令语法检查通过。
报告audit_reports/mesh_7slot_v2_validation.json；--require-tensors禁止所有tensor skip（含CUDA）。
首次Windows沙箱临时目录权限失败后，沙箱外同流程复验通过。未执行commit/push、远程生产资产读取/转换、GPU训练或八卡恢复，不登记远程PASS。

## 2026-11-07：四槽 MeSH + CED-small，GBS256独立音频正式训练

按已确认计划从六槽音频母本完整复制隔离，见[四槽音频正式训练与恢复说明](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_4slot_ced_small/README.md)。
默认文本初始化固定为用户指定的
`stage4_5_10x2_5_mesh_4slot/formal_1epoch_bf16_20261106_103531451890103/checkpoint-009244`，
显式使用既有 `recursive_model_5_10x2_5_mesh_4slot.py`，offline/FP32严格加载完整LM和已训练routers。
无需重新转换，禁止截取六槽权重或重新初始化已训练routers；检查真实20层和六个独立router及weight[4,576]/bias[4]。
文本阶段BF16 autocast的参数和保存权重仍为FP32；本音频训练按母本使用FP32前向、无autocast、
原GradScaler/AdamW/LambdaLR/GradNorm、TF32关闭。仅继承文本模型/tokenizer，不继承文本优化器/调度/RNG/游标。

hidden576、20物理层/30次执行、`5+10×2+5` 保持，四槽瞬时memory覆盖完整多模态序列，
训练为[B,4,683,576]，首槽原始input embeddings、其余三槽零；三组write/read、六routers、
12份参数共13,848；prefix output-query/value、loop input-query/output-value与读写公式保持。
CED-small冻结/eval，官方freq_merge_126 mapper随机初始化并训练，完整LM/router继续训练；构建顺序沿母本。
原hybrid1,477,492行、fixed20、audio126/SEP1/prompt256/answer300、原answer-only CE/PAD/EOS/attention保持。
loss仍为rank-local microbatch有效token均值→四次GA均值→DDP均值，不引入文本预训练的全局token归一化。
8×8×4=GBS256、3epochs/17313updates/866warmup、LM/router maxLR3e-4及mapper6e-4、原保存点与数据顺序保持。

当前共享路径启动流程原样保留：登录节点CPU索引准备，计算节点initial_steps=0，无音频prefill等待；
共享原始音频立即用于训练，单producer/默认4workers后台复制raw bytes；完整文件/字节校验后，
八rank在optimizer边界、读取四个microbatch前all_reduce(MIN)，一致切换至节点本地。
producer失败/死亡/心跳异常中止；先停止并reap producer再清理本token拥有tmpfs；每次resume重新staging且从shared开始。
新训练目录、route/output/log/job/index-cache/tmpfs与恢复身份独立；原五/六槽、四槽文本和评测代码保持。

本地代码提交并推送GitHub后，远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_4slot_ced_small_3090.sh --stage-workers 4
```

沿用mellow_c8204d8、原容器及pdgpu-3090/64CPU/256G/8GPU/1node，直接提交一个formal，无自动smoke/tensor前置任务。
默认输出`outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_4slot_3090/formal_<timestamp>/`，
最终为其下`formal/checkpoints/model--step-17313.ckpt`与`model--epo-3.ckpt`。
完整resume只接受本四槽音频路线同代码/资产/数据/配方的完整正式checkpoint，按README传原run root和--resume-checkpoint；
指定文本checkpoint及旧五/六槽音频checkpoint均不能当音频full resume。

`python -B tests/validate_mizar_s1_mesh_4slot_ced_small.py`：64项实际本地检查通过，
47项真实tensor检查缺依赖跳过（四槽音频5、继承runtime16、六槽音频21、四槽文本5）。
覆盖真实Bash假提交、严格加载/错槽与dtype拒绝、共享启动/失败清理/集体切换、全部17314个LR位置与保存恢复合同；
94个当前Git原文件SHA保护、22个复制文件一致性及完整trainer/loader/contract受限AST对照通过。
快照以本轮开始时当前源码为准，包括先前已更新的两个评测入口；母本历史快照不修改。
六槽route/shared-start/tensor参与回归，旧behavior的历史评测SHA条款由新94文件快照覆盖。
报告`audit_reports/mizar_s1_mesh_4slot_ced_small_validation.json`；--require-tensors可禁止skip。
Windows沙箱临时目录访问失败后，沙箱外同一流程复验通过。未执行git commit/push、远程生产资产读取、
真实GPU训练或八卡恢复，不登记远程PASS。现有MMAU/MMAR evaluator仍需单独适配四槽音频合同。

## 2026-11-06：非 MeSH rank16、textmesh rank16与四套模态rank8的MMAU/MMAR评测

按用户授权扩展原评测入口，见[评测实现与六条提交命令](code/RSmol/scripts/README_mizar_s1_ced_multilayer_eval.md)。
继续本地改核心代码、GitHub同步、远程Linux保存权重/数据与执行评测。
新增三个variant及默认formal run：

| variant | 默认formal run |
|---|---|
| recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r16 | 20261105_173212933623545 |
| recursive_textmesh_layer8_4_inject_audio_reinject_loop_lora_r16 | 20261105_154241055340068 |
| recursive_textmesh_layer8_4_inject_audio_reinject_loop_modality_lora_r8 | 20261105_173710754949913 |

均精确选择各run的formal/checkpoints/model--step-17313.ckpt和formal/runtime.json，中间checkpoint不参与选择。
第二个相对outputs路径按训练启动器补全到/hpc_stor03/sjtu_home/jinwei.zhang/下。
真实训练模型直接导入，完整state_dict strict加载，route/runtime/资产/训练及模型源码SHA/step/geometry严格校验保持。
三条均保留30次执行、共享input投影一次、边界5的input→CED8及边界15的input→CED4，CED gate读取post-input hidden。
两套rank16/alpha32为280份A/B；四套模态rank8/alpha16为560份A/B；LoRA均3,256,320、串行分支共6,034,179参数。
textmesh仅位置126起的SEP/文本/padding/生成答案进入五槽memory，音频绕过memory/router；
每完整forward重建[B,5,L-126,576]，六routers保持prelude-output及loop-input query、loop-output write。
四套模态LoRA按固定音频126/文本L-126选择；每forward验证140投影/280低秩分支、四套各70次及索引复位。

原fixed20、prompt256/prefix383、首答案logit382、FP32/单GPU/batch1/seed1234/TF32关闭、
无cache/attention_mask=None/argmax/300tokens/首EOS停止、原始输出和官方scorer/各1000条保持。
沿用mellow_c8204d8和pdgpu-5090/8CPU/32G/1GPU/1node，无自动前置测试作业。
输出为各run的eval/mmau_test_mini_step17313/与eval/mmar_step17313/。

python -B tests/validate_mizar_s1_ced_multilayer_eval.py：282项实际检查通过，32项真实tensor检查缺Torch/Transformers跳过，
含当前适配23项、旧回归9项；覆盖17路线/34种千条模拟流程与假提交、训练源码NumPy行为、原协议AST及旧分支回归。
两条textmesh历史训练快照含本次获授权修改的四个评测文件，统一验证明确替代该历史快照条款，
其余185/117文件字节保护和完整训练AST/配方对照通过；不改训练快照或生产源码SHA检查。
报告audit_reports/mizar_s1_ced_multilayer_eval_validation.json；--require-tensors可禁止skip。
本地Windows沙箱临时目录受限后，沙箱外同流程验证通过。
未执行git commit/push、远程checkpoint/runtime读取、真实GPU评测或作业提交，不登记远程PASS。

## 2026-11-06：官方 Mizar final 5+20×2+5，S1 GBS128独立训练实现

按用户授权实施，见[正式训练与恢复说明](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_official_final_s1_gbs128_5_20x2_5/README.md)。
继续本地改代码、GitHub中转、远程Linux保存权重/数据与运行训练。原转换包与参考训练路线保持原样。

从完整 `models/Mizar159M-5-20x2-5-serial84-looplora-r8` 初始化，严格加载全部1038份state，
完整保留官方final的LM/mapper/CED和转换时保存的新参数，不重新初始化覆盖LoRA/MLP/gates。
GPU-safe加载入口与CPU转换入口分开，不调用会清空CUDA_VISIBLE_DEVICES的转换dependencies/reload_verified。
30物理层、50次执行、逻辑5/25串行input→CED8/4、两套独立rank8/alpha16 LoRA、CED冻结/eval保持。

循环基础权重（包括norm）及全部新增参数maxLR6e-4；原mapper、prelude/coda、embedding/最终norm/lm_head为3e-4。
原S1 hybrid1,477,492条、3epochs，8×microbatch4×GA4=GBS128，11542updates/epoch、34626总步、1731warmup。
FP32、无autocast/TF32、参考GradScaler/AdamW/LambdaLR/GradNorm保留；683token及原answer-only CE保持：
rank-local microbatch有效token均值→四次GA均值→DDP均值，不改成全局有效token加权。

按用户再次强调的要求：计算节点只等待本次路径metadata/owner就绪，initial_steps=0，
先直接读共享原始音频训练，同时单producer/默认4workers后台复制raw bytes到独立tmpfs。
producer全部完成且文件/字节计数通过后，八rank在optimizer边界、取microbatches之前all_reduce(MIN)，共同切到节点本地。
无音频prefill/watermark等待；producer失败/死亡/心跳异常中止；先停止/reap producer再清理本token拥有的tmpfs。

六个半epoch保存点5771/11542/17313/23084/28855/34626及epoch3 final均存完整状态；
同代码/依赖/资产/数据/配方/本路线和原output root才可full resume，保存八rank RNG与明确游标。
恢复时重建当前节点staging，重新从shared开始，不接受旧分配report；校验参数名/顺序、moments、LR/scheduler/GradNorm。
launcher直接提交formal，无自动smoke/tensor前置作业，沿参考pdgpu-5090/32CPU/256G/8GPU/1node与mellow_c8204d8。
训练目录、输出、日志、job、index-cache、tmpfs与resume身份独立。

本地代码提交并推送GitHub后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_official_final_s1_gbs128_5_20x2_5_5090.sh --stage-workers 4
```

默认输出为`outputs/RSmol/mizar_official_final_s1_gbs128_5_20x2_5_5090/formal_<timestamp>/formal/`。
恢复入口同脚本，传`--resume-checkpoint /原root/formal/checkpoints/model--step-5771.ckpt`，自动推导原output root。
`python -B tests/validate_mizar_official_training.py`检查实际文件/后台线程协议、真实Bash夹具、原50文件SHA保护、
数据/loss/clipping AST、所有调度位置与恢复合同；`--require-tensors`要求真实小尺寸CPU tensor检查且禁止skip。
本机缺Torch/Transformers/audio依赖，真实tensor检查明确跳过；不登记远程完整模型/八卡GPU/收敛/恢复PASS。
报告`audit_reports/mizar_official_training_validation.json`。Windows沙箱临时目录/Bash受限后，在沙箱外复验。
既有转换产物training_intent.implemented=False保留其历史含义，实际trainer合同另存；旧MMAU/MMAR evaluator仍需单独适配。
本轮未执行commit/push、远程数据/权重访问、真实作业提交或训练。

## 2026-11-06：官方 Mizar 5+20×2+5，第6步CloudStor发布兼容与续发布

用户新日志已到第6步，staging为 `Mizar159M-5-20x2-5-serial84-looplora-r8.staging-q492wvyj`，
随后目录发布报 `[Errno 22] Invalid argument`。按控制流，模型完整转换、首次前向及保存重载全部已完成；
正式目录发布尚未通过，不登记整条流程最终PASS。错误符合CloudStor不支持renameat2的RENAME_NOREPLACE标志。

`atomic_publish()`对EINVAL/ENOSYS/EOPNOTSUPP增加兼容：独占mkdir全新空目标，再用普通POSIX rename
原子替换本进程占位。已有目标拒绝；权限等错误不回退；失败只rmdir仍归自己的空占位，保留非空/被替换目录。
新增独立 `publish --staging ...`，只验证已有文件/完成验收记录并发布，不导入Torch、不重跑模型、不复制模型权重。
恢复仅允许conversion.py/audit.py/__main__.py的发布相关身份迁移，模型/合同/所有其他身份需一致；
原manifest/report/failure先备份到旁边publication-backup目录，再记录迁移；失败恢复原元数据。
要求strict、1038 tensor及683-token logits完整重载记录，因此旧第5步失败stage不可复用。

经GitHub同步修复后，远程仓库根目录执行：

```bash
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5.sh publish --staging /mnt/cloudstorfs/sjtu_home/jinwei.zhang/models/Mizar159M-5-20x2-5-serial84-looplora-r8.staging-q492wvyj
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5.sh verify
```

本地39项通过、11项真实依赖检查跳过，15个Python/Bash语法与原母本保护检查通过。
新增覆盖Linux不支持标志模拟、空占位/外部内容保护、失败恢复、完成记录/字节/模型身份拒绝、旧stage兼容与无Torch CLI。
Windows对POSIX替换空目录调用有适配模拟，尚未实测CloudStor发布，不冒充远程PASS。
未执行commit/push、远程命令或训练。诊断应cat目录内的failure.json，不能cat staging目录本身。

## 2026-11-06：官方 Mizar 5+20×2+5，第5步配置重载失败修复

用户反馈远程转换进入第5步后报 `recursive config must preserve every original source config field`，
失败staging为 `Mizar159M-5-20x2-5-serial84-looplora-r8.staging-aaic8a79`，最终目录尚未发布。
按现有控制流，反馈日志表示来源校验、官方严格加载及首次单遍对齐/完整50次执行前向已完成；
完整保存重载和发布尚未通过，不能登记整条转换PASS。

原因：`config_from_dict()` 将待保存字典直接传入 Transformers，库的from_dict会原地加入
`attn_implementation` 等运行时字段，导致随后保存的配置与严格预期不一致。
修复为传入deepcopy；保留完整字段校验，并在保存前提前校验，报错列出extra/missing/changed。
新增3项本地回归（含可变配置构造→JSON保存→真实文件审计）及2项可选真实Transformers配置测试。
本地27项通过、11项真实依赖检查因缺Torch/Transformers跳过，Python/Bash及原母本保护检查通过。
报告仍为 `audit_reports/mizar_official_recursive_conversion_validation.json`。

修复经GitHub同步后重跑原CPU `convert --seed 0 --threads 8`；旧staging可保留，不需删原始Mizar。
不手动改旧manifest/配置来绕过实现身份与完整字段检查。远程本次重跑、重载和发布仍待用户执行确认。
未执行commit/push、远程命令或训练。

## 2026-11-06：官方 Mizar final → 5+20×2+5，首阶段审计与转换代码

见[官方完整Mizar递归转换说明](code/RSmol/mizar_official_recursive_5_20x2_5/README.md)。
本次仅交付独立模型定义、来源/产物审计、CPU转换和严格重载接口；trainer与现有评测器接入留待下一阶段。
继续遵循本地仅代码、远程Linux存权重/数据、GitHub同步流程。

来源固定为远程 `models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt`，
完整继承官方final的LM、mapper、CED，不用assets中的基础LM权重替代final。
保留30个物理层与identity映射，前5层、中20层两轮、后5层，执行顺序 `0..24,5..24,25..29` 共50次。
中段5..24的七种投影加两套独立rank8/alpha16/scale2/dropout0 LoRA，A Kaiming、B零，基础权重也训练。
逻辑5/25仅音频位置0..125依次注入共享input残差，再分别注入CED8/CED4残差；
CED gate读取post-input hidden。输入取原final mapper输出，共享projection每LM forward一次；
CED每批一次，4/8块独立MLP，CED冻结/eval，无memory/router。
LoRA新增3,256,320、音频新增2,777,859，共6,034,179；完整模型165,334,019独立参数、1038份state tensor。

已记录用户确认的未来训练配置：S1 hybrid 1,477,492条、3epochs、8×4×4=GBS128，
11,542updates/epoch、34,626总步、1,731warmup；FP32及683-token布局沿参考。
循环基础权重（含norm）与全部新增参数maxLR6e-4；原mapper、prelude/coda、embedding/最终norm/lm_head为3e-4。
本轮仅提供参数角色审计，不构造训练optimizer或启动训练。

转换强制核对18个官方文件SHA/大小、454份plain state、严格加载后才注册新增模块，逐tensor核对继承值；
分别检查同30层单遍数值对齐、真实50次执行轨迹、完整20秒音频/683-token前向，保存后重构并精确核对全部tensor和logits。
零gate/B不意味着递归50次等于官方30次。产物打包原CED构造资产/tokenizer、完整model-only权重与报告/manifest；
不含optimizer、scheduler、训练RNG或游标。默认拒绝覆盖、源目标嵌套及Git checkout内权重输出；
独占锁、私有staging、Linux no-replace原子发布，失败保留诊断且不发布最终目录。

代码经GitHub同步后，远程checkout根目录执行：

```bash
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5.sh audit
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5.sh convert --seed 0 --threads 8
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5.sh verify --tensors --threads 8
```

wrapper沿用mellow_c8204d8 / Transformers4.54.1，强制离线、CPU、无调度器；默认输出远程
`models/Mizar159M-5-20x2-5-serial84-looplora-r8`，可用`--source/--output`指定独立目录。
显式`reload_verified()`重载，不能用标准AutoModel直接读取递归产物；旧MMAU/MMAR evaluator尚未适配。

`python -B tests/validate_mizar_official_recursive_conversion.py`：24项实际本地文件/合同检查通过，
9项真实CPU tensor检查因本机缺Torch/Transformers跳过；15个Python语法及Bash语法/LF通过。
覆盖官方mapper AST、30层映射/LR边界、真实文件SHA/篡改/缺失、路径安全、拒绝覆盖/原子发布、完整配置与产物身份。
Windows沙箱临时目录和Bash signal pipe受限后，在沙箱外同流程验证通过。
报告`audit_reports/mizar_official_recursive_conversion_validation.json`；`--require-tensors`禁止任何tensor skip。
未执行git commit/push、远程真实权重读取/生产转换、GPU训练，不登记远程PASS。

## 2026-11-06：两循环、四记忆槽 MeSH 文本转换与 BF16 完整一轮训练

从六槽文本路线完整复制隔离，见[四槽文本转换、正式训练及恢复说明](code/RSmol/text_5_10x2_5_mesh_4slot/README.md)。
新增独立 `recursive_model_5_10x2_5_mesh_4slot.py`、转换/训练器及四个提交入口；
类名、架构/训练/精度合同、`RSMOL_5_10X2_5_MESH_4SLOT_*`、模型、输出与日志隔离。
hidden576、20物理层/30次执行、`5+10×2+5` 保持；memory 为 `[B,4,L,576]`，
首槽取原始 input embeddings，其余三槽为零，每 forward 重建。三组 write/read、六个独立
`Linear(576,4)`、12份权重/偏置共13,848参数；初始化仍为 `sqrt(2/(slots×hidden))`。
保留 prefix output-query/value、循环 input-query/output-value 及原记忆读写公式。

采用历史五槽 BF16 autocast 方式：FP32 参数/AdamW 状态，模型前向 BF16 autocast，
FP32 CE/router softmax、无 GradScaler、TF32关闭；CE/归一化与 backward 在 autocast 外。
启动检查 CUDA BF16 支持；正式/短验证训练不允许 CPU 或静默精度回退，Gate A 可显式CPU诊断。
FP32 embedding/residual/memory 语义保留；checkpoint 的 `config.torch_dtype=float32` 是权重精度，
完整 precision policy 另存于 metadata/training state/report。旧五槽和六槽当前仍为FP32，未修改。

默认从远程干净的 `models/SmolLM2-5-10-5` 转换，亦支持原30层 `models/SmolLM2`；
保持原30→20层映射，20层源不二次抽层。只新初始化四槽 routers，逐tensor核对继承权重和tied embedding，
保留 tokenizer；默认拒绝训练态/已有MeSH源、Git checkout产物及已有模型目录覆盖。
转换wrapper激活 `rsmol`、离线且清空 `CUDA_VISIBLE_DEVICES`，仅CPU运行。

正式训练保留原Parquet/分片/行序、全局有效token CE归一化、AdamW和梯度裁剪、
8×8×16=GBS1024、context1024、9244updates/1epoch、463warmup、LR8e-4→8e-5、save500/retain3。
按已确认计划使用 `pdgpu-3090 / 64 CPU / 256G / 8 GPU / 1 node`，原容器与rsmol。
直接提交formal，无自动smoke/tensor前置任务。只接受同四槽BF16合同的完整formal checkpoint恢复；
校验precision policy、每rank游标/RNG、manifest、batch与调度。fresh只接受干净四槽转换来源。

本地代码提交并推送GitHub后，远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/scripts/convert_stepwise_5_10x2_5_mesh_4slot.sh
cd code/RSmol
bash run_stage4_5_10x2_5_mesh_4slot_formal_3090.sh
```

转换产物为 `models/SmolLM2-5-10x2-5-mesh-4slot`；正式输出为
`outputs/RSmol/stage4_5_10x2_5_mesh_4slot/formal_1epoch_bf16_<timestamp>/checkpoint-009244`。
这些路径均以 `/hpc_stor03/sjtu_home/jinwei.zhang/` 为根。

`python -B tests/validate_text_mesh_4slot_bf16.py`：50项实际本地检查通过（四槽27、原六槽23），
8项真实tensor检查因本地缺Torch/Transformers/safetensors跳过（四槽5、六槽3）。
覆盖真实Bash模拟转换/提交、CPU-only转换、64CPU、精度/旧路线隔离及正式恢复拒绝、
Parquet游标、全9244步LR、原骨干/数据核心AST对照；8个Python与6个四槽Bash语法/LF检查通过，
14个母本原文件SHA保持。报告为 `audit_reports/text_mesh_4slot_bf16_validation.json`。
Windows沙箱临时目录访问失败后，在沙箱外同一流程复验通过；可加 `--require-tensors` 禁止任何skip。
未执行git commit/push、远程生产权重读取/转换、真实BF16 GPU训练或八卡恢复，不登记远程PASS。
四槽音频训练及现有MMAU/MMAR evaluator适配不包含在本轮。

## 2026-11-05：仅文本 MeSH + 音频串行8→4，四套模态/循环 LoRA rank8

从 `run_mizar_s1_gbs256_recursive_textmesh_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16_5090.sh`
对应实现完整复制隔离，见[正式训练实现、提交及恢复说明](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_textmesh_ced_small_layer8_4_inject_audio_reinject_loop_modality_lora_r8/README.md)。
只将每轮完整序列的一套rank16/alpha32改为独立audio/text两套rank8/alpha16；两轮共四套。
物理层5..14的q/k/v/o、gate/up/down按位置0..125选audio，126起（含SEP）选text，按轮次索引0/1；
无需可学习的LoRA模态router，基础W共享并继续全量训练，self-attention仍覆盖完整多模态序列。
四套各814,080参数，共3,256,320参数/560份独立A/B张量，与母本两套rank16总参数量相同。

原文本五槽MeSH、六routers、三次write/read、prefix output-query/value及loop input-query/output-value全部保留；
音频不进memory/router，prelude输出后input→CED8，loop1输出后input→CED4，loop2输出直接进suffix。
共享input projection每forward一次，CED gate仍读取post-input hidden；20物理层/30次执行、683-token布局、
FP32、8×8×4=GBS256、3epochs/17313updates/866warmup、原数据/AdamW/LambdaLR/GradNorm/staging/保存恢复保持。
四套LoRA均使用maxLR6e-4、scale2/dropout0、A Kaiming/B零，纳入独立优化器角色、梯度及完整保存恢复。
保留原初始化资产：MeSH文本checkpoint-009244与CED-small；不从母本rank16音频checkpoint恢复。

本地代码提交并推送至GitHub后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_textmesh_ced_small_layer8_4_inject_audio_reinject_loop_modality_lora_r8_5090.sh --stage-workers 4
```

沿用mellow_c8204d8、pdgpu-5090 / 32CPU / 256G / 8GPU / 1node，直接提交一个formal，无自动smoke/tensor前置任务。
新训练目录、launcher、route、output、job、日志、index-cache、tmpfs和resume身份独立；原版训练与评测保持。
母本textmesh rank16等旧checkpoint不能full resume到本路线；现有MMAU/MMAR evaluator需要单独适配四套LoRA。

`python -B tests/validate_mizar_s1_textmesh_modality_lora_r8.py`：92项实际本地检查通过，
24项真实CPU tensor检查因缺依赖跳过（新版9项）。覆盖真实Bash模拟提交、位置125/126边界、四分支有限差分、
梯度/调用审计及原母本回归；Python/Bash语法、121原文件SHA保护、19复制文件除名称外一致、完整骨干/train AST对照通过。
报告为`audit_reports/mizar_s1_textmesh_modality_lora_r8_validation.json`；手动加`--require-tensors`可禁止skip。
本轮未执行git commit/push、远程权重读取、真实GPU训练或八卡恢复，不登记远程PASS。

## 2026-11-05：六槽 MeSH + CED-small / 官方 mapper，GBS256 独立正式训练

从原五槽 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small` 完整复制隔离，
见[六槽音频训练实现、GitHub同步与提交/恢复说明](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_6slot_ced_small/README.md)。
文本初始化固定为用户指定的
`stage4_5_10x2_5_mesh_6slot/formal_1epoch_fp32_20261031_124955519529617/checkpoint-009244`，
显式使用既有 `recursive_model_5_10x2_5_mesh_6slot.py`，FP32严格加载完整LM与已训练routers。
hidden576、20物理层/30次执行、两次中段循环、六个瞬时memory槽、三组write/read及六个独立routers；
memory覆盖完整多模态序列，训练为`[B,6,683,576]`，首槽为原始input embeddings，其余五槽为零。
保持prelude output-query/value、循环input-query/output-value及三次write/read。
只加载文本模型/tokenizer作为新音频训练初始化，不恢复文本optimizer、scheduler、RNG或数据游标。

CED冻结/eval、官方freq_merge_126 mapper随机初始化并训练，完整LM/router可训练；
hybrid 1,477,492行、fixed20、683-token布局、原answer-only CE/PAD/EOS/attention、
FP32/GradScaler、8×8×4=GBS256、3epochs/17313updates/866warmup、
原AdamW/LambdaLR/GradNorm、数据顺序、保存与后台staging全部保留。
按用户要求资源改为`pdgpu-3090 / 64 CPU / 256G / 8 GPU / 1 node`，沿用mellow_c8204d8及原容器。
新目录、入口、route、output、job、日志、index-cache、tmpfs与resume身份隔离；
原五槽训练、六槽文本实现及现有评测器保持原样。

本地改动提交/推送至GitHub后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_6slot_ced_small_3090.sh --stage-workers 4
```

直接提交一个formal，无自动smoke/tensor前置任务。默认输出为
`outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_6slot_3090/formal_<timestamp>/`，
最终产物为其下`formal/checkpoints/model--step-17313.ckpt`及`model--epo-3.ckpt`。
完整resume只接受本六槽音频路线相同源码/资产/数据/配方的正式checkpoint和原run输出root；
上述文本checkpoint及旧五槽音频checkpoint均不能当完整resume。

`python -B tests/validate_mizar_s1_mesh_6slot_ced_small.py`：56项实际检查通过，
24项真实CPU tensor检查因缺依赖跳过（新音频路线5项、继承runtime16项、六槽文本3项）。
Python/Bash语法、真实Bash假提交/拒绝、43个原文件SHA保护、23个复制文件一致性、
完整train/contract受限AST对照、全17314个LR位置、严格加载/实际router形状拒绝、
staging与原五槽回归通过。首次沙箱运行遇Windows临时目录权限问题，沙箱外同命令复验通过。
报告为`audit_reports/mizar_s1_mesh_6slot_ced_small_validation.json`，手动加`--require-tensors`可禁止skip。
未执行git commit/push、远程checkpoint读取、真实GPU训练或八卡恢复，不登记远程PASS。
现有五槽MMAU/MMAR evaluator尚未适配六槽合同，需要另行扩展。

## 2026-11-05：仅文本 MeSH 五槽 + 音频串行8→4 + 双循环 LoRA rank16

从原 MeSH `layer4_8_inject_loop_lora_r8` 母本完整复制隔离，
见[实现、恢复说明与正式提交命令](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_textmesh_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16/README.md)。
同一20物理层/30次执行的模型中，仅位置126起的SEP及文本参与五槽memory和六个routers；
memory实际为`[B,5,L-126,576]`，slot0取原始文本侧embeddings，其余为零，每次forward重建。
保留三次write/read；prelude query/value均为prefix文本输出，后两组query为循环输入、write value为循环输出。
音频不分配memory、不调用routers、不直接参与记忆读写；prelude audio输出先input残差再CED8，
第一轮audio输出先input残差再CED4，第二轮audio输出直接进入末五层。
共享input projection/gate、每次LM forward一次投影，CED4/8 MLP/gate独立，CED gate读取post-input hidden。
两轮中间十层七种投影使用独立LoRA，rank16/alpha32/scale2/dropout0，共3,256,320参数；
基础W共享且全量训练，六个文本routers继续训练；串行音频2,777,859与LoRA共新增6,034,179参数。

数据、FP32、8×8×4=GBS256、3epochs/17313updates/866warmup及原优化器/调度/GradNorm、保存和staging保持。
严格加载原MeSH文本checkpoint-009244及routers后注册新增模块，隔离CPU RNG；默认从文本初始化开始音频训练。
12类优化器角色、三门控及双LoRA梯度解锁、实际memory/router形状、注入/轮次轨迹纳入审计。
新目录、启动器、route、output、job、日志、index-cache、tmpfs与恢复合同全部隔离，旧音频checkpoint不能full resume。

本地代码经GitHub提交/推送后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_textmesh_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16_5090.sh --stage-workers 4
```

沿用mellow_c8204d8、pdgpu-5090 / 32CPU / 256G / 8GPU / 1node，直接提交formal，无自动smoke/tensor前置任务。
`python -B tests/validate_mizar_s1_textmesh_ced_loop_lora_r16.py`：85项实际检查通过，22项真实CPU tensor检查因缺依赖跳过（新路线8项）。
Python/Bash语法、真实Bash假提交/拒绝、189原文件SHA保护、18复制源码受限差异、完整train AST及母本/rank16回归通过。
NumPy执行真实源码验证文本memory/query/write、音频串行门控及最终直通、LoRA轮次和梯度审计失败用例；不冒充PyTorch autograd。
报告为`audit_reports/mizar_s1_textmesh_ced_loop_lora_r16_validation.json`；可手动加`--require-tensors`禁止skip。
未执行git commit/push、远程权重读取、真实GPU训练或八卡恢复，不登记远程PASS。
现有MMAU/MMAR evaluator尚未适配该hybrid/rank16合同，需要单独适配。

## 2026-11-05：非 MeSH 串行8→4双循环 LoRA 扩展为 rank16 / alpha32

从原8→4的rank8/alpha16正式训练路线完整复制隔离，
见[实现、恢复说明与正式提交命令](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16/README.md)。
仅物理层5..14的q/k/v/o和gate/up/down增加两套独立LoRA，两轮基础W共享且全量训练。
rank16、alpha32、scale2、dropout0、LoRA max_lr6e-4；两套LoRA共3,256,320参数，
加原串行音频分支2,777,859后，相对普通CED循环母本新增合计6,034,179。
保留两轮input→CED8 / input→CED4、post-input gate、全部初始化与训练流程。
数据、FP32、8×8×4=GBS256、3epochs/17313updates/866warmup、原优化器/调度/GradNorm及保存/staging保持。
新目录、启动器、route、output、job、日志、index-cache、tmpfs与恢复合同均隔离。
原rank8两顺序、MeSH路线、共享模型和现有评测实现保持原样。

本地代码经GitHub提交/推送后，远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16_5090.sh --stage-workers 4
```

沿用mellow_c8204d8、pdgpu-5090 / 32CPU / 256G / 8GPU / 1node，直接提交formal。
从原文本checkpoint-step-009244初始化，rank8音频checkpoint不能直接full resume。
`python -B tests/validate_mizar_s1_ced_loop_lora_r16.py`：108项实际检查通过，
33项真实CPU tensor检查因缺依赖跳过（新版7项）。Python/Bash语法、179原文件SHA保护、
26复制源码文件受限差异校验及完整train/optim/骨干/loader AST对照通过。
报告为 `audit_reports/mizar_s1_ced_loop_lora_r16_validation.json`；手动加 `--require-tensors` 禁止skip。
本次未执行git commit/push、远程资产读取、真实GPU训练或分布式恢复，不登记远程PASS。
现有MMAU/MMAR评测器需要另行适配rank16/alpha32合同。

## 2026-11-05：非 MeSH 串行4→8与 MeSH 4→8 的双 LoRA MMAU/MMAR 评测

两个原 CED 评测入口新增以下路线，见[完整说明与四项提交命令](code/RSmol/scripts/README_mizar_s1_ced_multilayer_eval.md)：

| variant | 默认 formal run |
|---|---|
| recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8 | 20261104_131843931946322 |
| recursive_mesh_layer4_8_inject_loop_lora_r8 | 20261104_161727489909082 |

均精确读取各自 `formal/checkpoints/model--step-17313.ckpt` 和同 run 的 `formal/runtime.json`；
中间 checkpoint 不参与默认选择，训练 geometry 仍为8×8×4，评测为单GPU/batch1。
直接导入各自实际训练模型并 strict 加载全部已训练参数，训练目录/共享模型及源码 SHA 校验保持。
非 MeSH 保留每次 forward 一次 input projection、边界5的 input→CED4、边界15的 input→CED8，
CED gate 读取 post-input hidden。MeSH 保留五槽瞬时 memory、六个独立 routers 与三次 write/read，
首组 query/value 为 prefix output，后两组 query 为循环输入、write value 为循环输出，
仅前两次 read 后注入 A4/A8，无额外 input 残差分支。
两路线每次完整 forward 均检查30次层调用及中间十层七种投影的双 LoRA：两轮各70次调用、
280份独立 A/B、rank8/scale2/dropout0与索引复位；禁止cache、gradient checkpointing及adapter merge。
音频只直接注入前126位置，LoRA作用于完整序列。新增参数总计分别为4,406,019、4,073,090。

fixed20、256-token prompt/383 prefix、单GPU/batch1/FP32/seed1234/TF32关闭、无cache、
argmax/300tokens/首EOS停止、完整原始输出、官方 scorer及两个benchmark各1000条的原协议保持。
沿用 mellow_c8204d8 与 pdgpu-5090 / 8CPU / 32G / 1GPU / 1node，无自动前置任务。
每个训练 run 各自产出 `eval/mmau_test_mini_step17313/` 与 `eval/mmar_step17313/`。

`python -B tests/validate_mizar_s1_ced_multilayer_eval.py` 本地223项检查通过，
28项真实 tensor 检查因缺 Torch/Transformers 跳过（当前适配19项、旧回归9项），Python/Bash语法通过。
覆盖十四路线/28种完整千条模拟流程与假提交、非MeSH两种串行顺序及MeSH双LoRA的实际训练源码NumPy行为、
strict加载接口、错轮/漏调用/重复调用/索引残留/错误router-gate顺序拒绝及旧路线回归。
真实 tensor 用例另覆盖两条新路线的adapter/direct logits/token、首答案对齐、完整权重恢复、
非零LoRA禁用/交换对照、post-input/post-read gate数值、MeSH memory/query/write来源及跨样本隔离。
报告为 `audit_reports/mizar_s1_ced_multilayer_eval_validation.json`；可手动加 `--require-tensors` 禁止skip。
本轮未执行git commit/push、远程checkpoint/runtime读取或真实GPU评测，不登记远程PASS。

## 2026-11-04：非 MeSH 串行8→4 + 双循环 LoRA 的 MMAU/MMAR 评测

两个原 CED 评测入口新增 `recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8`，
见[实现与两项提交命令](code/RSmol/scripts/README_mizar_s1_ced_multilayer_eval.md)。
默认绑定 `mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8_5090/formal_20261103_165410438279511`，
精确读取同 run 的 `formal/checkpoints/model--step-17313.ckpt` 和 `formal/runtime.json`。
直接导入非 MeSH 真实训练模型，strict 加载全部 mapper、A4/A8 MLP、input projection/gates、
共享 LM 基础权重与两轮 LoRA；训练源码/共享模型与原 SHA 身份校验保持。
每次 forward 保留 input→CED8 / input→CED4、一次 input 投影及30次物理层调用；
中间十层七种投影使用两轮各70次 LoRA，检查280份独立 A/B、rank8/scale2、调用复位及禁止 gradient checkpointing。
音频分支2,777,859与LoRA1,628,160合计4,406,019新增参数，基础W不重复计数。

fixed20、256-token prompt/383 prefix、单GPU/batch1/FP32、无cache、argmax/300tokens/首EOS停止、
原始输出与官方 scorer，以及两个 benchmark 各完整1000条的原协议保持。
沿用 mellow_c8204d8 与 pdgpu-5090 / 8CPU / 32G / 1GPU / 1node，无自动前置作业。
结果在该训练 run 的 `eval/mmau_test_mini_step17313/` 与 `eval/mmar_step17313/`。

`python -B tests/validate_mizar_s1_ced_multilayer_eval.py` 本地192项检查通过，
26项真实 tensor 检查因缺 Torch/Transformers 跳过（当前适配17项、旧回归9项），Python/Bash语法通过。
覆盖十二路线/24种千条模拟流程与假提交、LoRA真实源码NumPy数值行为、错轮/漏调用/重复调用/残留索引拒绝、原评测回归；
真实 tensor 用例覆盖已训练共享W/两套A/B strict restore、非零LoRA数值、首答案对齐和跨样本/异常隔离。
报告为 `audit_reports/mizar_s1_ced_multilayer_eval_validation.json`，可手动加 `--require-tensors` 禁止skip。
本轮未执行git commit/push、远程权重/runtime读取或真实GPU评测，不登记远程PASS。
非 MeSH 4→8双LoRA及MeSH 4→8双LoRA已于2026-11-05完成原MMAU/MMAR评测适配。

## 2026-11-03：MeSH CED4→8 注入 + 中间十层两套独立 LoRA

从 MeSH `layer4_8_inject` 母本完整复制隔离一个正式训练版本，
见[实现、恢复说明与提交命令](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8/README.md)。
仅物理层5..14的q/k/v/o与gate/up/down投影加两套独立LoRA，按两轮分别选择；
基础W共享且继续全量训练，前五层、后五层和六个routers不加LoRA。
rank8、alpha16、scale2、dropout0、LoRA max_lr6e-4，两套新增1,628,160参数。
原五槽/六router、三次memory write/read及read后CED4→8注入保持；
router query仍为各轮输入，写入值为各轮中段output。

严格加载原MeSH文本checkpoint及全部routers后再注册gate与LoRA，
新增初始化隔离CPU RNG，B零初始化；两套LoRA纳入优化器、梯度审计、GradNorm和完整保存/恢复。
新目录、启动器、route、output、job、日志、index cache和tmpfs隔离。
母本、共享模型及上一轮两个非MeSH LoRA版本保持不变。
原数据、FP32、8×8×4=GBS256、3epochs/17313updates/866warmup及训练配方保持。

本地代码经GitHub提交/推送后，远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8_5090.sh --stage-workers 4
```

沿用mellow_c8204d8及pdgpu-5090 / 32CPU / 256G / 8GPU / 1node，直接提交formal。
统一验证 `python -B tests/validate_mizar_s1_mesh_ced_loop_lora.py`：100项实际检查通过，
23项真实CPU tensor检查因缺依赖跳过（新版本7项）。Python/Bash语法、102原文件SHA保护、
完整MeSH骨干/train函数AST对照和母本回归通过。
报告为 `audit_reports/mizar_s1_mesh_ced_loop_lora_validation.json`；可手动加 `--require-tensors` 禁止skip。
本次未执行git commit/push、远程资产加载、真实GPU训练或分布式恢复，不登记远程PASS。
本路线双LoRA已于2026-11-05接入原MMAU/MMAR评测器，见上方最新说明。

## 2026-11-03：非 MeSH 串行音频注入 + 中间十层两套独立 LoRA

从原 `layer8_4_inject_audio_reinject` 完整复制两条正式训练路线，
见[双 LoRA 实现、隔离说明与提交命令](code/RSmol/scripts/README_mizar_s1_ced_loop_lora.md)。
仅物理层5..14的q/k/v/o与gate/up/down投影加两套独立LoRA；
两轮共享基础W且继续全量训练，前五层/后五层不加LoRA。
rank8、alpha16、scale2、dropout0、LoRA maxLR6e-4；两套新增1,628,160参数。
两版分别保持 input→CED8 / input→CED4，以及 input→CED4 / input→CED8，
中间gate读取post-input hidden，input投影/gate共享，A4/A8 MLP/gate独立。

严格加载原文本checkpoint后再注册LoRA；CPU RNG隔离，B零初始化，
新增四类LoRA A/B参数角色，纳入优化器、首步/更新后梯度审计、GradNorm、保存及完整恢复。
两个目录/启动器/route/output/log/job/index-cache/tmpfs/resume身份完全隔离。
母本及共享模型未修改；原数据、FP32、GBS256（8×8×4）、3epochs/17313updates/866warmup、
原loss、AdamW/LambdaLR/GradScaler/GradNorm、保存与后台staging配置保持。

本地代码经GitHub提交/推送同步后，远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
# 8→4 + 两套 LoRA
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8_5090.sh --stage-workers 4
# 4→8 + 两套 LoRA
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8_5090.sh --stage-workers 4
```

两命令各提交一个formal作业；沿用mellow_c8204d8及pdgpu-5090 / 32CPU / 256G / 8GPU / 1node，
无自动smoke/tensor前置任务。只能从对应新版本、同源码/资产/数据/配方的完整checkpoint恢复。
非 MeSH 8→4双LoRA已于2026-11-04适配原MMAU/MMAR评测；4→8双LoRA于2026-11-05完成适配。

统一验证 `python -B tests/validate_mizar_s1_ced_loop_lora.py`：149项实际检查通过，
51项真实CPU tensor检查因缺依赖跳过（新版本共14项）。Python/Bash语法、
两启动器假提交、非零LoRA数值/轮次对照、有限差分、梯度审计失败用例、
12类优化器分组、跨路线恢复拒绝及母本回归通过。
34个母本原文件SHA保护，完整骨干和train函数经过明确新增节点剥离后的AST对照通过。
报告为 `audit_reports/mizar_s1_ced_loop_lora_validation.json`；可手动加 `--require-tensors` 禁止skip。
本次未执行git commit/push、远程资产加载、八卡训练或真实分布式恢复，未登记远程PASS。

## 2026-11-03：补充 MeSH 8→4 注入的默认评测产物

两个原 CED MMAU/MMAR 入口的 `recursive_mesh_layer8_4_inject` 已绑定
`mizar_s1_ced_small_gbs256_recursive_mesh_layer8_4_inject_5090/formal_20261102_112040402999384`，
精确选择同 run 的 `formal/checkpoints/model--step-17313.ckpt` 和 `formal/runtime.json`。
复用已支持的真实训练模型与 adapter，保持 read 后先 A8 再 A4 注入、五槽/六router，
原推理和评分协议保持。完整命令见[评测说明](code/RSmol/scripts/README_mizar_s1_ced_multilayer_eval.md)。
本次复验15项评测合同/上下文检查及9项Bash假提交检查全部通过；未读取远程权重或提交真实作业。

## 2026-11-03：扩展 MMAU/MMAR，支持 prompt_mix、串行8→4与四循环 MeSH

原两个 CED 评测提交入口已扩展为十一条路线，见
[完整实现说明、默认产物和八项提交命令](code/RSmol/scripts/README_mizar_s1_ced_multilayer_eval.md)。
本次四份产物均精确绑定同 run 的 formal/checkpoints 和 formal/runtime.json：

| variant | 默认 formal run 时间戳 | step / 训练 geometry |
|---|---|---|
| recursive_mesh_layer4_8_audio7slot_split3_inject | 20261102_002211827036990 | 17313 / 8,8,4 |
| recursive_mesh_layer4_8_audio7slot_split3_prompt_mix | 20261102_132232677352730 | 17313 / 8,8,4 |
| recursive_5_10_5_layer8_4_inject_audio_reinject | 20261102_112121702891801 | 17313 / 8,8,4 |
| recursive_mesh_5_10x4_5 | 20261103_093224843011222 | 34626 / 8,4,4 |

直接导入各路线真实训练模型并 strict 加载完整音频训练 checkpoint，训练源码/共享模型及 SHA 校验保持。
prompt_mix 使用有效 prompt 输入 embedding 计算一次三路权重，E_mix 作 prefix，
每个完整 LM forward 显式传入相同三份加权 seeds 重建音频七槽/文本五槽；不重复加权。
串行8→4保留每次 forward 一次原 mapper prefix 投影，以及边界5/15的
input→CED8、input→CED4顺序。四循环保留20物理层/50次执行、全序列七槽和10个 routers，禁止两循环 ablation。

fixed20、256-token prompt、383 prefix、FP32/batch1/单GPU、无KV cache、argmax/300 tokens、
首次EOS停止、原始输出及官方 scorer 均保持；两个 benchmark 各完整1000条、零skip。
沿用 mellow_c8204d8 和 pdgpu-5090 / 8CPU / 32G / 1GPU / 1node，无自动前置测试作业。
中间 checkpoint 不参与默认选择；四循环输出目录带 step34626，其余带 step17313。

本地验证入口仍为 `python -B tests/validate_mizar_s1_ced_multilayer_eval.py`，
覆盖十一条路线、22种千条模拟流程/提交组合、实际训练源码 NumPy/stdlib 行为及旧评测回归，
177项实际检查通过，23项真实 tensor 检查因缺 Torch/Transformers 跳过，Python/Bash语法通过。
报告为 audit_reports/mizar_s1_ced_multilayer_eval_validation.json。
缺少 Torch/Transformers 的真实 tensor 检查明确记录 skip；可手动加 --require-tensors 禁止跳过。
本轮未读取远程权重或提交真实作业，服务器上的完整权重加载和八项 full 评测尚待执行。

## 2026-11-02：四循环 MeSH + CED-small / 官方 mapper，GBS128 正式训练

完整复制CED MeSH母本，新增
[四循环CED训练路线、GitHub同步及提交/恢复说明](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs128_3epochs_recursive_mesh_5_10x4_5_ced_small/README.md)。
文本采用现有x4严格loader及stage4_5_10x4_5_mesh的checkpoint-018488初始化：
hidden576、20物理层、50次逻辑执行、`0..4,(5..14)×4,15..19`、7个瞬时memory槽和10个独立router。
CED冻结/eval，官方freq_merge_126 mapper随机初始化并全量训练，完整LM与router可训练。

除文本架构/初始化源和GBS改为128（8 GPU×microbatch4×GA4）外，
沿用原CED的hybrid 1,477,492行、683-token布局、answer-only CE、FP32/GradScaler、
AdamW、GradNorm、LM/router maxLR3e-4、mapper maxLR6e-4、3epochs和后台staging。
对应11,542updates/epoch、34,626total、1,731warmup（沿用round规则）、每epoch丢弃116行。
route/output/log/job/index-cache/tmpfs和恢复身份独立，原启动器/原路线/共享模型未修改。

本地代码经GitHub提交并同步后，远程登录节点直接提交formal：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs128_recursive_mesh_5_10x4_5_ced_small_5090.sh --stage-workers 4
```

保持mellow_c8204d8及pdgpu-5090 / 32CPU / 256G / 8GPU / 1node，无自动smoke/tensor前置任务。
默认输出根为
`outputs/RSmol/mizar_s1_ced_small_gbs128_recursive_mesh_5_10x4_5_5090/formal_<timestamp>/`，
最终文件在其formal/checkpoints/model--step-34626.ckpt及model--epo-3.ckpt。
完整恢复须使用本新路线相同源码/资产/数据/合同checkpoint和原输出根目录；现有评测器需独立适配。

统一检查入口 `python -B tests/validate_mizar_s1_mesh_x4_ced_small.py`：
本地63项实际测试及Python/Bash语法通过；21项真实CPU tensor检查因缺依赖跳过。
覆盖strict loader、四循环trace、全部LR位置、GBS128 staging与恢复身份、Bash模拟提交和原路线回归。
53个原文件normalized-LF SHA保护、完整训练更新循环和CED配方对照通过；
报告为audit_reports/mizar_s1_mesh_x4_ced_small_validation.json，可手动加--require-tensors禁止skip。
本轮未执行git commit/push、远程真实权重加载、八卡训练或真实分布式恢复，未登记远程PASS。

## 2026-11-02：有效 prompt 条件的 E/A4/A8 输入融合，MeSH 七槽 split3 独立训练

从clean split3完整复制新增
[七槽 prompt_mix 正式路线与提交/恢复说明](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_audio7slot_split3_prompt_mix/README.md)。
原256-token prompt的有效input embeddings均值，经过新增Linear576→576/GELU/Linear576→3和softmax，
得到每样本E/A4/A8三个权重。池化只用tokenizer attention_mask去掉padding，
保留有效!及EOS，不使用answer、SEP或contextual hidden，梯度不detach。
prelude音频输入为E_mix=αE·E+α4·A4+α8·A8；
七槽memory为[αE·E,0,0,0,0,α4·A4,α8·A8]，三份加权tensor只计算一次。
六个音频router初始概率仍为[p0/3,p1,p2,p3,p4,p0/3,p0/3]，文本仍五槽。
后续仅原MeSH all-slot write/read，无残差重注入或×3补偿。
新prompt MLP334,083参数，相对普通CED MeSH新增合计2,802,093。

原文本MeSH checkpoint-009244初始化、hybrid数据、683-token布局、FP32、
GBS256（8×8×4）、3epochs/17313updates/866warmup、AdamW/LambdaLR/GradNorm/GradScaler、
数据顺序及后台staging保持；新MLP同音频分支maxLR6e-4，LM/文本router仍3e-4。
route/output/log/job/index-cache/tmpfs/完整恢复身份隔离，旧路线与共享模型未修改。

本地代码经GitHub提交并同步后，远程登录节点直接提交formal：

~~~bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_split3_prompt_mix_5090.sh --stage-workers 4
~~~

沿用mellow_c8204d8及pdgpu-5090 / 32CPU / 256G / 8GPU / 1node，无自动smoke/tensor前置任务。
默认输出为
outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_prompt_mix_5090/formal_<timestamp>/，
最终文件在其formal/checkpoints/model--step-17313.ckpt与model--epo-3.ckpt。
仅支持本新路线同源码/资产/数据/配方完整checkpoint恢复，现有评测器需独立适配。

检查入口 python -B tests/validate_mizar_s1_mesh_ced_audio7slot_split3_prompt_mix.py：
90项实际测试通过，30项真实CPU tensor检查因缺依赖跳过（本路线17项）。
实际NumPy代码执行、finite difference、训练推理前缀/种子一致、原路线回归、Bash模拟提交及staging检查通过。
1978个原文件normalized-LF SHA保持，整个trainer/MeSH主体/优化器/恢复及原训练合同对照通过。
报告为audit_reports/mizar_s1_mesh_ced_audio7slot_split3_prompt_mix_validation.json；
可手动加--require-tensors禁止tensor skip。
尚未执行远程真实权重加载、八卡训练或完整恢复，未登记远程PASS。

## 2026-11-02：官方 Mizar final 权重沿用现有 CED MMAU/MMAR 协议评测

新增[独立官方 Mizar 评测目录与完整说明](code/RSmol/mizar_official_eval/README.md)。
默认读取已上传的 `/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M`，
使用官方 `checkpoints/final/stage3_q4_seed20260905.ckpt`：Stage3 / Q4 / seed20260905 / update200，
159,299,840 独立参数。checkpoint、15 个 base assets、model_info 和 provenance
共18个文件固定大小/SHA256；不需要训练数据 manifest 或项目训练 runtime.json。

通过现有非递归 CED 模型构造标准30个独立LM层，FP32下严格全量加载官方 plain state_dict；
不替换训练模型、不重初始化已训练 mapper。直接复用原 evaluator 的音频、prompt、
generate/greedy、预测写入和官方评分函数对象，保留中心裁20s、256-token prompt及EOS后缀、
383 prefix、PAD17/EOS0、batch1/FP32/TF32关闭、无KV cache、纯argmax/300tokens、
原始EOS与空格，以及两个benchmark各1000条、零跳过、官方评分分母1000。
只读hooks核验每次forward依次执行0..29，不改变输出。

两个旧提交脚本各只增加 `official_mizar` 分支；默认base和全部旧训练版本评测保持。
本地改动经GitHub提交并同步后，远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
RSMOL_MIZAR_CED_EVAL_VARIANT=official_mizar bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=official_mizar bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

使用 `mellow_c8204d8`、`vc submit`、pdgpu-4090 / 8CPU / 32G / 1GPU / 1node，
保留原 `_5090.sh` 入口文件名，`official_mizar` 分支实际提交到 `pdgpu-4090`。
无自动smoke/tensor前置作业。独立输出根为
`outputs/RSmol/mizar_official_stage3_q4_seed20260905_matched_ced_v1/{mmau_test_mini,mmar}`，
同目录重提可恢复JSONL；改动权重/资产/模型源码/评测代码/协议等身份时拒绝混用。
新环境变量使用 `RSMOL_MIZAR_OFFICIAL_*`；路径覆盖及产物字段见目录README。
结果应标记“官方Stage3 final在本项目相同协议下的对照”，不等同于论文9k/五seed原生评测复现。

验证入口 `python -B tests/validate_mizar_official_eval.py`：
本地149项实际检查通过，22项真实CPU tensor检查因缺Torch/Transformers跳过（本扩展3项）。
覆盖strict loader、官方mapper源码AST一致性、两类1000条模拟流程、断点恢复、评分分母、
真实Bash假提交及原各CED评测回归；另只读核验本地已下载18个文件SHA256全部通过。
报告为 `audit_reports/mizar_official_eval_validation.json`；
可手动加 `--model-dir <已下载目录>` 检查资产，或 `--require-tensors` 禁止tensor skip。
本轮未提交远程作业，真实权重tensor加载及GPU完整评测尚未执行，未登记远程PASS。

## 2026-11-02：非 MeSH input embeddings → CED 8→4 串行注入正式训练

已完整复制非 MeSH 8→4 路线，新增
[串行组合路线及完整提交/恢复说明](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject/README.md)。
每次 LM forward 从原 mapper 音频前缀一次计算共享 A_input。
在零基逻辑索引5/15，先用共享 input gate 注入 A_input，再用更新后的 hidden
计算独立 gate8/gate4，分别注入 A8/A4。实际轨迹
`[[5,"input"],[5,"layer8"],[15,"input"],[15,"layer4"]]`。
CED仍每批一次编码提取最终层与完整第4/8层；原mapper和两套MLP保持。
三个gate全零，input投影沿用原随机初始化/零bias；
中间MLP/gate注册顺序仍4、8，新增input模块单独CPU RNG隔离。
相对普通CED循环母本新增合计2,777,859参数。

保留原文本 stage4_5_10_5 checkpoint-step-009244 初始化、20物理层/30次执行、
hybrid数据、683-token布局、原loss、FP32、GBS256（8×8×4）、
3epochs/17313updates/866warmup和原优化器/调度/GradNorm/staging。
LM maxLR3e-4；mapper、input投影、两套MLP与三个gate maxLR6e-4。
新route/output/log/job/index-cache/tmpfs和恢复身份隔离，只接受本组合路线的完整正式checkpoint。

本地代码经GitHub同步后，在远程登录节点直接提交formal：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_5090.sh --stage-workers 4
```

沿用mellow_c8204d8和pdgpu-5090 / 32CPU / 256G / 8GPU / 1node，无自动smoke或tensor前置任务。
默认输出根为
`outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_5090/formal_<timestamp>/`，
最终为其下formal/checkpoints/model--epo-3.ckpt及model--step-17313.ckpt。

统一检查入口`python -B tests/validate_mizar_s1_ced_layer84_audio_reinject.py`：
本地109项实际检查通过，37项真实CPU tensor检查因缺依赖跳过（新路线12项）。
Python/Bash语法、非零gate串行数值/有限差分、关闭单分支对照、梯度审计、
模拟提交及原8→4/audio_reinject/4→8/普通CED循环回归通过。
源码对照保护137个原文件，完整模型和train函数只保留明确新增注入及合同/诊断差异；
报告为audit_reports/mizar_s1_ced_layer84_audio_reinject_validation.json。
具备原依赖时可手动加--require-tensors禁止skip。
未执行远程真实权重加载、八卡训练和完整恢复，未登记远程PASS；现有评测器需独立适配。

## 2026-11-02：扩展原 MMAU/MMAR evaluator，支持七槽 split3、split3_inject 和 MeSH 8→4

两个原提交入口 `code/RSmol/run_mizar_s1_ced_small_mmau_eval_5090.sh`、
`code/RSmol/run_mizar_s1_ced_small_mmar_eval_5090.sh` 现支持八条多层交互路线。
新增 `recursive_mesh_layer4_8_audio7slot_split3`、
`recursive_mesh_layer4_8_audio7slot_split3_inject` 和 `recursive_mesh_layer8_4_inject`；
四层 `recursive_mesh_layer36912_listen` 继续使用原 v2 适配。
详见[评测说明及完整提交命令](code/RSmol/scripts/README_mizar_s1_ced_multilayer_eval.md)。

本次两份默认产物为四层 listen 的 `formal_20261101_113133327944502` 和
split3 的 `formal_20261101_170211458358358`，均精确读取同 run 的
`formal/checkpoints/model--step-17313.ckpt` 和 `formal/runtime.json`。
后两条训练完成后显式传入 `--run-root` 即可，无需再改代码；不自动扫描 latest。

直接导入各路线训练模型、严格加载完整 state_dict，训练源码及 SHA 校验保持。
七槽路线保留音频7槽/文本5槽及12个独立 routers；加载后用 `fresh=False`
校验已训练参数，不重新三等分 router 或清零 gate。
split3_inject 同时保留种子memory和read后4→8残差；8→4路线仍传 `(A4,A8)`，
由训练模型按 `[[5,8],[15,4]]` 注入。每步完整forward重建memory。
原 fixed20、256-token prompt、383 prefix、FP32/batch1、无KV cache、
argmax/300-token/EOS及官方scorer设置均保持，两个benchmark各完整1000条。

本地统一验证入口 `python -B tests/validate_mizar_s1_ced_multilayer_eval.py`：
125项实际检查通过，19项真实CPU tensor检查因缺 Torch/Transformers 跳过；
Python/Bash语法、八路线合同/上下文/16种1000条模拟流程、提交参数和旧评测回归通过。
报告为 `audit_reports/mizar_s1_ced_multilayer_eval_validation.json`。
本轮未读取远程权重或提交作业，真实checkpoint加载和full评测需在服务器执行。

## 2026-11-02：MeSH CED 第 8→4 层 read 后残差注入独立训练路线

已完整复制原 MeSH 第4→8层注入路线，新增
[MeSH 8→4 隔离路线](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer8_4_inject/README.md)。
唯一模型计算变化为：`read_routers[0]` 的 memory read 后、第一次中段循环前使用 A8/gate8；
`read_routers[1]` 的 memory read 后、第二次中段循环前使用 A4/gate4，实际轨迹 `[[5,8],[15,4]]`。
CED仍一次编码提取第4/8层，MLP/gate注册与初始化顺序仍为4、8，显式接口仍为`(A4,A8)`。
原final-layer mapper、残差公式、零gate、CPU RNG、20物理层/30次执行、五槽/六router、
循环input-query/output-value与第三次read后不注入均保持；新增参数仍为2,444,930。

沿用原MeSH文本`formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244`初始化，
保留hybrid 1,477,492行、683-token布局、loss/attention、FP32、GBS256（8×8×4）、
3epochs/17,313updates/866warmup、LM/routers LR3e-4与mapper/MLP/gate LR6e-4、
AdamW/LambdaLR/GradNorm/GradScaler、数据顺序、后台staging及完整恢复。
新route/output/log/index-cache/tmpfs和resume身份独立，原4→8、普通MeSH CED、非MeSH路线和共享模型未修改。

本地代码经GitHub同步后，在远程登录节点直接提交formal，无自动smoke或tensor前置任务：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer8_4_inject_5090.sh --stage-workers 4
```

沿用`mellow_c8204d8`与`pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`。
默认输出根为`outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer8_4_inject_5090/formal_<timestamp>/`，
内部最终为`formal/checkpoints/model--epo-3.ckpt`和`model--step-17313.ckpt`。
仅接受本8→4路线的完整正式checkpoint恢复，拒绝原4→8、普通MeSH或非MeSHcheckpoint full resume；
详细产物、恢复命令与架构见路线README。当前MMAU/MMAR evaluator尚未适配本新路线。

统一检查入口`python -B tests/validate_mizar_s1_mesh_ced_layer84.py`：98项实际检查通过，
26项真实CPU tensor检查因缺依赖跳过（新路线10项、原MeSH 4→8 9项、非MeSH 4→8 7项），
Python/Bash语法、完整源码/AST差异白名单、105个原文件保护和模拟提交检查通过。
报告为`audit_reports/mizar_s1_mesh_ced_layer84_validation.json`；路线内`reference_audit.json`记录母本对照。
具备原依赖时可手动加`--require-tensors`禁止skip；不下载资产、不用CUDA、不提交作业，不是formal前置任务。
真实远程权重加载、CPU tensor、8GPU训练与完整恢复尚未执行，未登记远程PASS。

## 2026-11-02：音频七槽 split3 + read 后零门控残差的独立训练

已从 split3 完整复制隔离出
[split3_inject 正式训练路线](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_audio7slot_split3_inject/README.md)。
保留音频七槽/文本五槽、六个音频 router 的 slot0 三等分初始化及两套 MLP。
同一对 A4/A8 先初始化音频 memory 槽5/6，再在 pre/loop0 memory read 后、
第一/第二轮循环前分别做残差注入；gate 输入明确为 memory read 后的音频 hidden：
`h_audio += tanh(Linear(h_audio)) * A`。
两个独立 Linear(576,1) 的 weight/bias 均为0，每个音频token一个标量；
仅音频前126位置直接注入，无第三次注入或额外 memory write。
循环后 query 仍取含残差的循环输入，write value 仍取循环输出。

仅增加1,154个gate参数；相对原五槽CED MeSH新增合计2,469,164。
原LM/文本router maxLR3e-4；mapper/MLP/音频router/两个gate maxLR6e-4。
零gate时初始forward与split3对齐；MLP仍可经已有memory路径取得首步梯度。
CED冻结/eval且每批一次编码，A4/A8的MLP不重复计算。
数据、FP32、683-token布局、loss、GBS256（8×8×4）、3epochs/17313updates/866warmup、
优化器/调度/GradNorm、后台staging和完整恢复配方保留。
原split3、K7、48listen、旧inject及共享模型未修改。

本地代码经GitHub同步后，在远程登录节点直接提交formal，无自动smoke前置任务：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_split3_inject_5090.sh --stage-workers 4
```

资源仍为 `pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`，环境 `mellow_c8204d8`。
默认输出根为
`outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_inject_5090/formal_<timestamp>/`，
内部最终 `formal/checkpoints/model--epo-3.ckpt` 和 `model--step-17313.ckpt`。
新route/output/log/index-cache/tmpfs/resume身份隔离，只接受本路线完整checkpoint恢复；
恢复不会清零已训练gate或重新三等分router。默认仍从原文本checkpoint初始化，
未实现已训练split3音频checkpoint权重迁移；现有评测器需要独立适配。
首两次及每25次update记录两gate幅度/符号/饱和比例/残差范数比及梯度。

统一验证入口 `python -B tests/validate_mizar_s1_mesh_ced_audio7slot_split3_inject.py`。
本地70项实际检查通过、Python/Bash语法通过；16项真实CPU tensor检查因缺依赖跳过。
实际forward/gate的NumPy检查覆盖零gate母本对照、非零gate读取post-read hidden、
两次注入边界、音频范围和普通memory写入；完整训练循环AST对照及母本回归通过。
报告 `audit_reports/mizar_s1_mesh_ced_audio7slot_split3_inject_validation.json`。
依赖齐备时可手动加 `--require-tensors` 禁止skip，不下载资产、不用CUDA、不提交作业，
不是formal前置任务。远程真实权重加载、8GPU训练与恢复未执行，不登记远程PASS。

## 2026-11-01：音频七槽 split3 独立训练——原第一槽概率三等分

已完整复制K=7七槽版本，新增[独立 split3 正式训练路线](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_audio7slot_split3/README.md)。
六个音频write/read router分别从对应原五路router初始化：原第一行权重复制到零基行0/5/6，
这三行bias均为原第一行bias减log(3)；原行1..4保持不变。
相同query下初始概率为`[p0/3,p1,p2,p3,p4,p0/3,p0/3]`，其他四槽及softmax分母不变。
三行可独立训练，不再使用K或新增行零权重；六个文本router仍为原五槽。
音频memory仍在prelude前初始化`[E,0,0,0,0,A4,A8]`，不是把新增槽内容改为E。
后续无encoder重复注入，保留全部槽普通write/read；不对三等分产生的读写贡献变化额外补偿乘3。

原六个文本router/LM maxLR3e-4，mapper/两套MLP/六个音频router maxLR6e-4；
数据、loss、FP32、GBS256（8×8×4）、3epochs/17313updates/866warmup、优化器/调度/GradNorm、
staging和完整恢复配方与K=7版本保持一致。模型主forward和memory交互代码对照相同，
新增参数量仍为2,468,010。新route/output/log/index-cache/tmpfs和resume身份隔离，
原K=7、48listen、CED MeSH及共享模型未修改。

本地改动经GitHub同步后，在远程登录节点直接提交formal，无自动smoke前置任务：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_split3_5090.sh --stage-workers 4
```

资源仍为`pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`，环境`mellow_c8204d8`。
默认输出根为`outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_split3_5090/formal_<timestamp>/`，
内部最终`formal/checkpoints/model--epo-3.ckpt`和`model--step-17313.ckpt`。
只接受本split3路线的完整checkpoint恢复，不重置已训练的三行；原K=7音频checkpoint不能full resume。
诊断增加槽0/5/6合计概率及差异，概率无固定上限。完整提交、记录和恢复说明见路线README。

本地统一入口`python -B tests/validate_mizar_s1_mesh_ced_audio7slot_split3.py`，
51项实际检查通过、Python/Bash语法通过；13项真实CPU tensor检查因缺依赖跳过。
报告`audit_reports/mizar_s1_mesh_ced_audio7slot_split3_validation.json`。
具备原依赖时可手动加`--require-tensors`禁止skip；不下载资产、不用CUDA、不提交作业，不是formal前置任务。
远程真实权重加载、8GPU正式训练和完整恢复尚未执行，不登记远程PASS；现有评测器需要独立适配。

## 2026-11-01：CED4/8 初始音频七槽、文本五槽的独立正式训练路线

已复制48listen并实现[隔离 audio7slot 路线](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_audio7slot/README.md)。
冻结CED一次提取第4/8层，经原两套独立MLP得到A4/A8，在prelude前与input embedding同时初始化：
音频七槽`[embedding,0,0,0,0,A4,A8]`，SEP/文本仍五槽。
六个新增音频write/read router分别复制对应原router前五行，新增两行weight=0、bias=-7；
原六个router继续仅用于SEP/文本。删除旧gate/两次listen注入，七槽后续均参与普通write/read。
保留30次层执行、三次交互及循环input-query/output-value语义，每个完整forward重建memory，无KV cache。

原LM及六个文本router maxLR3e-4；原mapper、两套MLP和六个新增音频router maxLR6e-4；
共新增2,468,010参数。原数据/683-token布局/loss/FP32/AdamW/LambdaLR/GradNorm/GradScaler、
GBS256（8×8×4）/3epochs/17,313updates/866warmup、后台staging和完整恢复配方保留。
源码AST核验原训练循环仅增加诊断，原CED MeSH、48listen和共享模型未修改。
新route/output/log/index-cache/tmpfs/resume身份隔离；完整恢复不会把已训练新行重置为0/-7。
首两次update和每25次记录七槽概率、两新槽合计阈值比例及梯度/RMS；K只控制初始化，不限制后续学习。

经GitHub同步后，在远程登录节点直接提交formal，无自动smoke前置任务：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_5090.sh --stage-workers 4
```

资源仍为`pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`，环境`mellow_c8204d8`。
默认输出`outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_audio7slot_5090/formal_<timestamp>/`，
其内最终`formal/checkpoints/model--epo-3.ckpt`及`model--step-17313.ckpt`。
完整结构、诊断字段和恢复命令见路线README；原评测器需后续适配。

统一检查入口`python -B tests/validate_mizar_s1_mesh_ced_audio7slot.py`，报告
`audit_reports/mizar_s1_mesh_ced_audio7slot_validation.json`；本地34项实际检查通过、Python/Bash语法通过，
12项真实CPU tensor检查因缺依赖明确skip，
具备原依赖的环境可手动加`--require-tensors`禁止skip。无资产下载/CUDA/作业提交，不是formal前置任务。
本地检查不等于远程真实权重、8 GPU正式训练或完整恢复通过。

## 2026-11-01：音频七槽设计的原五槽 MeSH 文本 router K 标定工具

新增独立的[第一阶段文本生成与K标定工具](code/RSmol/scripts/README_mesh_text_router_k.md)，
不修改训练路线或共享MeSH实现。使用两条MeSH CED训练入口共同的原文本初始化
`formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244`，
显式读取用户填写的十条`id/prompt` JSONL，FP32/eval、纯argmax、无KV cache、
每条最多100个新token、首次EOS停止；不加载CED或音频checkpoint，不注册七槽模块。
只读hooks记录原六个router的逐token logits、五槽概率和logsumexp，
每位置只记一次，另加末token观察前向但不生成第101个token。

按假设扩展`[z1..z5,-K,-K]`计算两新增槽**合计**概率，默认全部记录位置/六个router
均低于0.5%，取最坏位置边界再加1.0 logit余量、向上取整至0.1作为共用K。
报告保留每router差异、prompt/生成段统计和1%/0.5%候选边界；纯文本结果不保证音频输入上界。
模板创建/输入检查/CSV离线重算只用标准库。远程填好十条prompt后从`code/RSmol/`提交：

```bash
bash run_mesh_text_router_k_5090.sh \
  --prompts-jsonl /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mesh_text_router_k_5090/inputs/prompts.jsonl
```

单卡`pdgpu-5090 / 8 CPU / 32G / 1 GPU / 1 node`、`mellow_c8204d8`，
输出默认`outputs/RSmol/mesh_text_router_k_5090/run_<timestamp>/`，无smoke前置任务。
模板填写、完整提交和结果文件见上述README。统一验证入口为
`python -B tests/validate_mesh_text_router_k.py`，本地22项实际检查通过、
5项真实CPU tensor检查因缺torch/transformers跳过，Python/Bash语法通过；
报告为`audit_reports/mesh_text_router_k_validation.json`。
远程真实权重、文本生成和K数值尚未执行，不登记远程PASS。

## 2026-10-31：五条 CED 多层交互 checkpoint 的 MMAU / MMAR 评测扩展

已扩展两个现有提交入口 `run_mizar_s1_ced_small_mmau_eval_5090.sh`、
`run_mizar_s1_ced_small_mmar_eval_5090.sh`，新增五个 `RSMOL_MIZAR_CED_EVAL_VARIANT`：
`recursive_5_10_5_layer36912_crossattn`、`recursive_5_10_5_layer36912_query32_crossattn`、
`recursive_mesh_layer4_8_inject`、`recursive_mesh_layer4_8_listen`、
`recursive_mesh_layer36912_listen`。
模型适配、统一runner及compute wrapper位于训练目录之外；训练源码、共享递归模型和旧评测实现未修改。

三条已有产物分别默认绑定用户提供的 `formal_20261030_140459065784533`、
`formal_20261030_162028139130855`、`formal_20261030_172450546025072`，
加载各自 `formal/checkpoints/model--step-17313.ckpt` 与同run的 `formal/runtime.json`。
MeSH layer4_8_inject、layer36912_listen没有猜测默认run，须显式指定实际run或完整路径。

统一保留原CED的fixed20/32kHz mono、固定选项顺序和小写prompt、126+1+256 prefix383、
FP32/eval/inference mode/TF32关闭、attention_mask=None、无KV cache/逐token完整重算/纯argmax、
最多300新tokens/首次EOS停止、完整原样输出及原官方scorer；MMAU/MMAR各1000条、零skip。
每样本CED只编码一次；额外grids/(A4,A8)/prompt mask按各训练接口显式逐步传递，
LM内投影/query32压缩照常每次forward执行，MeSH memory每次完整forward重建。
query32输入prefix仍为383，实际合同为v1；四层MeSH listen加载逐token gate v2，无固定γ或/4缩放。

strict加载完整音频checkpoint并校验route/runtime/源码/资产/训练geometry和实际结构，
只读hooks检查30次物理层执行、两次交互、routers调用和query32压缩；不修改训练forward。
资源仍为pdgpu-5090 / 8 CPU / 32G / 1 GPU / 1 node、mellow_c8204d8，直接full评测。
三份现有checkpoint的六条提交命令、另外两条完成后的四条命令、输出和续跑说明见
[五路线评测说明](code/RSmol/scripts/README_mizar_s1_ced_multilayer_eval.md)。

本地汇总入口 `python -B tests/validate_mizar_s1_ced_multilayer_eval.py`：
93项实际检查通过，18项真实CPU tensor检查因缺Torch/Transformers跳过（新路线9项、旧回归9项），
Python/Bash语法通过。包含十种1000条模拟流程、断点续跑、原始输出/官方scorer分母、
十种提交组合与路径覆盖、旧评测回归。报告为
`audit_reports/mizar_s1_ced_multilayer_eval_validation.json`。
具备原依赖的环境可手动加 `--require-tensors` 禁止任何tensor跳过；无下载、无CUDA、无作业提交，
不作为full评测自动前置任务。三份远程权重内部合同/runtime、真实GPU推理和正式分数仍待服务器核验，
未登记远程PASS。

## 2026-10-30：MeSH CED 第 4→8 层在两次 memory read 后残差注入

新增 [MeSH CED 4→8 隔离路线](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject/README.md)，
完整复制自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small`。
保留20物理层/30逻辑执行、两次循环、5个瞬时memory slots和6个routers。
冻结CED在一次编码中提取第4、8个block最终LN之前的完整输出，经两套独立
`LN1536 -> Linear1536/576 -> GELU -> Dropout0 -> Linear576/576 -> LN576` 得到A4/A8。
在 `read_routers[0]`、`read_routers[1]` 的memory read之后，分别按原非MeSH 4→8方法
`h_audio += tanh(Linear576/1(h_audio)) * A` 注入前126个音频位置，再进入对应中段循环。
两个gate独立且weight/bias全零；第三次read后不注入，SEP/文本不直接注入。

新增2,444,930个参数；新增MLP/gate和原mapper maxLR6e-4，原LM/routers maxLR3e-4。
保留母本683-token布局、hybrid 1,477,492行、FP32、GBS256（8×8×4）、3epochs /
17,313updates / 866warmup、loss、优化器/调度/GradNorm、数据顺序、后台staging和完整恢复。
新实现、入口、输出、日志、index cache、tmpfs与resume身份隔离；原两条路线及共享MeSH实现未修改。

经GitHub同步后，在远程登录节点直接提交formal，无smoke前置任务：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_inject_5090.sh --stage-workers 4
```

沿用mellow_c8204d8和pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node。
默认输出根为 `outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_5090/formal_<timestamp>/`，
最终产物为其中 `formal/checkpoints/model--epo-3.ckpt`，同时保留 `model--step-17313.ckpt`。
仅支持本新路线完整checkpoint resume；现有普通MeSH CED evaluator尚未适配新架构。

本地汇总：`python -B tests/validate_mizar_s1_mesh_ced_layer48.py`，66项实际检查通过，
16项真实CPU tensor检查因缺依赖跳过（新路线9项、原4→8路线7项）。Python/Bash语法通过。
报告为 `audit_reports/mizar_s1_mesh_ced_layer48_validation.json`。
远程可手动加 `--require-tensors` 禁止跳过；不下载资产、不用CUDA、不提交作业，不作为formal前置任务。
真实权重、远程tensor、8 GPU训练及完整恢复待服务器确认，不登记远程PASS。

## 2026-10-30：MeSH CED 四层 audio listen 更新为逐音频 token gate v2

已从 `run_mizar_s1_gbs256_recursive_mesh_ced_small_5090.sh` 的母本复制隔离，完成
[四层 listen 路线](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer36912_listen/README.md)。
原最终层mapper及MeSH主路径、20物理层/30逻辑执行、五槽与三组原write/read routers保留。
单次冻结CED编码提取第3/6/9/12个block的最终LN前状态，按segment/time/frequency排列为
`[B,126,1536]`，四套独立非affine LN1536→Linear1536x576→非affine LN576生成音频bank。
Prelude后与第一次循环后，分别用有效prompt的归一化masked mean、独立Linear576x4及sigmoid
聚合四路特征；两个独立softmax listen routers写入原记忆前126个audio位置。
顺序为原write→listen→原read；循环2后仍只有原write/read。
原循环router仍用循环输入计算权重，循环输出作为原写入值。

原入口和实现就地更新：两个标量β替换成两套独立 `Linear(576,1) → tanh`，分别直接读取
Prelude和第一次循环输出的原始音频潜状态，为每个音频token产生幅度，weight/bias均全零。
保留四路聚合A_r和原 `LN_query → Linear(576,5) → softmax` 槽位概率P_r，
写入为 `delta_M[r,s,t] = tanh(W_r H_audio[r,t] + b_r) * P_r[t,s] * A_r[t]`。
**删除固定γ和/4缩放**，初始新增写入仍为零。初版γ=0.3的RMS界限不再适用；
不额外添加缩放、归一化或梯度裁剪。
新增分支共3,552,788个参数、20个tensors，比标量初版增加1,152个参数；
全部新增LN非affine、eps1e-5；新参数max LR6e-4，矩阵decay0.01/bias不衰减沿用原分组。
保留原LM LR3e-4、mapper LR6e-4、loss、AdamW/LambdaLR/GradNorm、FP32、683-token布局、
hybrid 1,477,492行、GBS256（8×8×4）、3 epochs/17,313 updates/866 warmup及后台staging。
新增模块在原文本权重strict加载后注册并隔离CPU RNG；保留初版投影/router精确初值。
入口、输出、日志、index cache、tmpfs及resume身份独立。
完整恢复额外保存/校验梯度解锁状态与optimizer步数；完成checkpoint可审计但不能重新resume。
架构身份为 `ced_small_mizar_s1_fixed20_gbs256_recursive_mesh_layer36912_listen_v2`；
旧标量gate checkpoint不能完整resume，从原文本checkpoint在新的空输出目录开始训练。

GitHub同步后，在远程登录节点直接提交formal，无smoke或tensor检查自动前置任务：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_layer36912_listen_5090.sh --stage-workers 4
```

资源为 `pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`、`mellow_c8204d8`。
默认run根为 `outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_layer36912_listen_5090/formal_<timestamp>/`，
最终为其中 `formal/checkpoints/model--epo-3.ckpt` 与 `model--step-17313.ckpt`。
恢复命令及诊断字段见路线README；原母本和共享MeSH实现未修改，普通CED evaluator尚未适配新架构。

本地统一入口 `python -B tests/validate_mizar_s1_mesh_ced_listen.py`，结果见
`audit_reports/mizar_s1_mesh_ced_listen_validation.json`：49项实际检查通过，32项真实CPU tensor检查因缺依赖跳过；
另通过Python/Bash语法及母本源码对照。远程可手动加 `--require-tensors` 强制tensor全部实际运行，
不下载资产、不用CUDA训练、不提交作业。真实权重、8 GPU稳定性与恢复仍待远程确认，不登记远程PASS。

## 2026-10-30：四条非 MeSH CED checkpoint 的 MMAU / MMAR 评测扩展

两个既有入口 `run_mizar_s1_ced_small_mmau_eval_5090.sh`、
`run_mizar_s1_ced_small_mmar_eval_5090.sh` 新增四个
`RSMOL_MIZAR_CED_EVAL_VARIANT`：`recursive_5_10_5`、
`recursive_5_10_5_layer8_4_inject`、`recursive_5_10_5_layer4_8_inject`、
`recursive_5_10_5_audio_reinject`。分别绑定用户指定的四个step17313 checkpoint及同run的
`formal/runtime.json`；默认base和已有recursive_mesh仍走原实现。
新增runner/adapter/compute wrapper位于训练目录之外，四条训练路线与共享递归实现未改。

保持原CED的fixed20、126+1+256 prefix383、FP32/eval/TF32关闭、
attention_mask=None、逐token完整重算/无KV cache/纯argmax、最多300新tokens/首次EOS停止，
保留完整生成原样评分，MMAU/MMAR各1000条及原官方scorer。
两条层注入路线每样本编码一次，始终传原顺序(A4,A8)；8→4选择由模型内部完成。
strict加载完整音频权重，并核对训练/runtime/源码/资产身份和实际30层轨迹、注入顺序。
资源仍为pdgpu-5090 / 8 CPU / 32G / 1 GPU / 1 node、mellow_c8204d8；
直接full评测，无smoke前置任务。四层cross attention不在本次评测适配范围。

八条提交命令、四个完整run路径、结果目录、续跑与覆盖参数见
[非MeSH CED评测交付](code/RSmol/scripts/README_mizar_s1_ced_recursive_eval.md)。
本地汇总 `python -B tests/validate_mizar_s1_ced_recursive_eval.py`：
65项实际检查通过（新增28项、原CED/MeSH37项），9项真实CPU tensor检查因缺依赖跳过，
Python及Bash语法通过。报告为 `audit_reports/mizar_s1_ced_recursive_eval_validation.json`。
远程可手动加`--require-tensors`要求tensor实际运行；不下载、不用CUDA、不提交作业，
不作为full自动前置任务。真实checkpoint/配套runtime/资产一致性、GPU推理与官方分数待远程确认，
不登记远程PASS。

## 2026-10-30：四层 CED cross attention 的分支归一化与固定注入尺度 v2

同日最新调整：按用户要求将固定 γ 从0.1改为 **0.3**，以下描述已同步。
其余架构与训练配置不变；旧γ=0.1 checkpoint因合同/源码身份不同，不能直接full resume。

用户反馈原四层 cross attention 训练可以运行，但出现注意力残差放大及梯度爆炸。
已在[现有隔离路线](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_crossattn/README.md)落实两项架构调整：
四个深度各自采用 `LN1536 -> Linear(1536,576) -> LN576`；
共享 `LN_text(T)` 仅供有效prompt的Gate1池化、Gate2和CA query使用，原T仍作为残差基底。
CA输出经过共享且 `elementwise_affine=False` 的LN，按
`T_new = T + 0.3 * tanh(Gate2(LN_text(T))) * LN_out(CA(...))` 注入。
轮次偏置仍只在文本LN之后加入query，Gate2仍全零初始化，CA输出投影保持原随机初始化。
所有LN的eps为1e-5，其余LN可训练weight=1/bias=0；两轮共享文本/输出LN、CA及两个gate。
每个token的实际注入RMS在数值有限时不超过约0.3，不代表主路径的30%或全部梯度稳定性证明。

新增18,048个LN参数，交互模块共4,892,741个参数；新增LN沿用分支max LR6e-4、
一维参数不衰减。原LM LR3e-4、mapper LR6e-4、AdamW/LambdaLR/GradNormTracker、
loss、数据顺序、FP32、GBS256（8×8×4）、3 epochs / 17,313 updates / 866 warmup、
683-token布局、后台staging、保存节点和提交资源全部保留。
runtime/checkpoint架构身份升级为 `ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer36912_crossattn_v2`，
拒绝旧v1音频checkpoint的完整resume；从原文本checkpoint重新初始化，使用新的空输出目录。

经GitHub同步后，在远程登录节点直接提交formal，不传resume参数，无smoke前置任务：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer36912_crossattn_5090.sh --stage-workers 4
```

默认新时间戳run中的最终产物仍为 `formal/checkpoints/model--epo-3.ckpt` 和
`model--step-17313.ckpt`。metrics区分原始CA输出RMS、归一化后RMS、实际残差RMS，
并记录 `residual_token_rms_max` 与固定 `residual_scale=0.3`。
本地58项实际检查通过、15项真实CPU tensor检查因缺少依赖跳过，汇总入口仍为
`python tests/validate_mizar_s1_ced_multilayer_crossattn.py`。
v2真实tensor、远程权重加载、8 GPU训练稳定性与完整恢复待远程确认，不登记远程PASS。
旧路线与共享递归实现未修改；现有普通CED evaluator仍未适配本架构。

## 2026-10-29：CED 第 3/6/9/12 层选择与两轮共享 cross attention

已接续完成从非 MeSH CED 母本复制隔离的
[四层 cross attention 路线](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_crossattn/README.md)。
冻结 CED，在同一次编码中提取第 3/6/9/12 个 block 最终 LayerNorm 之前的输出，
分别按 segment/time/frequency 顺序整理为 `[B,126,1536]`，经四个独立 Linear 得到 `[B,126,576]`。
原最终层 mapper 和音频 prefix 主路径保留。

prelude 五层后及第一次中段循环后（零基逻辑索引 5、15），
共享 Gate1 对有效 prompt 的 masked mean 产生四层 softmax 权重，混合音频作为 K/V；
当前全部文本位置作为 Q 输入，共享 hidden576 / 9-head / dropout0 cross attention。
两个独立零初值轮次偏置仅加到 Q 输入；共享逐文本 token 的 tanh Gate2 零初始化，
将 cross attention 输出残差加到文本位置，audio/SEP 不直接注入。
Gate1 不读取 answer，LM 保留 `attention_mask=None`，推理要求完整 prefix 重算、无 KV cache。
共新增 4,874,693 个参数；新增参数及原 mapper max LR6e-4，原 LM max LR3e-4。

保持母本 hybrid 1,477,492 行、683-token 布局、FP32、GBS256（8×8×4）、
3 epochs / 17,313 updates / 866 warmup、原 loss/AdamW/LambdaLR/GradNorm、
后台 staging 和完整恢复。新代码、入口、输出、日志、index cache、tmpfs 与 resume 身份隔离；
母本及共享递归实现未修改。现有普通 CED evaluator 尚未适配本架构。

经 GitHub 同步后，在远程登录节点直接提交 formal，无 smoke 前置任务：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer36912_crossattn_5090.sh --stage-workers 4
```

沿用 `mellow_c8204d8`、`pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`。
默认输出为 `outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer36912_crossattn_5090/formal_<timestamp>/`；
最终产物为其中 `formal/checkpoints/model--epo-3.ckpt`，同时保留 `model--step-17313.ckpt`。

本地汇总入口：`python tests/validate_mizar_s1_ced_multilayer_crossattn.py`。
56 项实际检查通过，14 项真实 CPU tensor 检查因缺少依赖跳过；
包括 Python/Bash 语法、NumPy 行为/因果性、strict loader、参数/LR、Bash 模拟提交、
共享启动及退出清理、母本源码/AST 对齐和母本回归。
收尾修正了未执行 tensor 测试对首步 LR 的错误假设：母本实际从 0.1 倍 max LR 起步，
Gate2 首次更新后即可在下一次反向传递上游梯度；修正不代表 tensor 测试已通过。
真实权重加载及 8 GPU 训练/恢复待远程确认，不登记远程 PASS。
远程可手动执行 `python tests/validate_mizar_s1_ced_multilayer_crossattn.py --require-tensors`，
不下载资产、不运行 CUDA、不提交训练，不作为 formal 自动前置任务。

## 2026-10-29：CED 第 8→4 层递归注入独立训练路线

完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject`，
新路线为 `mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject`。
唯一训练语义变化：prelude 五层输出 h0 后、第一次中段循环前注入 CED 第 8 层 A8；
第一次中段循环输出 h1 后、第二次循环前注入 CED 第 4 层 A4。
对应零基逻辑索引为 5、15，实际审计轨迹为 `[[5,8],[15,4]]`。
特征接口仍为 `(A4,A8)`；A8 使用第 8 层独立 MLP/gate，A4 使用第 4 层独立 MLP/gate。
两套模块的注册、初始化顺序保持 4、8，零初始化 gate、参数量及原 CPU RNG 行为不变。

冻结 CED、原 final-layer mapper、683-token 布局、非 MeSH `5+10×2+5`、hybrid 1,477,492 行、
FP32、GBS256（8×8×4）、3 epochs / 17,313 updates / 866 warmup、LM LR3e-4、
mapper/MLP/gate LR6e-4、loss、优化器、调度、GradNorm、后台 staging 和完整恢复均保留母本。
新 route、输出、日志、index cache、tmpfs 和 checkpoint resume 身份隔离，拒绝原 4→8 路线 checkpoint。
原 4→8 路线及共享递归实现未修改；现有评测入口尚未适配新路线。

经 GitHub 同步后，在远程登录节点提交 formal，无 smoke 前置任务：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
git pull --ff-only
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_5090.sh --stage-workers 4
```

资源为 `pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`，沿用 `mellow_c8204d8`。
默认输出根为 `outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_5090/formal_<timestamp>/`，
最终产物为该 run 内的 `formal/checkpoints/model--epo-3.ckpt`，同时保留 `model--step-17313.ckpt`。
完整结构、恢复和可选验证命令见
[8→4 路线 README](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject/README.md)。

本地验证：`python tests/validate_mizar_s1_ced_layer84.py --output audit_reports/mizar_s1_ced_layer84_validation.json`，
共 67 项实际执行通过（含原 4→8 及普通 CED 母本回归），18 项真实 CPU tensor 检查因缺少依赖跳过；
Python 编译、两份 Bash 语法、模拟提交和 4→8 母本完整源码语义对照审计通过。
真实远程权重加载与 8 GPU 训练尚未验证，不登记远程 PASS。
可在远程仓库根目录手动执行 `python tests/validate_mizar_s1_ced_layer84.py --require-tensors`，
该命令不下载资产、不运行 CUDA、不提交训练，也不会在 formal 前自动执行。

### 2026-10-29：CED 第 4、8 层递归注入路线

已基于 `run_mizar_s1_gbs256_recursive_5_10_5_ced_small_5090.sh`
复制建立独立训练路线，实现代码、输出目录及 checkpoint 身份校验均已隔离。

正式训练入口：
`code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4_8_inject_5090.sh`

架构与训练配置：

- 保留原有 CED 最终层、mapper 和 LM 主干前向流程。
- 冻结 CED-small，额外提取第 4、8 层潜状态。
- 中间状态从 `[2B,252,384]` 按原 mapper 的时间顺序整理为
  `[B,126,1536]`，分别经独立 MLP 转换为 `[B,126,576]`。
- prelude 五层输出 h0，在第一次循环前注入 A4；
  第一次循环输出 h1，在第二次循环前注入 A8。
- 注入公式为 `h_audio += tanh(Linear(h_audio)) * A`。
  每个 audio token 使用一个标量门控，文本及 SEP 位置保持原有流程。
- 两个 gate 的权重和偏置均初始化为零。
- mapper、新增 MLP 和 gate 的 max LR 均为 `6e-4`；
  LM 的 max LR 为 `3e-4`。
- 沿用母本 GBS 256、3 epochs 和学习率调度。

验证状态：

- 本地验证结果：49 项通过，10 项因缺少 PyTorch 等依赖而跳过。
- 真实张量、远程权重及 GPU 训练尚未验证。
- 尚未执行 Git commit、push 或远程训练提交。
- 当前不包含这条新增架构的独立评测入口。

本地只维护核心代码；模型权重、数据集和正式训练环境位于远程
Linux 服务器，继续通过 GitHub 同步代码。

远程仓库更新后，正式提交命令：

```bash
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4_8_inject_5090.sh
```

入口通过现有作业包装器申请单节点 8 张 5090。

如需先补齐依赖实际张量计算的验证，可在具备项目依赖的服务器环境执行：

```bash
python tests/validate_mizar_s1_ced_layer48.py --require-tensors
```


## 2026-10-28：CED 非 MeSH 5+10×2+5 的两次循环前音频残差注入

新增 [audio_reinject 隔离路线](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_audio_reinject/README.md)，完整复制自非 MeSH CED 朴素循环母本。原20个物理层、30次逻辑执行和 `0..14,5..14,15..19` 轨迹不变。原 mapper 输出的 `[B,126,576]` 音频 embeddings 额外经一个共享 `Linear(576,576)` 得到 A；分别在前五层后、第一次中段循环后，用同一个 `tanh(Linear(576,1)(h_audio))` 逐音频 token 标量门控乘 A，残差加回前126个音频位置，再进入对应中段循环。A 每次 forward 只计算一次，SEP/文本位置不直接注入，第二次循环后不注入；不加额外归一化、dropout 或 detach。

投影权重随机初始化、bias为零，gate weight/bias全零，共新增332,929个参数；初始化隔离CPU RNG，先严格加载原文本权重再注册新模块。新增两组参数采用mapper的LR6e-4，原LM LR3e-4及所有原参数分组规则不变。保持冻结CED、官方freq_merge_126、683-token布局、原answer-only loss、hybrid 1,477,492行、FP32、8 GPU × microbatch8 × accumulation4、3epochs / 17,313updates / 866warmup，以及原优化器、调度、GradNorm、数据顺序、后台staging、保存节点和完整恢复行为。母本和共享递归模型未修改。

经GitHub同步后，从远程登录节点直接提交formal，无smoke前置任务：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_audio_reinject_5090.sh --stage-workers 4
```

使用 `mellow_c8204d8`、`pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`。输出、日志、index cache、tmpfs与resume身份独立，默认输出根为 `outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_audio_reinject_5090/`，最终产物为run下的 `formal/checkpoints/model--epo-3.ckpt`，也保留 `model--step-17313.ckpt`。

本地49项行为/合同/NumPy数值/Bash mock及母本回归检查、Python编译和Bash语法检查通过；7项新路线及3项母本真实CPU tensor测试因依赖缺失跳过，继承的16项runtime checks未执行。汇总入口为 `python tests/validate_mizar_s1_ced_audio_reinject.py`；远程可加 `--require-tensors` 禁止依赖缺失或跳过被误判为张量验收通过。真实权重加载和8GPU训练尚未验证，未登记远程PASS；原有评测入口尚未适配新架构。详细结构、梯度预期、对齐审计、正式提交和恢复命令见路线README。

## 2026-10-28?CED-Small / ? MeSH 5+10?2+5 / Mizar S1 GBS256 ????

??[? MeSH CED ????](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small/README.md)?????? MeSH CED ??????? MeSH ??? loader?????????????????? `stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244`?20 ???????30 ???????? `0..14,5..14,15..19`?? router / memory slots??? mapper ? LM ???????? CED ???? HTSAT ? MeSH ?????????????

? `run_mizar_s1_ced_small_gbs256_5090.sh`?`run_mizar_s1_gbs256_recursive_mesh_ced_small_5090.sh` ?????????? eval ? CED-Small??? freq_merge_126??? fixed20 ?? FP32 ???683-token ??/prefix383?hybrid 1,477,492 ??8 GPU ? microbatch8 ? accumulation4?3 epochs / 17,313 updates / 866 warmup?AdamW/LambdaLR/GradNorm?????/????/????????? MeSH masked answer-only CE??? tokenizer mask ???? CED ? ignore-PAD CE ????????? loss/???????S1 ???????????????index cache?tmpfs ? resume ?????

? GitHub ??????????????? formal?? smoke ?????

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_gbs256_recursive_5_10_5_ced_small_5090.sh \
  --ced-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

?? `mellow_c8204d8`?????? `pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`???????? output-dir ?? `formal/checkpoints/model--epo-3.ckpt`????? `model--step-17313.ckpt`????? resume ???????????/??/??/??? checkpoint????????????? tensor ????? README?

??33?stdlib/??/Bash mock???49?????????/?????????3??????CPU tensor?????????????16?????tensor??????????8 GPU??????????????????PASS?????????????????????evaluator?????checkpoint?

## 2026-10-27：两次循环、六记忆槽 MeSH 文本完整一轮训练

新增 [六槽隔离文本路线](code/RSmol/text_5_10x2_5_mesh_6slot/README.md)。模型与转换器复制自 `text_5_10x2_5_mesh_7slot`，保留 20 个物理层、30 次逻辑执行、`5+10×2+5` 和三组独立 write/read routers；仅将 memory slots 和 router 输出维度改为六，router 初始化按原公式使用 `sqrt(2/(6×hidden_size))`。从干净的 `models/SmolLM2-5-10-5` 或原始 SmolLM2 转换，逐 tensor 检查继承权重；不用已有五槽/七槽训练 checkpoint 初始化。

训练结合五槽 FP32 实现：8 GPU × microbatch 8 × accumulation 16 = GBS1024、context 1024、完整 1 epoch / 9,244 updates、463 warmup、max LR `8e-4` / min LR `8e-5`，保留原 AdamW 分组、全局有效 token 加权文本 NTP loss、clip 1.0、persistent Parquet 分片/首轮顺序、每 500 步及最终步保存并保留最近三份。采用五槽已有的实际跳行恢复，补齐各 rank Python/CPU/CUDA RNG，拒绝跨槽位/精度/调度合同 resume。五槽、七槽和音频路线代码未改。

GitHub 同步后在远程登录节点 CPU-only 转换：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/scripts/convert_stepwise_5_10x2_5_mesh_6slot.sh
```

随后直接提交正式训练（无 smoke 报告前置门槛）：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_stage4_5_10x2_5_mesh_6slot_formal_3090.sh
```

使用 `rsmol` 和 `pdgpu-3090 / 32 CPU / 256G / 8 GPU / 1 node`。转换默认输出 `models/SmolLM2-5-10x2-5-mesh-6slot`，训练默认写入 `outputs/RSmol/stage4_5_10x2_5_mesh_6slot/formal_1epoch_fp32_<timestamp>/`，最终为其中的 `checkpoint-009244/` 与 `stage4_gate_FORMAL_audit.json`。另保留独立 10+2 短验证入口和显式 formal resume，详细命令见路线 README。

本地新增 23 项依赖轻量行为/合同/Bash mock 检查和现有 23 项五槽/七槽回归通过；另提供 3 项真实小模型 CPU tensor 检查，本地缺少 Torch/Transformers，均跳过，留待远程 `rsmol` 执行。转换权重和真实 8 GPU 训练尚无远程执行证据，不登记为远程 PASS。

## 2026-10-27：Mizar S1 / MeSH 改用 2026-09-07 checkpoint 初始化

新增 [MeSH init0907 隔离训练路线](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_init0907/README.md)，完整复制自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh`。仅将默认文本初始化改为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_resume_000500_nonfatal_router_20260907_115533/checkpoint-009244
```

其余模型、HTSAT/c2l/projection、hybrid 1,477,492 行数据、fixed20、815-token 序列、padding/loss、FP32、优化器、LR、GradNorm、staging 和保存/恢复实现均沿用母本。仍为 8 GPU × microbatch 8 × accumulation 4、3 epochs、17,313 updates、866 warmup。只加载文本模型和 tokenizer 开始新的音频训练，optimizer/scheduler/RNG/数据游标从头开始；原路线及共享 MeSH 实现未修改。

新入口与输出、日志、index cache、tmpfs、resume 身份独立；直接提交 formal，不增加 smoke 或审计前置任务。GitHub 同步后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_init0907_5090.sh --stage-workers 4
```

资源保持 `pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`、`mellow_c8204d8`。默认输出为 `outputs/RSmol/mizar_s1_htsat_gbs256_recursive_mesh_init0907_5090/formal_<timestamp>/`，最终产物为其下的 `formal/checkpoints/model--epo-3.ckpt`，同时保存 `model--step-17313.ckpt`。详细提交和恢复命令见路线 README。本地 Python 编译、两份 Bash 语法检查及 31 项现有回归检查通过。继承的 `reference_audit.json` 仅为母本历史记录；本地不据此声明新 checkpoint 的权重加载或 GPU 训练 PASS，真实执行以远程作业为准。

## 2026-10-27：USAD2-Small + 两次循环 MeSH / Mizar S1 / GBS256

新增 [USAD2-Small + MeSH 正式训练路线](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_usad2_small/README.md)，复制隔离自原 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh`。音频、mapper、数据及 staging 等 16 个文件与普通 USAD2 路线逐字节一致：整段 fixed20 波形一次编码，999 帧前加均值摘要，经原 Mellow projection 后仅对时间帧做相邻 4 帧均值，输出 251 audio tokens，prefix 508，完整序列 808。

文本端从既有 `checkpoint-009244` 初始化全量 MeSH：hidden 576、20 个物理层、30 个逻辑层、5+10×2+5、5 个临时 memory slots、6 个独立 routers；共享 MeSH 实现未修改。S1 optimizer/scheduler/RNG 重新开始，USAD2 冻结。保留 hybrid 1,477,492 行、8 GPU × microbatch 8 × accumulation 4、3 epochs / 17,313 updates / 866 warmup，以及普通 USAD2 的全部公共训练设置、保存步数和完整状态恢复。比较结果同时包含文本结构与初始化差异。

新入口为 `code/RSmol/run_mizar_s1_gbs256_recursive_mesh_usad2_small_5090.sh`，默认使用 `mellow_c8204d8`、远程 USAD2、MeSH checkpoint 和普通 SmolLM2 tokenizer；输出、日志、tmpfs、index cache 和 resume 身份隔离。启动时检查严格权重加载、tokenizer 一致性、真实循环调度与六个 routers 的梯度。详细提交、恢复和验证命令见上方路线 README。新增 32 项 stdlib 回归检查及官方/母版对照审计；远程 CPU tensor 检查与真实 8 GPU 训练仍须在服务器执行，本地未宣称通过。

## 2026-10-27：USAD2-Small / Mizar S1 / GBS256 独立训练路线

新增 [USAD2-Small 路线](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_usad2_small/README.md)，完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs`。保留原始 fixed20 音频构造，整段 20 秒波形从 32 kHz 重采样至 16 kHz，再一次输入冻结且始终 eval 的 USAD2-Small。本地包从远程 `/hpc_stor03/sjtu_home/jinwei.zhang/models/USAD2` 显式导入，严格加载 safetensors，避开 HF 动态模块缓存缺失传递依赖的问题；沿用已验证的 `mellow_c8204d8` 环境，无需升级依赖。

用户已实测 20s 输出 `x=(1,999,384)`、`x_lengths=[999]`、`mel=(1,1998,128)`。mapper 在原始特征前加一枚有效帧均值作为全局摘要，使用母版 Mellow `384→576` projection，再仅对时间帧做相邻 4 帧均值；最后 3 帧按 3 帧平均。输出 `1+250=251` 个音频 token；完整布局为 `251 audio + 1 SEP + 256 prompt + 300 answer = 808`，prefix 为 508，无卷积或额外可训练 CLS。

其余设置与 HTSAT 母版对齐：hybrid 1,477,492 行、8 GPU × microbatch 8 × accumulation 4、3 epochs / 17,313 updates / 866 warmup、FP32、全量 base SmolLM2、mapper LR 6e-4 / LM LR 3e-4、原 AdamW/LambdaLR/GradNorm、answer-only CE、共享音频立即训练与后台复制切换，以及完整 checkpoint/resume。原路线未修改；新输出、日志、index cache、tmpfs 和 resume 身份均隔离。

GitHub 同步后，在远程登录节点提交正式训练：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_usad2_small_gbs256_5090.sh \
  --usad2-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/USAD2 \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_usad2_small_gbs256_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

资源为 `pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`。本地新增 24 项检查及母版 18 项回归检查通过，官方文本/RNG、LR、GradNorm 与母版 AST 审计通过；本地无 PyTorch、真实权重和数据，额外 11 项 CPU tensor 检查及真实 GPU 集成训练待远程执行，不能记录为训练 PASS。正式任务内含真实来源探针、首次梯度检查及冻结 USAD2 状态不变校验。交付、可选 tensor 检查、resume 和产物路径见路线 README；现有 HTSAT 评测入口尚未接入此路线。

## CED-Small / 两次循环 MeSH checkpoint 的 MMAU / MMAR 对齐评测

现有 `run_mizar_s1_ced_small_mmau_eval_5090.sh`、`run_mizar_s1_ced_small_mmar_eval_5090.sh` 新增 `RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh`，默认加载 `mizar_s1_ced_small_gbs256_recursive_mesh_5090/formal_20261027_081123150934256` 的 `formal/checkpoints/model--step-17313.ckpt` 与配套 `formal/runtime.json`。未设置 variant 或使用 `base` 时仍走原普通 CED 路线。详细说明见 [MeSH CED 评测交付](code/RSmol/scripts/README_mizar_s1_ced_recursive_mesh_eval.md)。

使用独立 MeSH runner、adapter 和 Bash wrapper；原 CED Python runner、adapter、运行脚本，以及两条 CED 训练路线和共享 MeSH 实现均未修改。完整音频 checkpoint 严格覆盖 CED、mapper、LM 和 routers；启动核对训练/runtime/资产/源码身份、hidden576、20物理层/30逻辑执行、两次循环、5个临时memory slots、6个独立routers，每次真实forward只读核对固定调度与无KV cache。

原评测设置保持一致：各完整1000条、固定官方选项和prompt、fixed20、CED small + freq_merge_126、prefix383、FP32/eval/TF32关闭、argmax完整重算/无cache/最多300新tokens/首次EOS停止、原始输出及EOS保留、原官方评分与续跑身份。每项仍为 `pdgpu-5090 / 8 CPU / 32G / 1 GPU / 1 node`，沿用原容器与 `mellow_c8204d8`。

GitHub同步后在远程登录节点提交：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh bash run_mizar_s1_ced_small_mmau_eval_5090.sh
RSMOL_MIZAR_CED_EVAL_VARIANT=recursive_mesh bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

结果分别位于上述MeSH run下的 `eval/mmau_test_mini_step17313/`、`eval/mmar_step17313/`。本地新增25项检查和原CED评测12项回归通过，18个runner函数AST、greedy_decode及生成协议一致，Python/Bash语法通过；两项真实小模型CPU检查因本地缺少Torch/Transformers跳过，真实权重加载、GPU生成与官方评分待远程作业验证。

## 2026-10-27：CED small checkpoint 的 MMAU / MMAR 对齐评测

新增两条独立的 CED-Small Mizar S1 评测入口，不改动原有 HTSAT 评测脚本：
`run_mizar_s1_ced_small_mmau_eval_5090.sh` 和 `run_mizar_s1_ced_small_mmar_eval_5090.sh`。
它们默认加载正式 checkpoint
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_5090/formal_20261026_193830617147305/formal/checkpoints/model--step-17313.ckpt`
及该 run 的 `formal/runtime.json`，使用隔离的 `evaluate_mizar_s1_ced_benchmark.py` 和
`mizar_s1_ced_eval_runtime.py`。

评测保持原 MMAU/MMAR 数据遍历、固定 choice 顺序、fixed20 音频处理、原始生成文本保存和官方 scorer 不变；仅按训练合同切换到 CED-Small 在线 FP32 encoder 与 `freq_merge_126` mapper：
`126 audio + 1 SEP + 256 prompt = 383 prefix`，`attention_mask=None`，全量重算、无 KV cache、argmax、最多300个新 token，首个 answer logit 位于382。运行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_ced_small_mmau_eval_5090.sh
bash run_mizar_s1_ced_small_mmar_eval_5090.sh
```

默认结果分别写入该 run 下的 `eval/mmau_test_mini_step17313/` 与 `eval/mmar_step17313/`，包含 `raw_generations.jsonl`、`predictions_*.json`、`evaluation_report.json` 和 `official_evaluation.txt`。入口支持 `RSMOL_MIZAR_CED_RUN_ROOT`、`RSMOL_MIZAR_CED_CHECKPOINT`、`RSMOL_MIZAR_CED_RUNTIME_CONFIG`、`RSMOL_MIZAR_CED_ROUTE_ROOT` 与对应数据集/输出目录变量覆盖路径。

每项评测提交一个 `pdgpu-5090 / 8 CPU / 32G / 1 GPU / 1 node` 作业，使用与原入口相同的容器和 `mellow_c8204d8` 环境。新增12项CPU合同/控制流测试及现有35项评测、CED路线、离线配置回归通过，Python编译、Bash语法和diff检查通过；本地没有真实权重、数据或PyTorch，真实GPU加载、1000条生成和官方评分尚需远程作业验证。

## 2026-10-27：CED small + MeSH 5+10×2+5 的隔离训练路线

新增 [MeSH CED 路线](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small/README.md)，从原 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh` 完整复制隔离。保留 `checkpoint-009244` 的20物理层/30逻辑层、2 loops、5 memory slots、6 routers，以及原 tokenizer/checkpoint 校验；音频侧统一为与 `run_mizar_s1_ced_small_gbs256_5090.sh` 相同的冻结 CED small、官方 `freq_merge_126` mapper 和 `126+1+256+300=683` 单音频布局。原MeSH、标准CED及非MeSH 5-10-5路线均未修改。

两条CED路线的16个音频/数据/优化文件一致，prefix/forward、共同训练合同和完整optimizer循环通过AST/配置对照；原MeSH的11个数据/staging文件、answer-only CE和checkpoint游标保持一致。MeSH保留按answer mask与非PAD共同选目标的原CE，实际tokenizer padding条件下与标准CED的ignore-PAD CE选取相同目标。正式配置仍为hybrid 1,477,492行、8 GPU × microbatch8 × accumulation4、3 epochs / 17,313 updates / 866 warmup，FP32、LM LR3e-4、mapper LR6e-4；独立输出、日志、index cache、tmpfs与resume身份。

本地新增21项stdlib测试、现有CED/离线loader/MeSH的36项回归检查通过；官方mapper类AST、文本/RNG、LR和GradNorm审计通过。Python编译与Bash语法检查另行执行。本地没有torch/torchaudio/transformers、真实权重或数据；tensor检查和真实GPU训练仍需远程执行，未登记训练PASS。

经GitHub同步后，在远程登录节点提交（只提交formal，没有smoke前置任务）：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_gbs256_recursive_mesh_ced_small_5090.sh \
  --ced-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_mesh_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

资源为 `pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`。最终产物位于输出root的 `formal/checkpoints/model--epo-3.ckpt`，也保存 `model--step-17313.ckpt`。详细流程、对比边界、resume及可选tensor检查见路线README。现有HTSAT评测入口尚未接入此新CED路线。

## 2026-10-26：CED small 离线配置加载修复

远程报错停在 CED 初始化：上游 `configuration_ced.py` 的 `CedConfig.__init__` 在 outputdim=527 时，无条件调用 `cached_file("topel/ConvNeXt-Tiny-AT", "class_labels_indices.csv")`；即使权重和代码都已在本地，离线环境仍会因缺少这张外部 CSV 失败。此前仅检查本地六个资产文件没有覆盖这一隐藏依赖。

CED 隔离路线新增 `models/ced_offline.py`：复用本地 `config.json` 中完整527项 `id2label` 生成临时标签表，仅在原版配置及模型加载期间，将 CED 配置模块内的 CSV 解析指向该表，再显式将原版配置实例交给 AutoModel。完成或异常时恢复模块函数绑定并删除临时文件；不修改权重目录、HF缓存源码、全局Hub函数或训练参数。保留重复显示名称的真实索引和原始 label2id；提交前增加标签完整性检查，不需要另行下载 CSV。

23项本地检查通过，另用真实上游 CedConfig 类和官方发布的 config.json 执行 stdlib 配置回归：原错误可复现，修复后额外 Hub 标签请求为0，保留527项 id2label / 524项 label2id。该回归使用 PretrainedConfig stand-in，不加载真实权重或 tensor，远程 GPU 仍需重跑确认。经 GitHub 同步后使用下方正式提交命令的新时间戳输出目录重新开始；本次错误发生在模型初始化，无需 resume，旧目录和缓存无需删除。代码SHA变化会正常选择新 CPU index cache。详细说明见 [CED路线 README](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_ced_small/README.md)。

## 2026-10-26：Mizar S1 / CED small + 官方 mapper / GBS256 正式训练

新增独立路线 [mizar_stage1_fixed20_gbs256_3epochs_ced_small](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_ced_small/README.md)，完整复制隔离自 `mizar_stage1_fixed20_gbs256_3epochs`。音频侧替换为冻结且始终 eval 的 CED small，以及可训练的官方 `freq_merge_126` mapper；保留原始30层 SmolLM2 全量训练。CED 从 `/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small` 离线严格加载 `model.safetensors` 和目录内模型代码，不使用 ONNX。

固定20秒波形先整段重采样、计算 mel 和冻结 BN，再按1012帧切分；正确保留 segment/frequency/time 顺序，mapper 输出126个576维音频 token。新的布局为 `126 audio + 1 SEP + 256 prompt + 300 answer = 683`，prefix383；采用逐样本 dB 截断，避免在线 batch 不同响度之间相互影响。官方 mapper 结构及初始化不变，在线 FP32 hidden 与官方 FP16 hidden cache 的差异已写入路线文档。

hybrid 1,477,492行数据、fixed20读取策略、Mizar padding/answer-only CE、AdamW/LambdaLR/GradNorm、共享音频立即训练及后台复制切换、完整 checkpoint/resume 均继承母本。正式训练仍为8 GPU × microbatch8 × accumulation4 = GBS256，3 epochs / 17,313 updates / 866 warmup。原路线未修改；新路线使用独立输出、日志、index cache 和 tmpfs 目录。

经 GitHub 同步代码至远程后，从远程登录节点提交（没有 smoke 前置任务）：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_s1_ced_small_gbs256_5090.sh \
  --ced-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

资源保持 `pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`。本地新路线17项检查与母本18项回归检查通过，官方源码对齐审计通过；本地无 PyTorch/CUDA、远程权重或数据，真实 tensor、权重加载及 GPU 训练仍待远程执行，不能将 CPU 结果记为训练 PASS。恢复限制、产物路径和可选 tensor 检查见上述路线 README。

## 2026-10-26: Non-MeSH 5-10-5 two-epoch Core SFT baseline

New isolated route: [mizar_stage1_fixed20_gbs256_2epochs_recursive_5_10_5_core_sft](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_2epochs_recursive_5_10_5_core_sft/README.md). It initializes the complete audio model from `model--step-17313.ckpt` in the completed non-MeSH route, restarts all optimizer/scheduler/RNG/data state, freezes HTSAT, c2l, projection, embeddings, outer physical layers, final norm and lm_head, and trains only shared physical layers 5..14. The fixed `0..14, 5..14, 15..19` schedule executes the middle ten layers twice.

The formal contract is 8 GPUs x microbatch 8 x accumulation 4, global batch 256, 2 epochs, 11,542 optimizer updates, max/min LR 5e-5/5e-6, Mizar padding with `attention_mask=None`, and answer-only CE including answer EOS while masking answer PAD. Submit from `code/RSmol`:

```bash
bash run_mizar_s1_recursive_5_10_5_core_sft_gbs256_5090.sh --stage-workers 4 --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_recursive_5_10_5_core_sft_gbs256_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

# RSM_bridge：Recursive SmolLM / Audio MeSH 项目交接

## 2026-10-26：非 MeSH 5+10×2+5 checkpoint 的 MMAU / MMAR 官方评测

两个现有评测入口新增 `RSMOL_MIZAR_S1_EVAL_VARIANT=recursive_5_10_5`，默认选择训练路线 `mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5`，加载以下 run 的 `formal/checkpoints/model--step-17313.ckpt` 和 `formal/runtime.json`：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_recursive_5_10_5_5090/formal_20261025_160348996763174
```

评测 adapter 按 route 选择 `recursive_model_5_10_5.py`，保留 checkpoint/runtime、训练源码 SHA、原始组件资产、词表和完整 state_dict 严格校验。完整音频权重加载后，再执行该训练路线真实的 `validate_loaded_model`：验证 30 logical / 20 独立 physical / middle-only 2 loops，以及参数和 buffer 中没有 router、memory-slot 或 MeSH 状态。MeSH 分支继续使用自己的实现及 5 memory slots / 6 routers 审计。

经 GitHub 同步到远程后，在 `code/RSmol/` 直接提交两个完整评测作业：

```bash
RSMOL_MIZAR_S1_EVAL_VARIANT=recursive_5_10_5 bash run_mizar_s1_mmau_eval_5090.sh
RSMOL_MIZAR_S1_EVAL_VARIANT=recursive_5_10_5 bash run_mizar_s1_mmar_eval_5090.sh
```

每个作业使用 `pdgpu-5090 / -c 8 -m 32G -g 1 -n 1`。两项评测共用此前的 fixed20 在线音频、`129+129+1+256` prefix、固定 answer 起点 515、FP32、可见 padding / `attention_mask=None`、无 KV cache 的确定性 argmax、EOS 停止和最多 300 new tokens。完整生成保留 EOS，不截取、不 trim、不提取选项，只交给各自官方 scorer；不执行额外 rescore，也不增加 smoke 前置门槛。

结果分别写到上述 run 下的 `eval/mmau_test_mini_step17313/` 和 `eval/mmar_step17313/`，包含 `raw_generations.jsonl`、`evaluation_report.json` 和 `official_evaluation.txt`。本地 CPU 验证覆盖真实训练审计函数调用、错误模型状态拒绝、checkpoint 合同，以及两个 Bash 入口的四种 variant 与资源参数；远程真实权重加载和 GPU 生成仍由评测作业验证。

## 2026-10-26：Mizar S1 / 5-10x4-5 MeSH 因 OOM 调整为 GBS128

现有隔离路线 [mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_5_10x4_5](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_5_10x4_5/README.md) 在远程 GBS256 运行中发生 OOM，按用户要求改为 **8 GPU × microbatch 4 × accumulation 4 = GBS128**。减半 microbatch 以降低单次前向/反向激活显存；仅减半 accumulation 不能降低原 microbatch 的激活峰值。总显存还包含参数与优化器状态，不能保证总峰值减半。

文本初始化仍为 `stage4_5_10x4_5_mesh/formal_2epochs_18488steps_20261007_004959_3090_v1/checkpoint-018488`，4 loops / 7 memory slots；HTSAT、fixed20、数据、815-token 序列、FP32、answer-only loss、AdamW 各组 LR 和 GradNorm 均保留。3 epochs 重新计算为 **11,542 updates/epoch、34,626 total、1,731 warmup**，每 epoch 仍丢弃 116 行。保存步骤和训练结束校验同步使用新总步数。

目录和提交脚本保留历史 `gbs256` 文件名；runtime/checkpoint route 已改为 `htsat_mizar_s1_fixed20_gbs128_recursive_mesh_5_10x4_5_v1`，输出/日志改为 GBS128。提交脚本同步远程实际资源 `pdgpu-3090 / 64 CPU / 256G / 8 GPU`，并缩短 job 名以满足 60 位上限。经 GitHub 同步后，在 `code/RSmol/` 执行：

```bash
bash run_mizar_s1_gbs256_recursive_mesh_5_10x4_5_5090.sh --stage-workers 4
```

默认写入新的 `outputs/RSmol/mizar_s1_htsat_gbs128_recursive_mesh_5_10x4_5_3090/formal_<时间戳>/`，从指定文本 checkpoint 重新初始化，不使用 OOM 任务的音频 checkpoint resume。真实 GPU 显存和训练仍需远程运行确认。

## 2026-10-27：GBS128 5-10x4-5 MeSH checkpoint 的 MMAU / MMAR 官方评测

已将两个既有 Mizar S1 官方评测入口扩展为独立的
`RSMOL_MIZAR_S1_EVAL_VARIANT=recursive_mesh_gbs128` 变体。原有的
`gbs32`、`gbs256`、`recursive_mesh` 和 `recursive_5_10_5` 变体保持不变。

新变体固定选择：

```text
checkpoint:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs128_recursive_mesh_5_10x4_5_3090/formal_20261026_003638413364331/formal/checkpoints/model--step-34626.ckpt

runtime:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs128_recursive_mesh_5_10x4_5_3090/formal_20261026_003638413364331/formal/runtime.json

route:
mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_5_10x4_5

step / geometry:
34626 / 8,4,4
```

推理和评分协议保持原评测实现：fixed20 音频、`129+129+1+256` prefix、
`attention_mask=None`、FP32、关闭 TF32、纯 argmax、`use_cache=False`、
最多 300 个新 token、保留 EOS，以及官方 MMAU/MMAR scorer。x4 loader 额外
校验 50 logical / 20 physical / 4 loops / 7 memory slots / 10 routers、
router 独立对象、瞬时 MeSH memory、strict state_dict、tokenizer 和训练
contract。GBS128 的训练 geometry 只用于 checkpoint 身份校验，不改变推理
batch 或评分设置。

远程同步后，在 `code/RSmol/` 分别提交两个独立的单卡评测作业：

```bash
RSMOL_MIZAR_S1_EVAL_VARIANT=recursive_mesh_gbs128 \
bash run_mizar_s1_mmau_eval_5090.sh

RSMOL_MIZAR_S1_EVAL_VARIANT=recursive_mesh_gbs128 \
bash run_mizar_s1_mmar_eval_5090.sh
```

结果分别写入：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs128_recursive_mesh_5_10x4_5_3090/formal_20261026_003638413364331/eval/mmau_test_mini_step34626/
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs128_recursive_mesh_5_10x4_5_3090/formal_20261026_003638413364331/eval/mmar_step34626/
```

两项评测继续提交到 `pdgpu-5090` 的单 GPU、8 CPU、32G 作业；输出目录
与旧 step-17313 结果隔离。评测链路本地 CPU 合同测试共 23 项通过，真实
checkpoint 加载、GPU 推理和官方评分仍需远程执行确认。

## 2026-10-25：Mizar S1 / GBS256 非 MeSH 5+10×2+5 文本初始化消融

新增隔离路线 [mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5/README.md)。它从已修复的 Mizar MeSH GBS256 路线复制，只把文本模型初始化替换为远程 checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244
```

该 checkpoint 使用 `code/RSmol/recursive_model_5_10_5.py` 的非 MeSH 实现：20 个独立物理层，逻辑执行为 `0..14, 5..14, 15..19`，即固定 `5+10×2+5`。loader 显式导入该类，不依赖全局 AutoModel 注册，并拒绝 router、memory-slot 或 MeSH 参数/缓冲区。文本权重只作为初始化载入；optimizer、scheduler、RNG、数据游标和音频侧模块均从新的 Mizar S1 GBS256 训练开始。

除文本模型加载、模型审计和路线身份外，数据 manifest、在线 fixed20、HTSAT、`129+129+1+256+300` 序列、Mizar padding/`attention_mask=None`、answer-only rank-local CE、AdamW/LambdaLR/GradNorm、后台 staging、checkpoint 与 resume 合同都保持修复后的 MeSH 母本一致。CPU 对齐审计已通过，母本训练/数据/优化文件逐字节一致；本地没有远程权重、数据或 CUDA，因此真实 GPU 加载和训练仍需远程执行。

远程同步后，先可运行 tokenizer/config CPU 检查，再从 `code/RSmol/` 提交正式训练：

```bash
bash run_mizar_s1_gbs256_recursive_5_10_5_5090.sh \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_recursive_5_10_5_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

## 2026-10-25：Mizar S1 / GBS256 MeSH tokenizer 审计修正

隔离路线 [mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh/README.md) 仅替换 GBS256 母本的文本模型初始化，加载既有 MeSH 文本 `checkpoint-009244`。用户远程 CPU 检查确认：canonical SmolLM2 和该 checkpoint 的完整词表及 token ID 一致，词表 SHA256 均为 `d59adeae1bb1c383e51bcf30fd8ad05a2b82f7e752f72106f0f94c324cdb44b6`；执行 Mizar 的 `add_special_tokens({"pad_token": "!"})` 返回 1，但词表大小不变，并与模型配置一致。此前将返回值误当作新增词表行数的审计已修正；现在检查实际词表大小和 `!` ID 前后不变、checkpoint `config.vocab_size` 及加载后的 embedding 行数。源 tokenizer 的 EOS-as-PAD 元数据由原 Mizar 调用覆盖。

本次只修审计及启动日志，不扩 embedding。固定 padding、`attention_mask=None`、answer-only loss、数据、优化器和正式训练步数保持原路线实现。正式提交入口为 `code/RSmol/run_mizar_s1_gbs256_recursive_mesh_5090.sh`，8 GPU × microbatch 8 × accumulation 4，3 epochs / 17,313 updates；使用新输出目录重新提交。本地 MeSH 审计 13 项、GBS256 母本 7 项、共享读取/后台 staging 11 项，共 31 项 CPU 测试通过。远程 tokenizer CPU 检查已确认兼容，真实 GPU 加载及训练仍待执行，不能把 CPU 结果记为训练 PASS。

## 2026-10-24：扩展 Mizar S1 音频范围后的完整 CPU 审计

当前数据准备重新从原始 `/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1.json` 出发，**MusicCaps 和 SpeechCraft train-other-500 均纳入使用范围**。hybrid manifest 现已接入 GBS256 训练入口；旧的 `s1_stage1_filtered.json` 和旧审计 TSV 不再用于该路线，下面历史段落保留旧实验记录。

新增 `code/RSmol/scripts/mizar_s1_audio_mapping_s2_hybrid.json`，保留既有各来源映射。AudioMCQ AudioCaps 继续用 `audiocaps_v2` CSV 校验 YouTube ID/start-time，优先复用原 `audiocaps_v2` 文件，只在缺失时查补充 `data/AudioCaps`。MusicCaps 继续通过 `data/AudioMCQ/audio_mapping.s2.jsonl` 的精确原始 key 和实际文件名查 `data/MusicCaps`。SpeechCraft 三个 split 改为精确相对路径映射：

- `train-clean-100` → `/hpc_stor03/public/shared/data/tts/LibriTTS/train-clean-100`。
- `train-clean-360` → `/hpc_stor03/public/shared/data/tts/LibriTTS/train-clean-360`。
- `train-other-500` → `/hpc_stor03/sjtu_home/jinwei.zhang/data/SpeechCraft/libritts_r_audios/train-other-500`，使用从学长 S2 传输、已解压的 LibriTTS-R。

clean100/360 使用原始 LibriTTS、other500 使用 LibriTTS-R，是用户选择的混合音频版本；路径存在性审计不能证明两种版本的波形或音质相同。新审计参数 `--require-all-sources` 将所有来源的缺失、歧义和无法映射都正常报错，不再豁免 MusicCaps/other500；不加该参数仍兼容旧策略。新增 `speechcraft_by_split` 统计，保留 `resolved_paths.tsv`、按真实路径去重的 `unique_files.tsv`、文件总字节数及容量检查；对重复引用复用文件大小检查，减少共享文件系统请求。脚本拒绝写入非空审计目录。

GitHub 同步后，在远程登录节点执行（不需要 GPU 或额外 Python 包）：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
audit_dir="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_$(date +%Y%m%d_%H%M%S)"
audit_log="${audit_dir}.log"
if python -u scripts/audit_mizar_s1_raw_audio.py \
  --manifest /hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1.json \
  --mapping-json scripts/mizar_s1_audio_mapping_s2_hybrid.json \
  --output-dir "$audit_dir" \
  --require-all-sources \
  --canonicalize-resolved-paths \
  --max-unique-audio-gib 256 \
  --progress-every 100000 >"$audit_log" 2>&1; then
  audit_status=0
else
  audit_status=$?
fi
echo "audit_exit=$audit_status"
echo "audit_dir=$audit_dir"
echo "audit_log=$audit_log"
if [ -f "$audit_dir/summary.json" ]; then
  python -c 'import json,sys; d=json.load(open(sys.argv[1])); k=("counts","source_status","speechcraft_by_split","method_rows","unique_resolved_files","unique_resolved_audio_gib","capacity_check"); print(json.dumps({x:d.get(x) for x in k},ensure_ascii=False,indent=2))' "$audit_dir/summary.json"
else
  tail -n 40 "$audit_log"
fi
```

本命令只核对路径和文件大小，不解码、不复制音频、不生成过滤 manifest。退出码 2 表示还有待处理数据问题，3 表示唯一音频总大小超过 256 GiB；两者仍会写出审计报告。唯一大小按真实路径去重，不做全库内容哈希，不能识别任意不同路径下的相同波形；AudioCaps 两来源复用原文件由优先映射保证。后续过滤如使用现有 `filter_mizar_s1_manifest.py`，必须显式保留 MusicCaps 与 other500（`--keep-musiccaps --keep-train-other-500`），并使用新的输出文件名；本次尚未执行过滤。新增混合映射、已有 AudioCaps/S2 映射共 20 项本地 CPU 测试通过；远程完整数据审计待用户执行。

## 2026-10-25：Mizar S1 / hybrid GBS256 checkpoint 的 MMAU / MMAR 官方评测

`code/RSmol/run_mizar_s1_mmau_eval_5090.sh` 和 `code/RSmol/run_mizar_s1_mmar_eval_5090.sh` 都支持 `RSMOL_MIZAR_S1_EVAL_VARIANT=gbs256`，默认仍为 GBS32。GBS256 自动选择 `mizar_stage1_fixed20_gbs256_3epochs`、`formal_hybrid_gbs256_v1/formal/runtime.json` 和 `model--step-17313.ckpt`，校验训练几何 `8×8×4` 及累积训练的 loss 标识。checkpoint/runtime 合同、源码 SHA、组件资产和模型 strict loading 校验继续执行；不要求训练 staging 目录仍存在。

经 GitHub 同步到远程后，在 `code/RSmol/` 执行：

```bash
RSMOL_MIZAR_S1_EVAL_VARIANT=gbs256 bash run_mizar_s1_mmau_eval_5090.sh
```

同一路线的 MMAR 评测提交：

```bash
RSMOL_MIZAR_S1_EVAL_VARIANT=gbs256 bash run_mizar_s1_mmar_eval_5090.sh
```

两项评测均为单卡、完整 1,000 行、确定性 greedy、EOS 停止且最多 300 new tokens；音频读取/重采样/fixed20 和 `129+129+1+256` prefix 与选中训练路线一致。完整生成原样交给各自官方 scorer，不做标签截取或文本预处理。默认结果分别位于：

- MMAU：`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_5090/formal_hybrid_gbs256_v1/eval/mmau_test_mini_step17313/`。
- MMAR：`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_5090/formal_hybrid_gbs256_v1/eval/mmar_step17313/`。

本次两项结果均以各自目录中的 `evaluation_report.json` 和 `official_evaluation.txt` 为准；`raw_generations.jsonl` 只作为生成审计，不执行额外标签重评分。实现经本地 CPU 合同和两种 variant 的 Bash 提交参数模拟验证，真实 checkpoint 加载及 GPU 推理仍需远程作业验证。

## 2026-10-25：Mizar S1 / recursive MeSH GBS256 checkpoint 的 MMAU / MMAR 官方评测

现已将 recursive MeSH route 接入同一官方评测器。使用 `RSMOL_MIZAR_S1_EVAL_VARIANT=recursive_mesh` 时，两个入口自动选择 route `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh`、step `17313` 的 checkpoint/runtime，以及独立输出目录。加载时额外校验 recursive route contract、`recursive_model_5_10x2_5_mesh.py` 源码 SHA、文本 checkpoint config/tokenizer 文件、30 logical / 20 physical / 2 loop / 5 memory slot / 6 router 参数审计和 strict state_dict。

远程同步后提交：

```bash
RSMOL_MIZAR_S1_EVAL_VARIANT=recursive_mesh bash run_mizar_s1_mmau_eval_5090.sh
RSMOL_MIZAR_S1_EVAL_VARIANT=recursive_mesh bash run_mizar_s1_mmar_eval_5090.sh
```

recursive MeSH 与普通 GBS256 评测共用 fixed20 音频、`129+129+1+256+300` 序列、FP32、`attention_mask=None`、纯 argmax greedy、最多 300 new tokens 和各自官方 scorer；差异只在文本 decoder route。默认输出为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_recursive_mesh_5090/formal_recursive_mesh_padfix_20261025_003124507033014/eval/mmau_test_mini_step17313/
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_recursive_mesh_5090/formal_recursive_mesh_padfix_20261025_003124507033014/eval/mmar_step17313/
```

## 2026-10-23：独立的 Mizar Stage 1 / GBS256 消融线

新增隔离副本 [mizar_stage1_fixed20_gbs256_3epochs](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs/README.md) 和提交入口 [run_mizar_s1_gbs256_5090.sh](code/RSmol/run_mizar_s1_gbs256_5090.sh)。它保留 GBS32 路线的 HTSAT + c2l + projection + SmolLM2、固定20在线音频、已过滤的 1,311,334 行数据及 Mizar Stage 1 优化设置；改为 8 GPU × microbatch 8 × accumulation 4，GBS256。原 GBS32 路线和 checkpoint 合同不变。

每 epoch **5,122** 次 optimizer update，3 epochs 共 **15,366** 次，5% warmup 为 **768** 次；每 epoch 消耗 1,311,232 行、丢弃 102 行。用户反馈三次未能启动后，GBS256 已改为：提交前在登录节点用标准库准备/复用 metadata 与去重文件索引；GPU 作业直接读原始共享音频开始 formal，同时默认 4 线程后台复制；全部复制完成后，在 optimizer update 边界八卡 MIN all-reduce 确认并切换到本地音频。取消预拷贝和跨 epoch staging 水位等待；不做 smoke。原数据、在线 fixed20、优化器/累积/LR/GradNorm 配方不变。

远程 GitHub 同步后，在 code/RSmol/ 提交：

~~~bash
bash run_mizar_s1_gbs256_5090.sh \
  --stage-workers 4 \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_gbs256_5090/formal_sharedstart_v2
~~~

首次 CPU 索引扫描可能耗时数分钟，此时还未申请 GPU；默认共享缓存为 `outputs/RSmol/mizar_s1_htsat_gbs256_5090/index_cache/<signature>/`，同代码/manifest/TSV 的重试直接复用。远程曾因旧版路径/大小/mtime 整体比较而报输入身份不一致；现改用 schema 2 的大小与内容 SHA256，时间戳和内容相同的路径别名不使校验失败，真实内容变化则列出具体输入、预期/当前路径、大小及 SHA256。新版本在申请 GPU 前自动建立新的 CPU 索引，不需删除旧缓存；GPU 启动仅校验两个 metadata 输入，不重扫音频。模型与权重加载仍需时间；共享读取期间训练与后台复制竞争带宽，不能据此保证平台不终止任务。日志 `AUDIO_READ_SWITCH` 与 metrics 的 `audio_read_mode`/`local_switch_step` 记录切换。本机没有 PyTorch/CUDA 与真实数据，验证仅限 CPU/stdlib、并发与脚本模拟。改前 GBS256 checkpoint 因代码 SHA 改变不能直接 resume；GBS32 checkpoint 也不能使用。

## 2026-10-21：HTSAT 架构采用 Mizar Stage 1，8×4×1，3 epochs

当前新增隔离路线 [mizar_stage1_fixed20_3epochs](code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_3epochs/README.md)，从用户指定的 `reasonaqa_global_batch256_lr_ablation_5epochs` 保留 HTSAT + c2l + residual projection + 完整 SmolLM2 架构，移植官方 Mizar S1 的文本、分词、answer-only rank-local CE、AdamW 分组、5% warmup/cosine 和 GradNormTracker。既有 ReasonAQA 路线不改动。

固定使用远程 `/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_filtered.json` 的 **1,311,334** 行，通过已有审计 TSV 的原始字段/路径映射；保留计算节点 staging 原始音频后在线解码。单文件 fixed20 中心裁剪/右补零后拆成两个 10 秒；HTSAT 各输出 129 tokens，两段之间去掉 SEP，最终 `audio[258] → SEP[1] → prompt[256] → answer[300]`，prefix 515、全序列 815。MusicCaps、SpeechCraft train-other-500 已过滤；SpeechCraft 使用原始 LibriTTS clean100/360。

用户最新确定为 **8 GPUs × microbatch 4 × accumulation 1 = GBS32**；3 epochs 共 **122,937** 次更新，warmup **6,147** 次。按用户要求直接 formal，不要求 smoke PASS 报告，也不自动先跑 smoke；默认模式已改为 formal。远程同步代码后，在 `code/RSmol/` 执行：

```bash
bash run_mizar_s1_5090.sh --mode formal \
  --stage-start-steps 512 --stage-workers 16 \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_5090/formal_async_v1
```

启动仍核对本次配置、数据映射、权重身份和 staging 就绪状态；不读取或比较历史 smoke 报告。checkpoint 恢复、输出文件和详细配方见新路线 README。官方 CED 的 126 个音频 tokens、缓存特征与这里的 HTSAT/在线解码有明确差异；8×4 和官方 4×8 保持 GBS32，但不保证相同参数轨迹。

远程 8 项小型张量检查已由用户确认通过（PyTorch 2.11.0+cu128 / CUDA 12.8，8 卡可见），按用户要求注释掉自动执行入口，保留手动诊断脚本；8 次真实 optimizer update 的端到端 smoke 仅在显式指定 `--mode smoke` 时运行。staging 只映射、检查容量/文件大小、复制原始音频字节，不解码或预处理。每来源 1 行前向探针和训练批次都在 trainer 启动后在线解码；formal 最终复制全部保留行引用的文件。

目前改为按实际训练顺序后台 staging：第 1 epoch 的私有 shuffle、八个连续 rank 分片决定文件首次使用的 batch，先准备前 **512 个 global steps** 与来源探针的音频，再启动真实训练，同时后台继续复制全量文件。用 `--stage-start-steps N` / `--stage-workers N` 调整启动门槛和线程数，日志给出 128/256/512/1024/2048 等档位的实际去重文件数/字节数。文件先写 `.partial`，检查大小后原子发布；训练在取下一批前检查八卡共同就绪状态，追上后台时等待，复制失败/进程退出/心跳或等待超时则报错。全量完成后后续 epoch 直接复用；resume 则按 checkpoint 的 epoch/batch 游标安排复制优先级。启动前仍需建立完整 metadata 索引，全量音频仍约 **333,510 文件 / 173.48 GiB**，没有额外完整归档副本。

远程 `smoke_v2` 曾被错误的 PAD=0 预检阻断，修正 PAD=17 / EOS=SEP=0 后用户确认 `smoke_v3` 的 8 次真实更新已 PASS。formal 随后在全量复制至约 215,040/333,510 文件时无 traceback 中止，GPU 空闲回收是用户的经验判断，尚无平台退出原因证明。本次采用后台 staging 后直接开始正式训练，无需重跑 smoke；原有 8 项小型张量测试仍不会自动执行。

后台 staging 实现时本地标准库单元验证共 **33 项通过**；本次移除 smoke 门槛后重新通过路线与后台 staging 的 **22 项测试**，验证不提供 smoke 报告即可生成完整 formal 配置，以及 formal 启动/失败清理流程。其他覆盖包括跨预拷贝边界等待、后续 epoch 等待全量、复制故障、死进程、心跳/等待超时、原子发布、八卡分片与 resume 优先级；此前已按官方源码比较文本/随机状态 1,344 次、tokenizer 参数 8 次、完整 LR 曲线 122,938 个位置与 GradNorm 标量策略 10,000 次。后台 staging 版本的真实 8 卡训练尚待远程执行：本地没有 PyTorch、权重、数据或 CUDA，不能将上述检查当作 GPU 训练通过。现有统一 evaluator 尚未接入该路线。

## 2026-10-20：LR10 路线复制出的 30-epoch、max_lr=4e-5 训练线

新增隔离路线 `code/RSmol/mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch256_lr_ablation_30epochs/`，完整复制自 `reasonaqa_global_batch256_lr_ablation_10epochs/`。保留其数据/固定 padding/因果注意力、随机第二音频槽、HTSAT eval/no_grad、global clip 0.5、GBS256、AdamW、FP32 与其他实现；训练 horizon 改为 30 epochs，提交 max_lr=4e-5，min_lr 自动为 4e-6。新 route、runtime、checkpoint、job 和输出目录均与 LR10 隔离。

远端 GitHub 同步后，在 `code/RSmol/` 正式提交（不需要 smoke）：

```bash
bash run_mellow_official_reasonaqa_gbs256_lr_ablation_30epochs_5090.sh --max-lr 4e-5
```

数据仍为 968,059 条时预期约 3,781 updates/epoch、113,430 total updates 和 5,672 warmup updates；训练器会按实际数据重新计算。新路线 route_variant 为 `lr_ablation_30epochs_htsat_frozen_global_clip`，runtime 为 `runtime_30epochs.yaml`，最终 checkpoint 为 `model--epo-30.ckpt`。只允许从本路线、相同 LR 和 30-epoch horizon 的 checkpoint resume。当前专用训练评测链路尚未登记这个 30-epoch route；训练线交付不包含评测适配。实际训练仍需检查远端 job log 和 checkpoint 审计结果。

## 2026-10-18：10-epoch LR 消融重新以完整 LR5 代码为母本

按用户最新确认，重新完整复制 `reasonaqa_global_batch256_lr_ablation_5epochs` 到 `reasonaqa_global_batch256_lr_ablation_10epochs`。本节覆盖下方 2026-10-17 关于此入口仍基于 formal30 compact 文本的描述。训练差异仅三项：10 epochs；采用 `htsat_frozen_5epochs` 的 HTSAT eval/no_grad 实现（c2l、projection 可训练）；采用 `global_clip_5epochs` 的全局 L2 硬裁剪 0.5 及配套保存/恢复处理。目录、job、输出路径和路线标识作相应隔离。

数据集、collate、decoder 保留 LR5 原文件：260 audio + 129 prompt + 250 answer 固定布局，分词前追加 EOS，保留原截断和 padding；传给 LM 的 `attention_mask=None`，保留因果掩码，prompt PAD 与 answer PAD 不额外做 attention 屏蔽，answer PAD 不计 loss。单音频空槽仍随机填充。GBS256、FP32、seed 1234、AdamW、global-token loss、5% warmup/cosine、显式 max LR 和 `min_lr=0.1*max_lr` 均保持母本。

经 GitHub 同步后，在远端 `code/RSmol/` 提交正式训练：

```bash
bash run_mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090.sh --max-lr 1e-3
```

新的路线标识为 `lr_ablation_10epochs_htsat_frozen_global_clip`，默认 run 目录前缀为 `formal_10epochs_lr5_frozen_clip_maxlr_`。每次独立 staging 并在退出时清理，不比较旧节点或已删除的 staging 路径；resume 只接受本新路线同 LR、同 10-epoch schedule 的完整 checkpoint。上一轮额外评测改动已撤回，当前统一 evaluator 仍是旧 compact 路线合同，尚未适配此次重建的 checkpoint。具体实现与模板见新路线 README；本地验证与远程实际训练分开记录。

远端首次启动在 staging 路径检查时退出，尚未复制音频或开始训练：启动脚本使用了 `...gbs256_lr10_...`，而完整复制的 staging 工具要求 `...gbs256_lr_` 前缀。已将启动路径修正为 `/dev/shm/mellow_adamw_cosine_reasonaqa_gbs256_lr_10epochs_frozen_clip_<run-id>`，工具本身及训练配置未改。同步后用原 max LR 和新输出目录重新提交，无需 resume。

本地现为 8 项复制/合同检查通过，新增用实际 staging 路径校验函数验证 runner 目录名，覆盖不同 run ID 并复现旧前缀拒绝；Python 与两份 Bash 脚本语法检查、提交参数模拟及 checkpoint 审计样例检查通过。数据集和 decoder 为保留原文件字节一致性，未清理其母本自带行尾空格。未执行 smoke 或远程训练，本机没有 PyTorch、权重和数据。

## 2026-10-18：两套音频评分均使用完整模型输出

更新统一评测入口 `code/RSmol/submit_mellow_formal30_eval_5090.sh` 对应实现：去掉官方 scorer 输入前删除 `a)` 至 `j)` 的预处理，官方 scorer 与选项前缀评分现在均直接接收同一份完整 `generated_text`，保留标签、括号、大小写和前后空白。前缀评分仍执行 `prediction.split(')')[0].lower()`，不额外 trim；官方脚本自身逻辑保持原样。checkpoint 审计、生成与第二音频槽复用合同保持不变。协议与默认输出目录升为 `eval_mellow_unified_v5`，避免与旧去标签评分结果混写；原提交命令继续适用，显式 `--output-dir` 应选择新目录。

Epoch-16 静音第二槽提交 wrapper 从 CRLF 的 `checkpoints.tsv` 读取别名路径时，Bash 会把行末 `\r` 保留进路径，导致存在的 checkpoint 被报告为不存在。wrapper 现清理该行尾字符，模拟测试改为 CRLF 别名表并验证三个别名均能提交 MMAU/MMAR；旧 v5 wrapper 和别名表不受影响。同步此修复后重提三模型命令。

Epoch-16 静音评测启动阶段移除全量音频预扫描：不再逐个 `stat`/哈希 1,000 个 WAV，也不再完整读取 MMAU parquet 求 SHA-256；metadata 与 scorer 的哈希/语义校验仍保留。音频在样本实际处理时按需解码并记录 waveform hash，MMAR 使用隔离入口内的轻量 artifact audit，旧共享 evaluator 未修改。checkpoint 完整加载后不再额外顺序读取整个文件计算 SHA-256，身份记录改用文件大小和修改时间。新协议与默认输出目录升为 v2，避免与包含全量音频哈希的旧身份文件冲突。重同步后使用原三条 `both` 提交命令即可。

## 2026-10-17：统一 Mellow 评测支持 10-epoch LR 消融 checkpoint

更新 `code/RSmol/mellow_formal30_eval/` 与 `submit_mellow_formal30_eval_5090.sh`：除既有 formal30 checkpoint 外，可显式传入 `run_mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090.sh` 产出的任意 1–10 epoch SmolLM2 checkpoint。评测器按 `route_variant` 选择隔离模型实现，验证 GBS256、训练策略、checkpoint 与 `runtime_10epochs.yaml` 的任意 max LR 和 `min_lr=0.1*max_lr` 一致；staging 和节点信息不作为评测门槛。统一评测输入现将第一音频 waveform 同时传入两个音频槽，各槽仍独立执行模型编码；协议和输出目录升为 v4，旧静音槽评测结果不会被续写。

远端同步代码后，在 `code/RSmol/` 提交 MMAU test-mini：

```bash
bash submit_mellow_formal30_eval_5090.sh mmau_test_mini \
  /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090/<run-root>/checkpoints/<job-dir>/model--epo-10.ckpt \
  --runtime-config /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090/<run-root>/runtime_10epochs.yaml
```

将 checkpoint 文件名替换为任意已保存 epoch。评测 README 见 `code/RSmol/mellow_formal30_eval/README.md`；本地合同回归测试通过后，远端权重加载和实际评分仍以正式作业为准。

## 2026-10-17：formal30 基础上的 10-epoch 学习率消融

新增独立路线 `code/RSmol/mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch256_lr_ablation_10epochs/` 和入口 `code/RSmol/run_mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090.sh`。母本是 `run_mellow_official_reasonaqa_gbs256_formal30_5090.sh` 对应的 formal30 实现；保留其 GBS256、FP32、AdamW、5% warmup/cosine、全局裁剪 0.5、compact 文本/EOS/padding/mask/loss、HTSAT 冻结和资源配置，仅改为 10 epochs、显式 `--max-lr`（`min_lr=0.1*max_lr`），以及单音频样本按历史 LR 消融路线随机填充第二音频槽。原 formal30 与 5-epoch 路线未修改。

代码经 GitHub 同步到远端 `code/RSmol/` 后提交：

```bash
bash run_mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090.sh --max-lr 1e-3
```

不同学习率使用独立输出根目录和 job；本地已加入静态合同测试，远程权重、staging 和正式训练结果仍需以作业日志与 `model--epo-10.ckpt` 审计为准。

## 2026-10-17：三条 formal30 训练线的统一 MMAU / MMAR 评测

新增隔离目录 `code/RSmol/mellow_formal30_eval/` 与入口 `code/RSmol/submit_mellow_formal30_eval_5090.sh`，仅用于本次 SmolLM2、5-10×2-5 MeSH、5-10×4-5 MeSH 的完整 30-epoch 音频 checkpoint。用户确认 SmolLM2 已完成并提供其最终路径；x2/x4 路径是用户指定的预期最终产物，提交前仍需实际存在。三种 checkpoint 别名和路径记录在新目录 `checkpoints.tsv`。

统一合同：FP32、关闭 autocast/TF32、单 GPU / batch 1、第二音频槽为零 waveform 且独立编码、260-token 双槽音频 prefix、问题和选项正文保持原始大小写、按原顺序添加小写 `a)` 至 `j)`（按实际选项数截取）。直接调用 formal30 的 prompt 分词方法，最多 129 tokens、强制保留最终 EOS、不补齐 prompt，EOS 后立即生成答案。MMAU 和 MMAR **均最多生成 300 tokens**，固定 top-p 0.8 过滤后 argmax、temperature 1、use_cache=False。MMAU 优先使用 `<id>.wav`，缺失时按既有 evaluator 规则回退 parquet 内嵌音频，并按样本 ID 固定种子裁剪；MMAR 沿用首 10 秒/短音频补零。

每个任务完整 1000 条，并同时输出首个 `)` 前标签分数和官方 scorer 分数；前者使用原始生成，不额外 trim，后者只去掉开头 `a)` 至 `j)` 标签再运行官方脚本。记录原始生成、prompt token IDs、输入音频片段 hash、完整分母、checkpoint 和评测身份。训练 staging 路径、节点/PID 不参加启动检查。三种架构仅切换加载 route，训练实现及旧 evaluator 未改动。

在远端 `code/RSmol/` 同步代码后直接提交正式评测，不要求 smoke；每条命令各提交 MMAU、MMAR 两个单卡作业，资源固定 `pdgpu-5090 -c 8 -m 32G -g 1 -n 1`：

```bash
bash submit_mellow_formal30_eval_5090.sh both smollm2
bash submit_mellow_formal30_eval_5090.sh both mesh_x2
bash submit_mellow_formal30_eval_5090.sh both mesh_x4
```

默认输出为各训练 run root 下 `eval_formal30_unified_v1/<task>/<checkpoint-job>/model--epo-30/full/`。详情和结果核对命令见 `code/RSmol/mellow_formal30_eval/README.md`。`compare.py` 会拒绝协议、实际 prompt token/音频片段、环境、非架构 LM 配置不一致的三模型结果。本地仅验证 dependency-light 测试、语法和 mock 提交，远端权重、FP32 CUDA 推理、完整 scorer 执行仍待实际作业验证；不能把实现已存在当作评测成功。下方历史 evaluator 和“formal30 尚未实现评测”描述为当时状态。

## 2026-10-15：formal30 的 5-10×2-5 MeSH 文本骨干对比线

新增隔离路线 `code/RSmol/mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch256_formal_30epochs_5_10x2_5_mesh/`，母本是当前 `run_mellow_official_reasonaqa_gbs256_formal30_5090.sh` 对应的 formal30。仅把文本骨干换成 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244` 的文本权重；音频模型、canonical SmolLM2 tokenizer、260 位前缀、文本拼接/mask/answer-EOS loss、GBS256、30 epochs、FP32、AdamW、`4e-5 → 4e-6`、5% warmup/cosine 和全局硬裁剪 0.5 保持基准合同。旧路线和共享递归实现未改动。

按用户要求直接准备正式训练，不新增或运行 smoke。经 GitHub 同步后，从远端 `code/RSmol/` 提交：

```bash
bash run_mellow_official_reasonaqa_gbs256_formal30_5_10x2_5_mesh_5090.sh
```

同一 `pdgpu-5090` 资源/容器与 `mellow_c8204d8` 环境，独立输出根目录 `outputs/RSmol/mellow_official_reasonaqa_gbs256_formal30_5_10x2_5_mesh_5090/`。启动校验源码/配置等价、tokenizer 兼容、完整 FP32 文本加载和音频初值/CPU RNG 精确对齐，记录权重/代码/数据/环境身份并拒绝跨路线 resume。cache/attention 使用基准实际设置，不直接沿用旧 adapter。详细提交、输出与恢复限制见新路线 README。

首次远端提交 staging 已 PASS，随后因原 SmolLM2 wrapper 的固化字节哈希不符而在生成配置前退出，尚未训练。已把该检查收窄：核心训练 Python、配置生成和 staging 实现用 AST 校验；wrapper、注释/格式及未使用模块的字节差异仅记录到 `provenance_only_changes`。源码检查提前到 staging 前。身份比较保留实际训练代码、权重、tokenizer 和软件/精度设置，排除节点名、PID、时间戳、输出目录和 staging 路径；换节点重新 staging、旧 staging 被退出清理后均可重新提交/恢复，最终 checkpoint 审计也不依赖旧 staging。训练循环与超参未改；本地 20 项测试通过，修复后的远端执行仍待确认。

此处记录本地实现，不代表远端加载或训练完成。递归实现目标 Transformers 4.54.1，旧环境快照为 4.47.1；实际环境兼容性和训练须以远端正式日志为准，不自动升级或改超参。该 checkpoint 有额外文本训练历史，因此属于文本骨干替换对比，不能把结果单独归因于架构。新路线 evaluator 尚未实现。

## 2026-10-15：x4 checkpoint-018488 的一致单卡 Stage 3 文本评测

新增独立入口 `code/RSmol/run_stage3_eval_5_10x4_5_mesh_5090.sh`，默认评测 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/formal_2epochs_18488steps_20261007_004959_3090_v1/checkpoint-018488`。它调用 `scripts/evaluate_stage3_5_10x4_5_mesh.sh` 和同名 Python evaluator，参考原 x2 的 `run_stage3_eval_5_10x2_5_mesh_5090.sh`，原 x2 文件保持不变。

评测配置沿用 x2：`pdgpu-5090` 单 GPU、8 CPU、32G；`cuda:0`、BF16、batch size 1、seed 0；完整 HellaSwag、MMLU、GSM8K、ARC-Easy、ARC-Challenge。MMLU/GSM8K 均为 5-shot；复用相同的离线 benchmark 路径、固定依赖版本校验，以及官方任务 YAML 的 prompt、数据处理、生成和评分协议。评测 BF16 设置与最近修改的 FP32 文本训练设置各自独立。`max_new_tokens=2` 只用于模型 preflight 的生成检查，不会限制 GSM8K 正式生成长度。

仅将模型结构审计适配到原 x4：20 个物理层、50 次逻辑执行（5 + 10×4 + 5）、7 个 transient memory slots、5 组/10 个 router 模块及 50 个逻辑 KV cache 槽位。保留 no-cache/prefill、增量解码、greedy generation、save/reload 检查及 router 失衡非致命诊断。

代码经 GitHub 同步后，在远端 `code/RSmol/` 提交：

```bash
bash run_stage3_eval_5_10x4_5_mesh_5090.sh
```

默认结果目录为 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage3_eval_5_10x4_5_mesh_018488/`，包含 `audit_report.json`、`summary.json`、`summary.csv` 和各任务的 `lm_eval_results.json`、样本日志及 `task_protocol.json`。日志位于 `outputs/RSmol/eval_logs/stage3_5_10x4_5_mesh_018488/`，缓存位于 `outputs/RSmol/eval_cache/stage3_5_10x4_5_mesh_018488/`，均与 x2 隔离。可通过 `RSMOL_STAGE3_5_10X4_5_MESH_*` 环境变量覆盖模型、结果路径或评测配置；比较时应与 x2 使用相同配置。本地仅完成代码与 mock 检查，远端权重加载、CUDA 审计及完整 benchmark 结果尚未验证。

## 2026-10-15：文本 x2 / x4 从历史权重开始 FP32 三分之一轮续训（当前设置）

直接修改原 Stage 4 文本训练路线。用户已确认 x2 也只加载权重、使用新 AdamW；两条路线均将指定 checkpoint 的模型权重加载为 FP32，并从本段 optimizer step 0 开始。旧 optimizer、scheduler、RNG、数据游标和历史步数均不恢复；tokenizer 默认从同一 checkpoint 加载，数据从原首轮顺序的起点读取。

| 路线 | 权重初始化目录 | batch geometry |
|---|---|---|
| x2，5 slots | `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244` | 8 GPU × microbatch 8 × GA 16 = GBS 1024 |
| x4，7 slots | `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/formal_2epochs_18488steps_20261007_004959_3090_v1/checkpoint-018488` | 8 GPU × microbatch 4 × GA 32 = GBS 1024 |

完整 epoch 仍固定为 9,244 步，本段取 `9244 // 3 = 3081` 个新 optimizer steps。前 `ceil(3081 × 5%) = 155` 步 warmup 到 `2e-4`，之后沿用原 cosine 公式，min LR 为 `1e-5`；scheduler 总步数为 3,081。强制 FP32 加载、参数和前反向，关闭 BF16 autocast 及 CUDA/cuDNN TF32。其余设置保持：context 1024、原 AdamW 分组/betas/weight decay/eps、全局有效 token 加权梯度、梯度裁剪 1.0、persistent Parquet 直接读取、每 500 步及最终步保存、保留最近三份。

将代码经 GitHub 同步到远端后，在远端 `code/RSmol/` 目录分别提交：

```bash
RSMOL_5_10X2_5_MESH_STAGE4_GATE=FORMAL \
RSMOL_5_10X2_5_MESH_MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244 \
RSMOL_5_10X2_5_MESH_RESUME_FROM= \
RSMOL_5_10X2_5_MESH_MAX_LR=2e-4 \
RSMOL_5_10X2_5_MESH_MIN_LR=1e-5 \
bash run_stage4_5_10x2_5_mesh_3090.sh

bash run_stage4_5_10x4_5_mesh_formal_3090.sh
```

两个正式入口均提交到 `pdgpu-3090`。x2 默认输出为 `formal_continue_3081steps_fp32_from009244_<timestamp>`，x4 为 `formal_continue_3081steps_fp32_from018488_<timestamp>_3090_v1`，分别位于各自原输出根目录；本段最终 checkpoint 为 `checkpoint-003081`。x4 的历史 4090 wrapper 也使用相同训练配置，仅队列和输出后缀不同。

`MODEL_DIR` 是本次加载历史权重的入口，`RESUME_FROM` 仅用于恢复本次新训练段被中断后的完整训练状态；不要将上述历史 checkpoint 填入 `RESUME_FROM`。启动日志和 gate report 的 `initialization` 会记录加载路径、`weights_only_new_training_state` 和起始步数 0。此处记录本地实现，尚未启动远程训练；显存、吞吐及训练结果须以远程日志和 checkpoint 为准。下方单轮和两轮配置均为历史快照。

## 2026-10-15：历史快照：原 7-slot 文本 x4 MeSH 改为 FP32 单轮训练

当前 x4 文本入口为 `code/RSmol/run_stage4_5_10x4_5_mesh_formal_3090.sh`；历史 `run_stage4_5_10x4_5_mesh_formal_4090.sh` 调用相同 trainer，仅队列不同。两个正式入口现均为 1 epoch、9,244 optimizer steps、463-step warmup（ceil(5% × 9244)）、max LR `8e-4`、min LR `2e-5`，随后沿用原 cosine 公式。强制 FP32 参数与前反向，关闭 BF16 autocast 和 TF32，并在启动日志、诊断及 checkpoint 中记录。

其余合同保持：20 个物理层执行 50 个逻辑层（5 + 10×4 + 5）、7 slots、5 组 write/read routers；8 GPU × microbatch 4 × GA 32 = GBS 1024、context 1024、原 AdamW 分组和全局有效 token 加权梯度、裁剪 1.0、persistent Parquet 直接读取、每 500 步及最终步保存、保留最近三份。默认初始化为 `/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x4-5-mesh`。旧 BF16／两轮调度 checkpoint 不能作为本次正式训练的 resume。

代码经 GitHub 同步到远端后，在远端 `code/RSmol/` 目录运行：

```bash
bash run_stage4_5_10x4_5_mesh_formal_3090.sh
```

正式 wrapper 显式指定 LR 并清空 resume，默认输出名为 `formal_1epoch_9244steps_fp32_<timestamp>_3090_v1`；4090 入口对应后缀为 `_4090_v1`。此次仅完成本地代码和检查，远程 FP32 显存、吞吐及训练结果尚待验证。下方 x4 两轮描述均为历史快照，当前设置以本节及路线 README 为准。

## 2026-10-15：历史快照：原 5-slot 文本 x2 MeSH 改为 FP32 单轮训练

直接更新原入口 `code/RSmol/run_stage4_5_10x2_5_mesh_3090.sh` 及其 Stage 4 trainer，没有新建训练路线。按最新确认，`FORMAL` 只训练 1 epoch，共 9,244 个 optimizer steps；固定使用 FP32 参数和前反向，关闭 BF16 autocast 和 TF32。全局前 463 steps（ceil(5% × 9244)）warmup，之后沿用原 cosine 公式，max LR `8e-4`、min LR `2e-5`，scheduler 总步数同为 9,244。保留原 rank 分片分配、首轮分片顺序和分片内行顺序，第 9,244 步结束后停止，不进入第二轮。

其他合同保留：8 GPU、每卡 microbatch 8、GA 16（GBS 1024）、context length 1024、全局有效 token 加权梯度、AdamW betas (0.9, 0.95)、weight decay 0.1（原有 no-decay 分组）、eps 1e-8、梯度裁剪 1.0；每 500 steps 和最终步保存，只保留最近三份。默认初始化仍是远程 `/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x2-5-mesh`，数据路径不变。checkpoint 保留 FP32 和 epoch 合同，游标保存 epoch、分片顺序和行位置；resume 会实际跳过已消费行，并校验训练配置与 manifest，旧 BF16 或两轮调度的 checkpoint 不能作为本轮的 resume。

代码经 GitHub 同步到远端后，在远端 `code/RSmol/` 目录提交新的正式训练（清空 resume，显式指定本轮 LR）：

```bash
RSMOL_5_10X2_5_MESH_STAGE4_GATE=FORMAL \
RSMOL_5_10X2_5_MESH_RESUME_FROM= \
RSMOL_5_10X2_5_MESH_MAX_LR=8e-4 \
RSMOL_5_10X2_5_MESH_MIN_LR=2e-5 \
bash run_stage4_5_10x2_5_mesh_3090.sh
```

默认队列为 `pdgpu-3090`，输出为 `outputs/RSmol/stage4_5_10x2_5_mesh/formal_fp32_<timestamp>/`。本地验证不等于远程训练完成，实际 FP32 显存、吞吐和训练结果须以远程日志与 checkpoint 为准。

## 2026-10-13：标准 Mellow gbs256 完整 30-epoch ReasonAQA 训练代码准备

在 `code/RSmol/mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch256_formal_30epochs/` 新增独立薄覆盖路线，提交入口为 `code/RSmol/run_mellow_official_reasonaqa_gbs256_formal30_5090.sh`。训练合同：8 GPU × 每卡 batch 8 × 累积 4 = GBS 256；从 HTSAT/SmolLM2 预训练组件重新开始，bridge 和 LM 共用 AdamW 的 max LR `4e-5`、min LR `4e-6`，按 optimizer step 做 5% warmup + cosine，30 epochs，FP32，全局梯度 L2 范数硬裁剪 0.5。HTSAT eval/no-grad 冻结，`c2l`、投影和 LM 可训练；空第二音频槽输入静音。文本序列为 260 位双音频前缀 + 不补齐的 prompt/EOS + answer/EOS，只有 answer/EOS 后才右填充；padding 不参与注意力，只监督 answer 与其 EOS，按全局有效目标 token 数归一化。prompt/answer 分别保留 129/250 token 上限并保留 EOS。代码已在本地准备，远程训练结果尚待提交与确认；该路线评测尚未实现。详情见路线 README。

## 2026-10-12：标准 Mellow gbs256 四条独立五 epoch 消融训练路线

在 `code/RSmol/mellow_official_training_c8204d8_adamw_cosine/` 下新增四个薄覆盖目录，均以 `reasonaqa_global_batch256_lr_ablation_5epochs/` 为共同基线，并通过各自的 `PYTHONPATH` 优先加载覆盖文件。每条路线只复制训练必需且需要隔离的入口、trainer、配置生成器和脚本；未复制权重、数据或评测实现。原基线代码未修改。

| 路线目录后缀 | 唯一训练语义改动 | `code/RSmol/` 下的提交入口 |
|---|---|---|
| `reasonaqa_global_batch256_global_clip_5epochs` | 全模型可训练参数的梯度全局 L2 范数硬裁剪至 0.5；不使用 EMA 梯度范数跟踪器 | `run_mellow_official_reasonaqa_adamw_cosine_gbs256_global_clip_1e-3_5epochs_5090.sh` |
| `reasonaqa_global_batch256_attention_mask_5epochs` | 音频前缀全有效，prompt 和 answer padding 无效；因果模型屏蔽未来 token，loss 仍仅计 answer token 与 EOS | `run_mellow_official_reasonaqa_adamw_cosine_gbs256_attention_mask_1e-3_5epochs_5090.sh` |
| `reasonaqa_global_batch256_htsat_frozen_5epochs` | HTSAT 参数冻结，前向处于 eval/no_grad；c2l 与投影保持可训练 | `run_mellow_official_reasonaqa_adamw_cosine_gbs256_htsat_frozen_1e-3_5epochs_5090.sh` |
| `reasonaqa_global_batch256_zero_audio2_5epochs` | 仅原始 `filepath2 == ''` 时第二路输入改为 10 秒、32 kHz 的零波形；读音频失败的回退逻辑不变 | `run_mellow_official_reasonaqa_adamw_cosine_gbs256_zero_audio2_1e-3_5epochs_5090.sh` |

四条线均固定 8 GPU × 每卡 microbatch 8 × 累积 4 = global batch 256、5 epochs、FP32、AdamW、max LR 1e-3、min LR 1e-4、5% warmup 和按优化步的 cosine 调度。将代码经 GitHub 同步到远端后，从远端 `code/RSmol/` 目录分别运行对应脚本并传入 `--max-lr 1e-3`；脚本通过 `vc submit` 提交到 `pdgpu-5090`，使用各自独立的输出目录。2026-10-12 收到的首个 `global_clip` 作业日志显示，共用 staging 脚本因临时目录缺少 `mellow_adamw_cosine_reasonaqa_gbs256_lr_` 前缀而拒绝启动；四条路线的临时目录名现已改为该前缀后接各自路线与 RUN_ID。该作业尚未进入训练，修复后的远端训练结果仍待验证；这四条线的评测尚未实现。

## 2026-10-10：补充交接——标准 Mellow gbs256 学习率／损失归一化消融与统一 MMAU 评测

本段记录标准 mellow_official_training_c8204d8_adamw_cosine 线路，与紧随其后的 Mellow-v0 两阶段 5 + 10×2 + 5 MeSH 主线并行且完全隔离。下方 Mellow-v0 两阶段段落仍是该主线的当前状态；不能把两条线路的模型目录、runtime YAML、checkpoint 或 MMAU evaluator 交叉使用。

### 标准 Mellow 的 gbs256、五 epoch ReasonAQA 学习率消融

- 路线根目录：

  ~~~text
  code/RSmol/mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch256_lr_ablation_5epochs/
  ~~~

- 这是完整 ReasonAQA 训练线路，不是 MCQ。合同固定为 8 ranks、每 rank microbatch 8、gradient accumulation 4、effective global batch 256、5 epochs、FP32、AdamW betas (0.9, 0.95)、weight decay 1e-4、5% warmup 和按 optimizer step 的 cosine 衰减。max_lr 由提交 wrapper 显式传入，min_lr = 0.1 × max_lr；该消融计划覆盖 2e-3、1e-3、8e-4、4e-4、2e-4。
- 训练提交入口是 code/RSmol/run_mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5epochs_5090.sh，提交到 pdgpu-5090、8 GPU、32 CPU、256G。checkpoint 写入 loss_reduction = global_token_mean。
- 本会话用于 MMAU 的已知目标 checkpoint 是用户提供的：

  ~~~text
  /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_lr_ablation_5090/formal_5epochs_maxlr_1e-3_20261009_151832898359917/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr_formal_20_20261009_074133259050800/model--epo-5.ckpt
  ~~~

  本地工作区无法替代远程文件系统确认该 artifact 当前可用；后续操作前应在远程检查文件、同一 run root 下的 runtime_5epochs.yaml 和作业日志。

### 同几何的旧 microbatch 均值损失消融

- 隔离路线根目录：

  ~~~text
  code/RSmol/mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch256_old_microbatch_mean_5epochs/
  ~~~

- 它与 global-token 基线保持同一模型、数据、batch geometry、优化器和 scheduler，只改变损失归一化。8 个 rank、每 rank 4 个累积 microbatch 时使用：

  ~~~text
  L_old = (1 / (R G)) sum_r sum_g [ sum_t CE[r,g,t] / N[r,g] ], R=8, G=4
  ~~~

  即先在每个 microbatch 内对有效答案 token 平均，再对累积步和 DDP ranks 平均；它不是基线的全局有效 token 加权平均。checkpoint 写入 loss_reduction = equal_microbatch_token_mean，因此拒绝从 global_token_mean checkpoint 恢复，反之亦然。
- 提交入口是 code/RSmol/run_mellow_official_reasonaqa_adamw_cosine_gbs256_old_microbatch_mean_1e-3_5epochs_3090.sh。该消融固定 max_lr = 1e-3、min_lr = 1e-4，提交到 pdgpu-3090（8 GPU、32 CPU、256G），默认输出根为：

  ~~~text
  /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs256_old_microbatch_mean_3090/
  ~~~

  从远程 code/RSmol 目录提交：

  ~~~bash
  bash run_mellow_official_reasonaqa_adamw_cosine_gbs256_old_microbatch_mean_1e-3_5epochs_3090.sh --max-lr 1e-3
  ~~~

### 两种 gbs256 checkpoint 共用的 MMAU test-mini evaluator

以下四个文件已经落地，服务于标准 Mellow gbs256 的五 epoch 旧路线和 LR5-based 10 epoch 学习率消融路线：

~~~text
code/RSmol/scripts/evaluate_mmau_test_mini_mellow_gbs256.py
code/RSmol/scripts/evaluate_mmau_test_mini_mellow_gbs256.sh
code/RSmol/submit_mmau_test_mini_mellow_gbs256_3090.sh
code/RSmol/submit_mmau_test_mini_mellow_gbs256_ablation_3090.sh
~~~

- evaluator 对旧路线保留五 epoch 合同；对 `lr_ablation_10epochs_htsat_frozen_global_clip` 接受 `1 <= epoch_completed <= 10` 的 schema-v2 checkpoint，并核对 batch geometry 为 8 × 8 × 4 = 256、AdamW betas、5% warmup、step cosine、min_lr = 0.1 × max_lr、global clip 0.5、冻结 HTSAT 和同一 run root 下的 `runtime_10epochs.yaml`。
- 它根据 checkpoint 的 loss_reduction 自动选择训练代码根：global_token_mean 对应学习率消融路线，equal_microbatch_token_mean 对应旧 microbatch 均值路线。手工传入不匹配的 route-root 会失败，避免错误加载模型代码。
- 生成和评分委托给既有标准 Mellow MMAU 实现，保留 Mellow author-reply prompt／音频协议、按实际选项数量顺序生成 `a)`、`b)`…标签、FP32、129-token prompt、300 个新 token 与官方 MMAU scorer；官方 scorer 和 author-reply scorer 都接收未经标签截断的 raw generation。它不适用于 Mellow-v0 两阶段 checkpoint；后者必须使用已有专用 evaluator。
- 提交 wrapper 使用 pdgpu-3090、1 GPU、16 CPU、64G。checkpoint、runtime YAML 与其 SHA256 的 identity 写在 output directory 同级的 `.<output-dir-name>.gbs256_eval_identity.json` sidecar 中，避免抢占底层 official evaluator 在 output directory 内的 `run_config.json` 所有权记录。旧版目录内 `gbs256_eval_identity.json` 会在内容匹配时自动迁移为 sidecar；非空但没有有效 identity 的目录，或 identity 与本次输入不一致时会拒绝继续，防止不同 checkpoint 的生成混写。

对 LR10 checkpoint 做完整 1000 条评测，从远程 code/RSmol 目录运行：

~~~bash
bash submit_mmau_test_mini_mellow_gbs256_3090.sh \
  /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090/formal_10epochs_lr5_frozen_clip_maxlr_2e-5_20261018_181826682119766/checkpoints/mellow_adamw_cosine_reasonaqa_gbs256_lr10_frozen_clip_20_20261018_101831978638346/model--epo-10.ckpt
~~~

full 是默认模式。wrapper 优先从 checkpoint 所属 run root 读取 `runtime_10epochs.yaml`，旧五 epoch run 才回退到 `runtime_5epochs.yaml`，并将该例输出到：

~~~text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090/formal_10epochs_lr5_frozen_clip_maxlr_2e-5_20261018_181826682119766/eval_mmau_test_mini/mellow_adamw_cosine_reasonaqa_gbs256_lr10_frozen_clip_20_20261018_101831978638346/model--epo-10/full
~~~

复用命令只需替换第一个位置参数为另一个 LR10 `model--epo-<n>.ckpt` 路径：

~~~bash
bash submit_mmau_test_mini_mellow_gbs256_3090.sh /complete/path/to/model--epo-10.ckpt
~~~

可选位置参数依次为 runtime-config、output-dir、full|smoke、dataset-dir、route-root；通常不要手传 route-root，让 evaluator 从 checkpoint 合同自动选择。正式结果必须以输出目录的 evaluation_report.json、官方 scorer 输出和 job log 为准。

当前两个 LR10 checkpoint 已固化在批量入口中。它逐个调用上述单 checkpoint wrapper，因此每个模型使用自己 run root 的 `runtime_10epochs.yaml` 和独立输出目录；单个提交失败会报告失败但不会阻止另一个提交：

~~~bash
bash submit_mmau_test_mini_mellow_gbs256_ablation_3090.sh
~~~

默认是 full；该批量脚本提交两个独立的 1-GPU pdgpu-3090 作业，不等待它们完成。需要临时检查提交链路时才传入 `smoke`。

### 当前可证实状态与下一位 agent 的起点

- 以上训练和评测代码已完成本地静态检查：相关 Python 文件可解析、Bash 语法通过，并以 mock checkpoint／mock vc submit 覆盖了 batch、学习率、loss-reduction、路径派生和输出 identity 的关键拒绝条件。
- 本地工作区没有可用于证明远程训练或 MMAU full job 已完成的回传报告。不要因为代码已经存在就推断远程训练或评测已成功；先检查远程 checkpoint、同级 `.gbs256_eval_identity.json` sidecar、output directory 内的 `run_config.json` 与 `evaluation_report.json`、官方 scorer 输出和提交日志。
- 接手时先区分标准 Mellow c8204d8、Mellow-v0 两阶段 MeSH、以及更下方的历史音频／文本路线；保留所有历史说明，按目标 checkpoint 的合同选择同一路线的 loader、runtime config 和 evaluator。

### 同日补充：其他近期隔离路线索引

以下路线均与标准 Mellow gbs256、以及紧随其后的 Mellow-v0 两阶段主线隔离。除明确给出的用户提供 checkpoint 外，本地代码存在不等于远程训练、生成或 full evaluation 已完成；接手时先查目标输出目录中的报告和 job log。

- **x5、9-slot 文本与音频链路。** 文本路线是 code/RSmol/text_5_10x5_5_mesh_9slot/，核心实现为 recursive_model_5_10x5_5_mesh_9slot.py、convert_stepwise_5_10x5_5_mesh_9slot.py、train_stage4_5_10x5_5_mesh_9slot_ddp.py 和 run_stage4_5_10x5_5_mesh_9slot_formal_3090.sh。固定合同为 20 个物理层、60 次逻辑执行（5 + 10×5 + 5）、9 个 transient memory slots、6 组 router、12 个 router 模块，contract 为 logical_60_physical_20_5_10x5_5_mesh_9slot_6router。它从干净的 SmolLM2-5-10-5 CPU 转换到 /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x5-5-mesh-9slot；formal 文本训练为 pdgpu-3090、8 GPU、microbatch 4、GA 32、3,081 steps。
- **9-slot 音频续训与 MMAU。** 音频路线为 code/RSmol/audio_5_10x5_5_mesh_9slot_mellow_shared_store_configurable_epochs/，从以下 text checkpoint 初始化，且只载入模型／tokenizer，不继承文本 optimizer、scheduler、RNG 或数据游标：

  ~~~text
  /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x5_5_mesh_9slot/formal_third_epoch_3081steps_20261005_212437_3090_v1/checkpoint-003081
  ~~~

  它沿用 fixed-260 shared-store、runtime-zero 第二音频槽、8 GPU × microbatch 8 × GA 4、3 epochs、LR 1e-3 → 1e-4 的音频合同；formal wrapper 是 run_audio_shared_store_formal_configurable_epochs_5_10x5_5_mesh_9slot_mellow_3090.sh。9-slot MMAU 的专用入口是 run_mmau_test_mini_audio_mesh_9slot_3090.sh，默认 checkpoint 是：

  ~~~text
  /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x5_5_mesh_9slot_mellow_shared_store_configurable_epochs/formal_3epochs_20261006_054353274259875-20/checkpoint-011343
  ~~~

  它强制 route=x5_9slot、full、pdgpu-3090 单 GPU，默认输出在 outputs/RSmol/mmau_audio_mesh_zero_slot/x5_9slot/ 下；不得让 x2/x4/x5-7slot evaluator 加载此 checkpoint。
- **x4 文本的当前两 epoch 合同。** code/RSmol/text_5_10x4_5_mesh/ 仍是 20 个物理层、50 次逻辑执行、7 slots、5 组 router 的路线，但当前 formal 合同已经是从 /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x4-5-mesh 初始化，pdgpu-3090、8 GPU、microbatch 4、GA 32、global batch 1024、直接 persistent Parquet 读取、两 epoch 共 18,488 steps。第二 epoch 仅确定性打乱每个 rank 的 shard 顺序，shard 所属和 shard 内行顺序不变；LR 925-step warmup 到 1e-3，再在全程降到 5e-5。当前 formal 入口是 run_stage4_5_10x4_5_mesh_formal_3090.sh，默认输出名为 formal_2epochs_18488steps_<timestamp>_3090_v1。更下方 2026-09-27 的描述是历史快照，不能覆盖这里的 batch、读取方式、队列和 LR 合同。
- **x4 两循环 ReasonAQA generation 探究。** scripts/generate_reasonaqa_x4_two_loop_ablation.py、scripts/generate_reasonaqa_x4_two_loop_ablation.sh、run_reasonaqa_x4_two_loop_ablation_generation_3090.sh 构成独立推理路线。它使用 x4/7-slot checkpoint-011343 的前五个 ReasonAQA test 行，执行 prefix 与前两次中段循环；第二次写入后由 read_routers[4] 读取，再执行最后五层，第三、四次中段循环跳过。该实验不会修改 checkpoint 或训练实现；输出为独立 timestamp 目录中的 samples.jsonl 和 generation_report.json。
- **router／循环相似性分析指标已替换。** run_analyze_audio_mesh_router_loop_similarity_3090.sh 调用 scripts/analyze_audio_mesh_router_loop_similarity.py，在 pdgpu-3090 单 GPU 上比较 x2/x3/x4 的五个固定 ReasonAQA 样本。router 输出的 7 维 softmax 分布使用以 2 为底的 Jensen-Shannon divergence，升序输出；循环模块与 router 参数展平后使用对称 Relative L2 distance：2×||θ1-θ2||₂ / (||θ1||₂ + ||θ2||₂)。旧 cosine similarity 与普通 L2 similarity 已不再是该分析输出指标。
- **标准 Mellow gbs32 训练。** code/RSmol/mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch32_10epochs/ 是完整 ReasonAQA、全局有效答案 token 平均 CE 的隔离副本。目录名保留 10epochs 兼容历史，但当前实际训练是 5 epochs：8 ranks、microbatch 4、GA 1、global batch 32、FP32、AdamW betas (0.9, 0.95)、5% warmup、1e-3 → 5e-5 step cosine、pdgpu-5090。提交入口是 run_mellow_official_reasonaqa_adamw_cosine_gbs32_10epochs_5090.sh。
- 该路线的 MMAU test-mini 独立评测入口为 code/RSmol/submit_mmau_test_mini_mellow_gbs32_3090.sh。它固定校验 schema-v2、5 epochs、4 × 8 × 1 = global batch 32、global_token_mean、AdamW betas (0.9, 0.95)、1e-3 → 5e-5 scheduler，并复用标准 Mellow 的 129-token prompt、FP32、最多 300 token 生成和官方 scorer。默认 checkpoint 是：
  ~~~text
  /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_gbs32_5090/formal_5epochs_20261009_111354/checkpoints/mellow_adamw_cosine_reasonaqa_gbs32_formal_20_20261009_031412485900644/model--epo-5.ckpt
  ~~~
  从远程 code/RSmol 目录提交 full evaluation：`bash submit_mmau_test_mini_mellow_gbs32_3090.sh`；也可以把第一个参数替换成同一路线的其他完整 checkpoint。输出目录和该 checkpoint 所属 run root 隔离，identity 使用 `.gbs32_eval_identity.json` sidecar。
- **5-10×2-5 官方 Mellow MeSH 的独立 full evaluator。** code/RSmol/mellow_official_training_adamw_cosine_5_10x2_5_mesh/ 下的 run_mmau_test_mini_mellow_official_training_adamw_cosine_5_10x2_5_mesh_full_3090.sh 和 run_mmar_mellow_official_training_adamw_cosine_5_10x2_5_mesh_full_3090.sh 分别是 MMAU、MMAR full 入口。它们只接收该 route-root 的 checkpoint 与 runtime config，使用 pdgpu-3090 单 GPU；脚本中的 30-epoch 默认路径必须先在远程验证存在，不能把它当成已完成作业的证明。

## 2026-10-10：最新交接补充（优先于所有下方历史快照）

本节记录当前会话结束时已知的最新事实。它只覆盖同一事项的旧状态，不删除下方任何仍有参考价值的历史路线、实验背景或旧产物记录。下一位 agent 应先读本节，再读取目标隔离路线内的文档、远程 runtime config、checkpoint 审计和作业日志；远程证据优先于本地静态检查。

### 当前活跃路线：Mellow-v0 初始化的两阶段 ReasonAQA 训练

当前需要重点区分的路线为：

~~~text
code/RSmol/mellow_official_training_mellow_v0_two_stage_adamw_cosine_5_10x2_5_mesh/
~~~

它是从原生 Mellow-v0 初始化的 5 + 10×2 + 5、5-memory-slot、3 组 write/read router 的递归 MeSH 路线；它与下列旧路线相互隔离，不能混用模型、runtime config、checkpoint 或 evaluator：

~~~text
code/RSmol/mellow_official_training_adamw_cosine_5_10x2_5_mesh/
code/RSmol/mellow_official_training_c8204d8_adamw_cosine/
code/RSmol/mellow_official_training_c8204d8/
~~~

训练主调用链为：

~~~text
submit_stage{1,2}_{smoke,resume,formal}_3090.sh
  -> submit_reasonaqa_stage_3090.sh
  -> scripts/rsmol/run_reasonaqa_8gpu_stage.sh
  -> stage_reasonaqa_raw_audio.py
  -> write_reasonaqa_runtime_config.py
  -> torchrun --nproc_per_node=8 train.py
  -> training/trainer.py
~~~

每次训练仍在计算节点的 /dev/shm 重新 staging 原始音频；正式训练必须从 pdgpu-3090 的 8-GPU wrapper 提交，不能在登录节点直接运行 CUDA 训练。

### Mellow-v0 来源、审计与当前初始化产物

- Mellow 权重快照只有 config.json 和 v0.ckpt，位于：
  /hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/Mellow-v0/models--soham97--mellow/snapshots/83672db0dae28764e283210d5bb732621e903d8a/
- Mellow 源码 checkout 不是上述 snapshot；审计所需的 mellow/config/v0_local.yaml 位于：
  /hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/mellow-main/。此前把 snapshot 当作 source root 导致的 v0_local.yaml 缺失错误已经解释清楚，不应再次将两类目录混用。
- CPU 初始化审计已通过：原生文本 decoder 为 30 层、hidden size 576、vocab 49152；目标模型为 30 个逻辑层、20 个物理层、2 次中间循环、5 个 memory slots、6 个 router 对象，即 3 组 write/read routers。逻辑到物理层 schedule 为 0..14, 5..14, 15..19。
- 当前训练必须使用 router-transfer 初始化目录：

  ~~~text
  /hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/converted/mellow_v0_5_10x2_5_mesh_epoch17_routers
  ~~~

  其中需要同时存在 text_model/ 与 mellow_v0_5_10x2_5_mesh_init.pt。
- 该初始化 checkpoint 有 401 个 state tensors、160,374,966 个参数；其中 389 个 tensor 继承 Mellow-v0，12 个 router tensor 精确来自以下 epoch-17 checkpoint：

  ~~~text
  /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5_10x2_5_mesh_5090/formal_30epochs_20261005_175504/checkpoints/mellow_adamw_cosine_reasonaqa_mesh_formal_20_20261005_095511181778133/model--epo-17.ckpt
  ~~~

  router source SHA256 为 3eeed406c6125056d3bc2be4b9d136085a1ded7864e174f18fa83fefb657e4b9。因此当前模型参数均有明确来源，并没有 fresh-random router；5 个 memory slots 是 forward 中的运行时状态，首槽按模型合同承载原始 input embeddings，并不是待初始化的独立参数。
- 初始化同时复制 Mellow-v0 的 HTSAT、c2l 和 bridge 权重；训练合同只冻结 HTSAT backbone，c2l、bridge、文本 decoder 和 routers 均可训练。初始化 artifact 不携带 optimizer、scheduler 或 RNG state。
- 转换报告为 mellow_v0_5_10x2_5_mesh_epoch17_routers_conversion.json，位于远程 outputs/RSmol/mellow_v0/preflight/；转换工具是 code/RSmol/scripts/initialize_mellow_v0_to_5_10x2_5_mesh.py。部分已有报告打印 /mnt/cloudstorfs/... 前缀，实际提交前应以当前机器上可解析的路径和 wrapper 默认的 /hpc_stor03/... 初始化根目录为准。

### 两阶段训练合同与已产生的 checkpoint

| 项目 | Stage 1 | Stage 2（当前默认版本） |
|---|---|---|
| 初始化 | epoch-17-router Mellow-v0 初始化 artifact | 仅加载 Stage 1 的模型权重 |
| 训练时长 | ReasonAQA 5 epochs | 额外、独立的 ReasonAQA 5 epochs |
| HTSAT | 冻结 | 冻结 |
| 其他可训练部分 | c2l、bridge、text decoder、routers | c2l、bridge、text decoder、routers |
| 学习率 | routers 1e-3 -> 1e-4；其他 1e-4 -> 1e-5 | 单一参数组 5e-4 -> 5e-5 |
| scheduler | 5% optimizer-step warmup，随后 step-level cosine | 同左 |
| 每 rank batch / GA | 8 / 4 | 4 / 1 |
| effective global batch（8 GPUs） | 256 | 32 |
| 精度 | FP32，use_mixed_precision 为 false | FP32，use_mixed_precision 为 false |
| optimizer / scheduler / RNG | 新建 | 新建；不恢复 Stage 1 的 optimizer、scheduler 或 RNG |

write_reasonaqa_runtime_config.py 会拒绝不符合上述 batch geometry 的输入；当前 Stage 2 代码默认是 global batch 32。曾运行过 global-batch-256 或 diagnostic 的 Stage 2 变体，它们是历史实验，不能当作当前 Stage 2 默认合同。

已确认存在并被后续流程使用的正式 checkpoint：

~~~text
# Stage 1，epoch-17 routers 初始化，5 epochs
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0_two_stage_reasonaqa/stage1_formal_5epochs_epoch17_routers/checkpoints/mellow_v0_two_stage_stage1_formal_20_20261007_045627285353623/model--epo-5.ckpt

# Stage 2，global batch 32，5 epochs
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0_two_stage_reasonaqa/stage2_formal_5epochs_gbs32_20261009_111507/checkpoints/mellow_v0_two_stage_stage2_formal_20_20261009_031515074161572/model--epo-5.ckpt
~~~

Stage 1 的 20-step smoke retry 已通过，产物在 stage1_smoke_20steps_retry1 下的 model--step-20.ckpt。每个 schema-v2 full checkpoint 都应由 scripts/rsmol/audit_stage_checkpoint.py 审计其 route、stage、batch geometry、trainability、parameter groups、scheduler 和 completion contract；不要仅按文件名判断来源。

### 当前评测与未解决的质量回退

- 两阶段路线的 MMAU test-mini evaluator 是 scripts/rsmol/evaluate_mmau_test_mini_mellow_v0_two_stage.py 与 scripts/rsmol/evaluate_mmau_test_mini_mellow_v0_two_stage.sh。
- 通用提交入口 submit_mmau_test_mini_mellow_v0_two_stage_3090.sh 对应 Stage 1 风格调用；当前 global-batch-32 Stage 2 checkpoint 的专用入口是 submit_mmau_test_mini_stage2_gbs32_3090.sh，默认输出目录为：
  /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0_two_stage_reasonaqa/eval_stage2_formal_epoch5_mmau_test_mini_20261009_gbs32。
- 用户已观察到：当前 Stage 2 的选项正确率低于 Stage 1，文本生成质量也下降。这是未解决的回退问题；目前没有证据能够归因于训练实现、初始化、数据/staging、checkpoint/evaluator 不匹配或学习率选择中的任一单项。不得把它描述为已修复或已定位。
- 曾为吞吐异常加入 training/runtime_diagnostics.py。仅在提交时显式设置 MELLOW_RUNTIME_DIAGNOSTICS_STEPS=30 才会启用；它默认关闭，且诊断窗口会额外同步 CUDA stream，不应使用启用诊断的短窗口衡量正式训练吞吐。此前 Stage 1/2 出现的约三倍速度差尚无确定根因结论。
- 当前下一步是对完整训练链做独立代码审查，优先顺序：submission/wrapper、run_reasonaqa_8gpu_stage.sh、staging、runtime config writer、training/trainer.py、training/runtime_diagnostics.py、dataset/sampler、distributed helper、audio/recursive model 和 checkpoint audit。审查尚未产出结论，因此不要更改历史实验的解释或删除任何 checkpoint。

### 已新增的 CPU-only Mellow 对照 case 分析

已新增无模型导入、无 GPU 的脚本：

~~~text
code/RSmol/scripts/compare_mmau_mellow_reproduction_cases.py
~~~

它比较以下两份 raw_generations.jsonl：

~~~text
# 复现模型
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5090/formal_30epochs_20261001_003838/eval/mmau_test_mini_official_training_v1/raw_generations.jsonl

# 原生 Mellow
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/mmau_test_mini_mellow_author_reply_matched_smollm2_113430_v2/raw_generations.jsonl
~~~

脚本按 row_index + id 对齐，验证题目、选项和 gold metadata，并只列出原生 Mellow 正确、复现模型错误的样本。它生成两份独立 Markdown 报告：

~~~text
mellow_correct_reproduction_wrong_choice_letter.md
mellow_correct_reproduction_wrong_official_text.md
~~~

前者严格按 Mellow author-reply 规则比较首个 ) 前的 choice label，转小写且不额外 trim；后者复现并 SHA256 校验 MMAU 官方 scorer 的 token-set string_match。它还输出 comparison_audit.json。截至本次 README 同步，脚本已落地但其远程运行结果尚未回传，不能假定报告已生成或完整。

### 新 agent 的最短接手顺序

1. 先读本节和 code/RSmol/mellow_official_training_mellow_v0_two_stage_adamw_cosine_5_10x2_5_mesh/ADAMW_COSINE_ROUTE.md。
2. 读取本次目标 checkpoint 同目录的 runtime_stage*.yaml、staging_report.json、作业日志和 audit_stage_checkpoint.py 输出。
3. 明确目标究竟是旧 MeSH 官方路线、Mellow-v0 两阶段路线，还是原生 Mellow 评测；三者不可交叉引用配置。
4. 对性能或质量问题先记录证据，不基于历史聊天推断远程状态；如更新本 README，请在本节上方添加新的带日期补充，而不是删除历史路线说明。

## 2026-10-03：交接补充（历史快照；已由 2026-10-10 更新覆盖）

本节只补充最近已经完成的路线和会影响下一次任务判断的当前事实。后面的按日期排列的小节仍然保留历史实验记录；如果历史快照与本节冲突，以本节、目标路线目录中的 README、远程 runtime config/report 和最终 checkpoint 审计为准。

### 当前工作边界和远程资源

- 本地工作区是 C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge；远程代码仓库是 /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM，RSmol 代码位于 /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol。
- 权重、原始音频、parquet 数据和训练输出只在远程服务器；本地主要保存核心代码、脚本、测试和文档。通过 GitHub 在本地和远程之间同步。
- GPU 任务必须使用远程 vc submit/wrapper 入口；不要在登录节点直接运行 CUDA 训练。
- 主要队列是 pdgpu-3090（文本、MeSH 音频和 official Mellow 路线）；AdamW/cosine 对比线和部分旧评测使用 pdgpu-5090。单 GPU 最多 8 个 CPU 核、32G 内存；完整约 70G 原始音频 staging 需要 8 卡节点约 256G。
- 原始音频 staging 是每次作业在计算节点 /dev/shm 中重新建立的临时副本，作业结束后清理，不能理解为一次 staging 后永久可用。

### 最近新增的文本路线：8 memory slots

最近已经新增两条与既有 7-slot 路线隔离的文本路线；本地 Python 静态检查和 contract/schedule 检查已通过，但本交接时没有远程转换、smoke 或 formal PASS 证据：

| 路线 | 逻辑轨迹 | memory slots | router groups | contract |
|---|---:|---:|---:|---|
| text_5_10x4_5_mesh_8slot | 5 + 10×4 + 5 = 50 | 8 | 5 | logical_50_physical_20_5_10x4_5_mesh_8slot_5router |
| text_5_10x5_5_mesh_8slot | 5 + 10×5 + 5 = 60 | 8 | 6 | logical_60_physical_20_5_10x5_5_mesh_8slot_6router |

对应文件分别使用独立的 _8slot 名称，例如：

    code/RSmol/recursive_model_5_10x4_5_mesh_8slot.py
    code/RSmol/scripts/convert_stepwise_5_10x4_5_mesh_8slot.py
    code/RSmol/scripts/convert_stepwise_5_10x4_5_mesh_8slot.sh
    code/RSmol/scripts/train_stage4_5_10x4_5_mesh_8slot_ddp.py
    code/RSmol/scripts/train_stage4_5_10x4_5_mesh_8slot_ddp.sh
    code/RSmol/run_stage4_5_10x4_5_mesh_8slot_formal_3090.sh
    code/RSmol/text_5_10x4_5_mesh_8slot/README.md

    code/RSmol/recursive_model_5_10x5_5_mesh_8slot.py
    code/RSmol/scripts/convert_stepwise_5_10x5_5_mesh_8slot.py
    code/RSmol/scripts/convert_stepwise_5_10x5_5_mesh_8slot.sh
    code/RSmol/scripts/train_stage4_5_10x5_5_mesh_8slot_ddp.py
    code/RSmol/scripts/train_stage4_5_10x5_5_mesh_8slot_ddp.sh
    code/RSmol/run_stage4_5_10x5_5_mesh_8slot_formal_3090.sh
    code/RSmol/text_5_10x5_5_mesh_8slot/README.md

转换器从干净的 /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5 出发；训练器会检查逻辑层数、物理层数、slot 数、router group 数和 contract。远程执行时要分别使用 x4/x5 的模型输出目录和训练输出目录，不要交叉使用 checkpoint。

共同文本配置仍按这组 MeSH 文本线的合同：pdgpu-3090、8 ranks、microbatch 4、GA 32、effective global batch 1024、context 1024、BF16、AdamW betas (0.9, 0.95)、weight decay 0.1、1e-3 -> 1e-4 step-level cosine、三分之一参考 epoch 为 3,081 optimizer steps。具体以对应 wrapper 生成的 runtime config 为准。

### 官方 Mellow 路线的当前状态

官方 training 分支已经复制到并隔离在：

    code/RSmol/mellow_official_training_c8204d8/
    code/RSmol/mellow_official_training_c8204d8_adamw_cosine/

目标 commit 为 c8204d8eb99b4384fd7a76ad57995731e0c0c2bf。此前 2026-09-29 历史快照中“尚未 vendoring”的表述已经过时：当前官方代码、独立环境适配、8 卡 staging、answer-token loss、full-checkpoint resume 修复和提交 wrapper 均已落地。官方 8 卡 staging smoke 和 full-checkpoint resume smoke 已由远程报告确认通过；30 epoch formal 是否完成仍必须查看远程最新报告，不能由 smoke 推断。

官方路线保留官方数据/模型训练主语义，项目适配主要是远程路径、vc submit/8 卡 torchrun、每作业 raw-audio staging、输出目录和可恢复 checkpoint。音频 loss 使用右移后的 token-aligned NTP，只对 answer token（包括 EOS）计算；audio prefix 和 prompt 参与 attention，但 label mask 为 -100，padding 也不计 loss。

官方路线环境为 mellow_c8204d8；AdamW/cosine 对比线只改变优化器和 scheduler：AdamW、betas (0.9,0.95)、weight decay 1e-4、max LR 1e-3、warmup 为 optimizer steps 的 5%、step-level cosine、min LR 5e-5。两条路线的模型、数据、staging、DDP、精度和 checkpoint 语义保持隔离。

### 当前 MeSH 音频和评测约定

已确认的 3-epoch 音频 checkpoint 包括：

    /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_7slot_mellow_shared_store_configurable_epochs/formal_3epochs_20261002_metadatafix_v1/checkpoint-011343
    /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/formal_3epochs_20261002_configfix_v3/checkpoint-011343
    /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x5_5_mesh_7slot_mellow_shared_store_configurable_epochs/formal_3epochs_20261002_151018142328161-20/checkpoint-011343

MeSH fixed-260 音频路线通常使用 8 GPU × microbatch 8 × GA 4 = effective global batch 256；raw audio 从本次作业 staging 的 /dev/shm 读取。单音频样本第二个 audio slot 按当前 runtime_zero 合同使用 GPU 上创建的全零 waveform，不复制第一条真实音频；双音频样本使用两条真实音频。

`text_5_10x3_5_mesh_7slot` 和 `audio_5_10x3_5_mesh_7slot_mellow_shared_store_configurable_epochs` 也有隔离实现；本交接时没有确认 x3 音频路线的最终正式 checkpoint，不要把它与下方已有 x2/x4/x5 checkpoint 混用或默认为已验证完成。

code/RSmol/scripts/evaluate_mmau_test_mini_audio_mesh_shared_store.py 支持 x2_7slot、x4 和 x5_7slot 路线，当前评测约定为 FP32、fixed-260、单音频第二槽 GPU zero waveform、prompt 最大 129、最多生成 300 token，并同时保存 Mellow 作者回复计分和官方 MMAU 诊断。评测和 smoke 也必须通过 pdgpu-3090 wrapper；不要把旧 evaluator 的“第二槽复用第一音频”语义带到这条新合同。

### 新会话最先做什么

1. 先确认任务属于哪条隔离路线：循环次数、memory slots、router groups、文本或音频阶段。
2. 读取目标路线目录内 README、model/converter/trainer/evaluator 和 wrapper；不要只依赖历史段落。
3. 检查远程当前 checkpoint、runtime config、staging report 和 job 日志；本地静态检查不能记为远程 PASS。
4. 新实验使用独立的模型输出、训练输出和评测输出目录；resume 必须是 schema-version 正确的 full checkpoint。
5. 训练或评测完成后，把远程 report、checkpoint 和状态补回本 README。

## 2026-10-01：文本 5-10x2-5、7 memory slots / 3 router groups 隔离路线（历史记录）

新增独立路线 `text_5_10x2_5_mesh_7slot`，不修改原 5-slot 的 5-10x2-5，亦不修改
5-10x4-5 或 5-10x6-5。模型为 20 个物理层、30 个逻辑层，轨迹 `5 + 10×2 + 5`；
每次 forward 有 7 个瞬时记忆槽和 3 组独立 write/read routers（6 个线性模块）。转换沿用
现有 30→20 源层映射，router 按 7-slot 初始化合同新建。

转换程序为 `scripts/convert_stepwise_5_10x2_5_mesh_7slot.sh`（CPU-only）；模型定义、转换器、
训练器及 smoke/resume/formal wrapper 都使用 `_7slot` 独立命名。训练使用 `pdgpu-3090`、
8 ranks、microbatch 4、GA 32、effective global batch 1024、1024-token context、BF16、
AdamW 和 1e-3→1e-4 step-level cosine。直接读取持久 parquet，不做 `/dev/shm` staging；
正式目标为参考 9,244 optimizer steps 的三分之一，即 3,081 steps，warmup 155 steps，
每 500 步及最后一步保存，保留最近 3 个完整 checkpoint。

本地新增静态合同测试已通过；这段记录反映该路线刚落地时的状态。后续已产生对应文本
初始化 checkpoint，具体当前路径和远程状态以路线 README、checkpoint audit 和最新报告为准。
远程执行顺序、路径和配置见
`code/RSmol/text_5_10x2_5_mesh_7slot/README.md`。远程 PASS 只能依据作业日志、report 和
checkpoint audit 记录，不能由本地检查推断。

## 2026-09-29：Qwen2-Audio-Instruct MMAU/MMAR 隔离对比评测

新增一条不转换权重、不影响现有 RSmol/Mellow 评测的 Qwen2-Audio-Instruct 隔离路线。
远程默认权重为
`/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen2Audio-Instruct`；加载固定使用本地
Hugging Face artifact、单张 `cuda:0` 和 BF16，不再包含 FP32 或显存试探分支。
三个执行入口默认使用
`/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/rsmol/bin/python`，不会继承登录
shell 的 `base` Python；可用 `RSMOL_QWEN2_AUDIO_PYTHON` 覆盖。

MMAU test-mini 完全复用当前
`audio_5_10x2_5_mesh_mellow_shared_store_configurable_epochs` 的 Mellow 作者回复协议：
parquet 物理顺序、官方 ID WAV 优先、小写固定选项 prompt、129-token 上限、32 kHz
repeat/random 10 秒裁剪、300-token top-p filter 后 argmax，以及完整 1000 条官方分母。
Qwen 专用作者 choice-label scorer 接受 `A)`/`a)` 或 `(A)`/`(a)`，大小写不敏感；这项
语义只作用于 Qwen2-Audio 路线，其他评测版本保持原逻辑。
MMAR 完全复用官方 metadata 顺序、固定选项、32 kHz 首 10 秒/右补零、32-token greedy、
动态 prediction key 与完整分母。两条路线都将同一份未经预解析的 Qwen decoded prediction
同时交给 `)` 前 choice-label-prefix scorer 和对应官方 scorer。

代码入口为 `run_mmau_test_mini_qwen2_audio_instruct_3090.sh`、
`run_mmar_qwen2_audio_instruct_3090.sh` 和 CPU artifact 预检
`run_qwen2_audio_instruct_eval_preflight.sh`。当前只登记本地代码与静态检查就绪；远程权重
preflight、GPU smoke/full 和正式分数尚未运行，不能记为 PASS。完整合同、输出文件和
smoke→full 命令见 `code/RSmol/qwen2_audio_instruct_eval/README.md`。

## 2026-09-29：交接状态历史快照（已由 2026-10-03 补充覆盖）

### 仓库、远程环境与状态边界

- 本地仓库只保存核心代码，不保存模型权重、正式数据集或远程训练输出。远程代码目录是 `/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol`，现有主环境是 `/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/rsmol`；代码通过 GitHub 在本地与远程之间同步。
- 本次 README 更新前，`main` 与 `origin/main` 都位于 `a984b96`，工作区干净。该提交已经包含 physical-batch-32 probe 的短 job 名修复；本节 README 更新本身仍需由后续操作者按正常流程提交并推送。
- 本节记录的是 2026-09-29 当时的状态，保留用于解释后续改动；当前状态请以本文最上方的 2026-10-03 补充、对应隔离路线 README 和最新远程 report/checkpoint 为准。

### 当前主线一：Mellow `official-faithful-v2` shared-store

- 隔离路线目录为 `code/RSmol/audio_smollm2_135m_mellow_official_faithful_v2_shared_store_configurable_epochs/`，锁定官方 Mellow training commit `c8204d8eb99b4384fd7a76ad57995731e0c0c2bf`。它保持 HTSAT 参数冻结但模块处于 train mode，使用 separator token ID 0，并使用官方逐参数 `GradNormTracker`。不要与旧 Mellow-faithful 路线混用 checkpoint 或 report。
- 当前 formal 配置仍是 `8 GPU × microbatch 8 × GA 4 = effective global batch 256`、30 epochs、113,430 optimizer steps。这种梯度累积不与官方每 rank physical batch 32、GA 1 的一次大 batch 严格等价，尤其 HTSAT 处于 train mode 时，BatchNorm、SpecAugment 和随机状态都会按 microbatch 分别执行。
- 第一轮 formal 已实际运行到 `step=3780/113430`，随后在第一个 epoch 边界的 `broadcast_epoch_state()` 中因 NCCL 直接广播非连续 tensor 而报 `ValueError: Tensors must be contiguous`。提交 `700dca5` 已修复：连续 tensor 直接广播；非连续 model/optimizer state 先复制到连续 staging tensor，广播后再 copy back，并记录 staging 审计。由于保存间隔是 5000 步，该轮在失败前没有 checkpoint，只能从头重跑。用户随后确认 formal 已重新跑起来，但截至交接时尚无完整训练 PASS。
- 为评估严格 physical batch 32 方案的显存和速度，已新增完全隔离、只跑两步、不写 checkpoint 的 probe：每 rank microbatch 32、GA 1、8 GPU、FP32，并记录各 rank 峰值显存和吞吐。入口是 `code/RSmol/run_audio_smollm2_mellow_official_faithful_v2_physical_batch32_probe_3090.sh`。提交 `d4fdd32` 修复了参数传递中出现的字面量 `+`，提交 `a984b96` 将 job 名缩短到 60 字符限制以内。尚未收到远程 probe report，不能据此断言 physical batch 32 可用。
- probe 建议从远程 `code/RSmol` 目录运行：

```bash
PROBE_OUTPUT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow_official_faithful_v2_physical_batch32_probe/probe_$(date +%Y%m%d_%H%M%S)
bash run_audio_smollm2_mellow_official_faithful_v2_physical_batch32_probe_3090.sh --output-dir "$PROBE_OUTPUT"
```

### 当前主线二：文本 `5-10x6-5` 恢复训练

- 已有 checkpoint：`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x6_5_mesh/formal_third_epoch_3081steps_20260928_3090_v1/checkpoint-002000`，目标仍是恢复后训练到总计 3,081 optimizer steps，而不是再训练 3,081 步。
- `run_stage4_5_10x6_5_mesh_formal_3090.sh` 已支持显式 `--resume-from`，底层 wrapper 会把 checkpoint 传给 trainer；队列保持 `pdgpu-3090`。
- 恢复训练曾在最初窗口出现 `nonfinite gradient norm`。当前实现保留 `clip_grad_norm_(..., 1.0)`，并将任一 rank 检测到非有限梯度后的整段 GA accumulation window 在所有 rank 上同步丢弃：清空梯度、消费下一段数据、保持 optimizer step 不变，然后重试同一个目标 step。跳过次数写入 diagnostics、report 和 checkpoint，并有连续及累计上限，避免无限静默跳过。
- 后续一次作业出现 C10d rendezvous store 连接超时；结合日志判断更像节点或硬件故障。用户已重新提交 `text_x6_formal`，截至交接时没有最终 `checkpoint-003081` 完成证明。不要把 rendezvous 故障和训练器的非有限梯度策略混为一类问题。

### 下一条独立路线：直接使用 Mellow 官方 `training` 分支

- 用户决定另开一条与现有 RSmol v2 完全隔离的路线，直接以官方仓库 `soham97/mellow` 的 `training` 分支为基础做全面复现。已确认和审计的目标 commit 同样是 `c8204d8eb99b4384fd7a76ad57995731e0c0c2bf`。本条是 2026-09-29 的历史记录；官方源码、环境适配、staging、loss、resume 修复和 wrapper 后续已在本文顶部补充中落地。
- 计划是在远程保留官方训练核心语义，只做基础设施适配：独立环境、远程文本模型和 HTSAT checkpoint 路径、ReasonAQA 数据路径、单节点 8-GPU `torchrun`、`vc submit` 的 `pdgpu-3090` 入口、日志与输出目录，以及将官方代码实际读取的原始音频文件预加载到节点共享 `/dev/shm`。
- 严格官方复现线不要直接复用当前 v2 的预解码 waveform store，因为官方 dataset 在运行时通过音频文件路径加载、重采样和裁剪；改用 waveform store 会改变数据管线语义。现有 JSONL manifest 也未必符合官方 dataset 预期的 JSON 结构，需要先逐字段核对并生成独立的官方格式数据文件。
- 不要直接修改现有 `rsmol` 环境。优先创建独立环境或克隆环境后审计依赖；官方 `requirements.txt` 带有平台和旧版本约束，不能在 Linux 上盲目整文件安装。也不要在未记录的情况下修复官方 sampler、batch、optimizer、scheduler、HTSAT mode、随机音频选择或 resume 语义，否则失去“直接官方代码复现”的可审计性。
- 当时新聊天接手后的下一项工作是完成上述官方源码、配置、dataset、DDP launcher 和依赖审计；该阶段已经推进到本文顶部记录的 vendored/staging/smoke/resume 状态。旧命令和旧状态保留作历史依据，不应覆盖当前路线 README 和远程 report。

### 新聊天的建议阅读顺序

1. 本 README 的本节。
2. `code/RSmol/audio_smollm2_135m_mellow_official_faithful_v2_shared_store_configurable_epochs/README.md`。
3. `code/RSmol/text_5_10x6_5_mesh/README.md`。
4. 最近五个提交：`457fdcf`、`700dca5`、`11036bc`、`d4fdd32`、`a984b96`。
5. 再开始官方 Mellow `training` 分支的独立复现路线；不要继续修改正在运行的现有 formal 训练线，除非新的远程日志证明存在必须修复的问题。

## 2026-09-28：Mellow official-faithful-v2 SmolLM2 shared-store 隔离训练线（历史记录）

新增完全隔离的 audio_smollm2_135m_mellow_official_faithful_v2_shared_store_configurable_epochs 路线。它修正旧 Mellow-faithful 路线相对官方运行时合同的三项偏差：冻结 HTSAT 参数但保持 HTSAT train mode、两个音频分隔符固定使用 token ID 0，以及使用锁定 Mellow commit c8204d8eb99b4384fd7a76ad57995731e0c0c2bf 的逐参数 GradNormTracker 代替固定 0.5 梯度裁剪。旧路线的 package、trainer、checkpoint 和 report 保持原合同，不能与 v2 混用。

v2 checkpoint 保存各 rank 的 HTSAT mutable buffers 和 GradNormTracker running_norm；resume 在首个恢复 forward 前加载本 rank buffer，并保留 DataLoader iterator 创建后的第二次 RNG 恢复。普通 train-mode BatchNorm 在 epoch 内可按 rank 分化；每个完整 epoch 结束后从 rank 0 广播模型状态与 optimizer tensor state。fresh smoke 固定运行 0 到 20，resume 固定运行 20 到 22；formal 必须同时读取本路线两个 PASS report。qualification 为 8 GPU × microbatch 4 × GA 1，formal 为 8 GPU × microbatch 8 × GA 4、30 epochs、每 5000 步保存并保留最新 4 份。所有提交 wrapper 使用 pdgpu-3090。

截至 2026-09-29，formal 已实际运行到首个 epoch 边界，暴露并修复了非连续 tensor 的 NCCL 广播问题；修复后的 formal 已从头重新启动，但 30-epoch 完成状态仍待远程 report 与最终 checkpoint 证明。另有隔离的 physical-batch-32 两步显存 probe 已就绪，远程结果尚未回传。完整合同和命令见 code/RSmol/audio_smollm2_135m_mellow_official_faithful_v2_shared_store_configurable_epochs/README.md。

## 2026-09-28：x4 MeSH 音频 shared-store 隔离训练线（历史记录）

新增隔离路线 audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs。
文本模型从 x4 三分之一 epoch 的 checkpoint-003081 仅加载模型与 tokenizer，音频训练
从 global step 0 开始，不继承文本 optimizer、scheduler、RNG 或数据游标。HTSAT 保持
冻结；Mellow c2l、原有 768→576→576 映射层和完整 x4 文本模型参与训练。固定保留两个
129-token 音频槽和两个分隔符：结构性单音频行的第二槽使用 GPU 运行时创建的全零
waveform，不复用第一槽真实音频；双音频行正常使用两个真实输入。训练使用完整 unique
waveform store、全 manifest shuffle、8 GPU × microbatch 8 × GA 4、3 epochs、
LR 1e-3→1e-4、每 500 步保存并保留最新 4 份。smoke、resume 和 formal 提交入口全部
使用 pdgpu-3090。完整合同与命令见
code/RSmol/audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/README.md。

## 2026-09-28：文本 5-10x6-5 MeSH 隔离训练线

在已成功启动的文本 5-10x4-5 最终 direct-parquet 路线基础上，新增完全隔离的 5-10x6-5 MeSH 路线：20 个物理 decoder 层执行 70 个逻辑层，十层中段循环六次；每次 forward 使用 7 个瞬时记忆槽，前段结束后和六次中段循环后分别使用一组独立 write/read routers，共 7 组、14 个 router 模块。20 个继承物理层继续使用 5-10x2-5/x4 相同的 30→20 层映射，新 routers 按七槽合同确定性初始化。

模型转换仍从干净的 /hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5 在 CPU 终端执行。训练保持 x4 最终合同：8 ranks、每卡 microbatch 4、GA 32、effective global batch 1024、context 1024；直接流式读取远程挂载 parquet，按排序后 shard index modulo 8 固定分片，不在正式训练中复制到 /dev/shm。训练目标为三分之一历史 epoch，即 3,081 optimizer steps；warmup 155 steps，LR 从 1e-3 cosine 降至 1e-4；每 500 步和最终步保存，只保留最近 3 个完整 checkpoint。Stage-1、smoke、resume 和 formal 提交入口均使用 pdgpu-3090。完整合同和命令见 code/RSmol/text_5_10x6_5_mesh/README.md。

## 2026-09-27：文本 5-10x4-5 MeSH 隔离训练线代码就绪

新增隔离文本路线 5-10x4-5-mesh：20 个物理层执行 50 个逻辑层，十层中段循环四次，使用 7 个瞬时记忆槽和 5 组独立 write/read routers。训练保持旧文本 5-10x2-5 的 8-rank 固定 parquet 分片分配、microbatch 8、GA 16 和 1024-token 合同；正式训练固定为 2 个 epoch，每 epoch 9,244 optimizer steps，总计 18,488 steps，925-step warmup，LR 从 1e-3 cosine 降至 1e-4。每 500 steps 和最终步保存 checkpoint，只保留最近三份。

模型转换只在 CPU 终端直接执行，不提供提交 wrapper。所有审计、smoke、resume 和 formal GPU 作业使用 pdgpu-4090。每个训练作业会先把约 24G 的完整 parquet 数据只复制一份到节点共享 /dev/shm，对源目录与 staged copy 做逐 shard footer、行数、字节数和 SHA256 一致性检查，然后 8 个 rank 共同读取。完整命令、文件清单和远程 PASS 边界见 code/RSmol/text_5_10x4_5_mesh/README.md。

## 2026-09-26：原生 Mellow-v0 的 MMAU/MMAR 公平对比评测

原生 Mellow-v0 的 MMAU test-mini 评测已重新锁定到与
`formal_30epochs_20260923_v1/checkpoint-113430` 对照结果相同的协议：使用
MMAU-v05.15.25 的 1000 条完整分母，同时执行 Mellow 作者回复中的小写固定选项 prompt、
129-token prompt、32 kHz/10 秒 MellowWrapper 音频处理、最多 300 tokens 的
top-p filter 后 argmax，并把同一份未预解析预测分别交给作者 choice-label scorer 和新版
MMAU 官方 scorer。新的默认输出目录为
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/mmau_test_mini_mellow_author_reply_matched_smollm2_113430_v2`，避免复用旧结果。

此前没有原生 Mellow-v0 的 MMAR 入口；现在新增
`scripts/evaluate_mmar_mellow_v0.py`、运行脚本和 `pdgpu-4090` 提交入口。它与
`mmar_checkpoint_113430_dual_scoring_v1` 共享官方 MMAR metadata 顺序、完整分母、
固定选项 prompt、首 10 秒音频、32-token 生成上限、动态探测的官方 prediction key，
并将同一份原始预测同时交给官方 MMAR scorer 与 choice-label-prefix scorer。原生
Mellow-v0 固有的 129-token prompt 和 FP32 推理与对照 checkpoint 的 476-token/BF16
能力差异会明确写入 report，不会伪装成完全相同的模型运行合同。两条评测都要求先运行
`run_mellow_v0_artifact_preflight.sh`，当前状态仅为本地代码和静态检查就绪，远程 GPU
分数仍以各自 `evaluation_report.json` 为准。

## 2026-09-26：隔离的 Mellow-faithful SmolLM2 shared-store 路线代码就绪

仅修改 audio_smollm2_135m_mellow_shared_store_configurable_epochs 路线，旧 SmolLM2 fixed-260 shared-store 与全部 MeSH 路线保持原实现。新路线锁定 Mellow training 分支 commit c8204d8eb99b4384fd7a76ad57995731e0c0c2bf，使用完整变长 32 kHz 单声道 waveform store、运行时独立随机 10 秒裁剪、缺失音频槽的随机 filepath1 音频、固定 639-token 布局、Adam 与 epoch-level cosine。

按最新确认改为 Mellow 的 global-batch 几何：8 GPU × microbatch 4 × GA 1，effective global batch 32；968,059 条数据每 epoch 30,251 optimizer steps，30 epochs 共 907,530 steps。每个 epoch 保存 checkpoint，但只保留最后三份。正式训练前必须完成 fresh smoke20、独立 uninterrupted reference22、从 step 20 恢复到 step 22 的精确对照，以及 formal gate。当前状态仅为本地代码与静态检查就绪，远程 GPU 尚未登记 PASS。完整生成和提交命令见 code/RSmol/audio_smollm2_135m_mellow_shared_store_configurable_epochs/README.md。

## 2026-09-25：2～10 次共享递归训练阶段 5～8 代码就绪

从 fixed-T=2 shared-store `checkpoint-011343` 迁移得到的可变递归深度路线，阶段 1～4
（初始化迁移、全深度结构/抽样梯度审计、T=10 正式配置显存预检）已由用户确认远程 PASS。
阶段 5～8 现已使用独立 trainer 和提交入口准备完成，不修改原 fixed-T=2 shared-store 线。
训练时每个 micro-step 由 rank 0 使用独立 CPU generator 均匀采样一个
`T ∈ {2,...,10}`，广播到全部 8 个 rank，并完整反向传播。由于 `T=2` 不使用
`refine_read`，新 trainer 不跨不同 T 使用 DDP `no_sync`；每个 micro-step 都单独完成梯度
同步，再进行 GA=4 累积，避免最后一个 micro-step 恰为 T=2 时早先的 `refine_read` 梯度
停留在 rank 本地。

阶段划分为：阶段 5 隔离训练器与 shared-store staging；阶段 6 fresh smoke20；阶段 7 从
`checkpoint-000020` 精确恢复并续跑到 step 22；阶段 8 仅接受本路线阶段 6/7 PASS report
的 7-epoch formal。深度采样器的 generator state、draw cursor、histogram、8-rank 一致性摘要
与各 rank RNG 一同进入 checkpoint；resume 会逐项校验。968,059 行正式数据对应
3,781 step/epoch、26,467 总 optimizer steps、1,324 warmup steps。远程阶段 6～8 尚未执行，
当前状态只能记为“本地代码与静态检查就绪”，不能记为 GPU PASS。完整命令和文件清单见
`code/RSmol/audio_5_10x2to10_5_mesh_mellow_shared_store/README.md`。

## 2026-09-25：30-epoch shared-store checkpoint-113430 的 MMAU/MMAR 评测

MeSH 与原始 30 层 SmolLM2 shared-store 对比线的默认评测 checkpoint 已切换到各自的
`formal_30epochs_20260923_v1/checkpoint-113430`。checkpoint 仍可通过命令行
`--checkpoint` 或提交 wrapper 的环境变量传入，评测器不再把 epochs、warmup、总步数或
checkpoint 编号写成固定放行标准；shared-store 审计只保留可迁移的完成性证明：
`checkpoint-<step>` 目录名、completion marker、training-state cursor、8-rank RNG、固定
260-token 双槽、shared-store/模型架构合同，以及非空的 bridge/c2l 权重。

MMAU 对两条线均使用 Mellow 作者回复协议：小写 prompt、129-token prompt 截断、最长 300
tokens、FP32、`top_p=0.8` filter 后 argmax、单音频复用 audio1 HTSAT embedding 并分别执行
两次 bridge；原始预测不删除开头标签，同时写出作者字母分数和新版官方诊断。MMAR 同样把
同一份原始预测直接交给两套 scorer：`choice_label_prefix_evaluation.{json,txt}` 比较首个
`)` 前的标签，`official_evaluation.txt` 使用官方 MMAR scorer。answer 无法精确映射 choice、
空预测和逐样本 skip 都保留在完整分母内并计错，不因单行 metadata 问题终止整个评分。

单 GPU 提交入口继续限制为最多 8 CPU、32G MEM：

```bash
bash code/RSmol/run_mmau_test_mini_audio_5_10x2_5_mesh_mellow_shared_store_4090.sh
bash code/RSmol/run_mmar_audio_5_10x2_5_mesh_mellow_shared_store_5090.sh
bash code/RSmol/run_mmau_test_mini_audio_smollm2_5090.sh
bash code/RSmol/run_mmar_audio_smollm2_5090.sh
```

## 2026-09-23：MMAU 改为 Mellow 作者回复协议并保留新版官方诊断

Mellow 原生线与 10-epoch shared-store checkpoint-037810 的 MMAU test-mini 评测已切换到
Mellow 作者在 GitHub issue #5 中给出的复现协议：问题末字符替换为 `?`、拼接小写
`a) ... b) ...` 选项并将整个 prompt 转小写；prompt 按 MellowWrapper 截断到 129 tokens；
音频使用 MellowWrapper 的 32 kHz/10 秒策略，短音频重复后截断，长音频随机截取；生成
最多 300 tokens，执行 `top_p=0.8`、`temperature=1.0` 的 nucleus filter 后 argmax，
不是随机采样。主要比较分数使用作者回复的选择字母规则，即比较预测和带标签答案在首个
`)` 之前的前缀。1000 条仍全部进入分母，skip 仍写空预测并计错。

当前 `MMAU-v05.15.25` 的官方 `evaluation.py` 使用答案正文 token-set `string_match`，与
作者回复的字母 scorer 不同；评测会同时保存两套结果：
`mellow_author_reply_evaluation.{json,txt}` 是与 Mellow 论文/issue 比较的主结果，
`official_evaluation.txt` 和 report 中的 `mmau_v051525_evaluation` 是新版官方诊断结果。
作者字母 scorer 若遇到 answer 因前后空格等 metadata 问题而无法精确映射到 choices，不再
中止整个任务；该行保留在完整分母中、按错误计分，并记录到 `record_errors`。
需要注意，issue 发布于 2025-04-16，而 `MMAU-v05.15.25` 发布于 2025-05-15，后者约修改
25% 问答和 5% 音频；因此这是“作者推理/计分协议在新版数据上的复现”，不能声称精确复现
论文中的 52.11。

shared-store 评测仅改变 MMAU 输入/生成/评分协议，模型槽位仍严格保持训练合同：同一个
audio1 的 Mellow/HTSAT embedding 复用到第二槽，然后 bridge 分别执行两次。Mellow-v0
原生评测仍将同一音频源分别走两个 MellowWrapper 风格预处理与 audio encoder。两条路线
均使用新的输出目录，禁止续用旧 32-token/新版 scorer 输出。

如需先判断旧输出到底有多少差异来自 scorer，可在不加载模型、不重新推理的情况下离线重算
作者字母规则（这不等于完整协议复现）：

```bash
python code/RSmol/scripts/rescore_mmau_mellow_author_reply.py \
  /path/to/old_output/predictions_fixed_order.json \
  --expected-rows 1000 \
  --output-dir /path/to/old_output/author_reply_rescore
```

## 2026-09-23：Mellow-v0 原生 MMAU test-mini 隔离评测线

新增一条不转换权重、不影响现有 RSmol 评测的原生 Mellow-v0 路线。执行顺序固定为：
CPU-only artifact preflight → MMAU parquet 物理顺序前 5 条 smoke → 同一输出目录继续 full
1000 条并调用官方 `evaluation.py`。preflight 从本地 GitHub checkout 构造原生 Mellow，
离线加载本地 SmolLM2-135M 和完整 `v0.ckpt`，严格覆盖 HTSAT、c2l、projection 与
text decoder；只检查 artifact，不运行仓库中的两个 example 音频。

评测沿用当前 MMAU 的官方 metadata/scorer artifact 审计、append-only resume 和完整分母
语义，但推理与主评分改为上节所述 Mellow 作者回复协议。优先读取
`test-mini-audios/<id>.wav`；目录或单个 WAV 缺失时，回退到经 ID/选项审计的 parquet 内嵌
音频。MMAU 单音频源分别执行两次 MellowWrapper 风格 repeat/random-crop，再分别调用
audio encoder；prefix 为 `129 + 1 + 129 + 1 + 129 = 389` tokens。推理使用 FP32、
`top_p=0.8` filter 后 argmax、`temperature=1.0`、LM 默认 cache 参数、最多 300 tokens；
解码文本不删除开头 `a)`--`d)`。
smoke 未请求官方 scorer，因此只能是 `INFERENCE_ONLY_PASS`；full 才能产生可比较的
官方 `PASS`。成功 smoke 会持久化独立 gate；full 缺少该 gate 或 artifact identity 已变化时
会拒绝启动，full 中断后则仍可依靠同一 gate 和 append-only progress 继续。GPU 提交入口
统一为 `pdgpu-4090`、1 GPU、8 CPU、32G MEM：

```bash
bash code/RSmol/run_mellow_v0_artifact_preflight.sh
bash code/RSmol/run_mmau_test_mini_mellow_v0_smoke_4090.sh
bash code/RSmol/run_mmau_test_mini_mellow_v0_full_4090.sh
```

默认输出目录为
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/mmau_test_mini_mellow_author_reply_protocol_v1`；
full 会复用 smoke 已完成的前 5 条，不重新推理。远程 preflight/smoke/full 尚未运行，
本地静态检查不能写成远程 GPU PASS。

集群上的同一共享存储可能在不同节点分别解析为 `/hpc_stor03/...` 与
`/mnt/cloudstorfs/...`。preflight gate 将这两个前缀视为同一存储别名，但不会仅凭路径
放行：运行时仍重新核对完整 checkpoint SHA256、snapshot config SHA256、Mellow 源码
逐文件 SHA256 和本地 SmolLM2 config/tokenizer inventory。

## 2026-09-23：文本 5-10-5 Adjacent-layer Average 隔离消融线

新增不训练、直接评测的文本初始化消融线
`SmolLM2-5-10-5-adjacent-average`。它保持原固定 5-10-5 的 20 个物理层、
30 个逻辑层和精确 `0..14,5..14,15..19` 执行轨迹不变；源 SmolLM2-135M
的 0--4 与 25--29 层精确复制，中间层按 `(5,6),(7,8),...,(23,24)`
逐张量使用 FP32 算术平均后转换回原 dtype。转换元数据强制记录 10 组 source
pair、FP32 accumulator 和 0--29 每层恰好使用一次；旧单层抽取 mapping 字段为空，
因此新旧 evaluator 会互相拒绝错误 checkpoint。独立 Stage 3 评测继续使用
HellaSwag、MMLU、GSM8K、ARC-Easy 和 ARC-Challenge 的既有离线官方协议。
转换与评测提交入口均固定 `pdgpu-3090`、单 GPU、8 CPU、32G MEM。代码入口为：

```text
code/RSmol/recursive_model_5_10_5_adjacent_average.py
code/RSmol/scripts/convert_stepwise_5_10_5_adjacent_average.py
code/RSmol/scripts/convert_stepwise_5_10_5_adjacent_average.sh
code/RSmol/run_convert_stepwise_5_10_5_adjacent_average_3090.sh
code/RSmol/scripts/evaluate_stage3_5_10_5_adjacent_average.py
code/RSmol/scripts/evaluate_stage3_5_10_5_adjacent_average.sh
code/RSmol/run_stage3_eval_5_10_5_adjacent_average_3090.sh
```

## 2026-09-23：shared-store 可传 epochs 的隔离训练副本

在不修改原 3-epoch shared-store 入口和当前作业的前提下，新增隔离的
configurable-epochs 提交线。新入口要求显式传入正整数 `--epochs`；模型、完整 manifest
shuffle、固定 260-token prefix、单音频复用 audio1 HTSAT embedding 后两次独立 bridge、
answer-EOS、8×8×GA4、LR `1e-3 -> 1e-4`、checkpoint/resume 和 formal gate 均保持不变。
真实调度按 `steps_per_epoch=floor(rows/256)`、`total_steps=steps_per_epoch*epochs`、
`warmup_steps=ceil(total_steps*0.05)` 计算。当前 968,059 条数据传 `--epochs 10` 时为
3,781 step/epoch、37,810 总步数、1,891 warmup steps。10-epoch smoke20、resume2 和 formal
必须全部显式传 `--epochs 10`；旧 3-epoch report/checkpoint 会因 epochs、shape、warmup 或
checkpoint config 不一致而被拒绝。完整命令见
`code/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/README.md`。该路线当前是本地代码就绪，
不表示 10-epoch 远程 smoke/formal 已 PASS。

## 2026-09-22：隔离的整库 node-shared `/dev/shm` 全量打乱训练线

在 shared-store PERF20 通过后，新增隔离训练线
`code/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/`。它把完整 v3 unique waveform
store 在每个作业开始时只复制一次到该计算节点的 `/dev/shm`，8 个 rank 通过 mmap
共享同一个 `waveforms.f32` inode；不会给每个 rank 克隆一份完整 waveform store。
metadata/index 与 waveform store 一起暂存，canonical manifest 单独暂存并通过 SHA256
与 store metadata 绑定。训练使用完整 manifest 上的 `DistributedSampler(shuffle=True)`，
每个 epoch 重新生成全局排列后分配互不重叠的 rank slice，不使用六分区 schedule。

该路线恢复旧 3-epoch 的音频槽语义：Mellow HTSAT/c2l/bridge、固定
260-token single/dual prefix、answer-only loss 和受监督的终止
`<|endoftext|>`。独立合同是
`node_shared_unique_store_fullshuffle_fixed260_audio_reuse_answer_eos_v2`，拥有自己的 trainer、
checkpoint config、report、20+2 精确 resume gate、输出根目录与 `pdgpu-5090` 三个提交入口，
不会修改六分区或 online trainer。单音频第二槽复用 audio1 的 HTSAT embedding，再由 bridge 分别计算；正式入口固定 3 epochs、LR `1e-3 -> 1e-4`、5% warmup
向上取整、每 500 step 保存、保留 4 个 checkpoint。完整命令和审计合同见
`code/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/README.md`。用户已确认改动前的共享存储
smoke/resume 跑通；本次 fixed260 合同尚待远程重新运行 20+2，旧 compact report/checkpoint
不能用于本次放行或续训。本地检查不代表新合同 GPU PASS。

## 2026-09-21：隔离的固定 260-token 运行时静音槽实验

新增隔离训练线 `code/RSmol/audio_5_10x2_5_mesh_mellow_silence_slot/`。每条样本固定保留两个音频槽和 260-token prefix；单音频样本的第二槽使用模型 forward 内在 GPU 即时创建的全零 waveform，每个含单音频样本的 microbatch 只编码一个静音 waveform，再在 trainable bridge 前展开。它不会新增静音文件、partition store 条目或 CPU→GPU 静音传输。显式双音频样本仍使用两个真实槽；显式相同路径可以复用第一路 HTSAT embedding，但不会被当作静音槽。

独立训练合同为 `component_partitions6_rank_ram_fixed260_runtime_silence_second_slot_answer_eos_v2`，LR 为 `1e-3` 经 5% warmup 后 cosine decay 到 `1e-4`。该线拥有独立 model/data 模块、trainer、checkpoint config、20+2 smoke 门禁、report、输出目录和 `pdgpu-5090` 提交入口；拒绝 compact MeSH、SmolLM2 和 recursive checkpoint 跨合同 resume。完整命令和审计项见 `code/RSmol/audio_5_10x2_5_mesh_mellow_silence_slot/README.md`。

> 历史同步：2026-09-23。本文件仍是新 Codex 会话的首要交接依据，但当前状态以本文最上方的 2026-10-03 补充、当前代码行为和最新远程 report 为准。

## 历史状态快照（截至 2026-09-23；当前状态先读 2026-10-03 补充）

> 状态口径：`远程 PASS` 只表示已有真实 report/log/checkpoint 证据；`代码就绪` 表示本地实现和静态检查完成，不能写成 GPU PASS。新 Codex 必须先读本节，再读对应路线的代码和测试。

项目最初在 SmolLM2-135M 上研究循环语言模型，目前唯一默认主线是 ReasonAQA 音频训练：

```text
已解码 waveform（mono、32 kHz、10 s、float32）
  -> 冻结的 HTSAT AudioSet backbone
  -> 可训练的 Mellow c2l(527->768)
  -> 可训练的 Mellow bridge(768->576->576, avgpool8)
  -> 单音频 130-token / 双音频 260-token prefix
  -> 5-10x2-5 MeSH / SmolLM2-135M
  -> 只对 answer token 计算 causal-LM loss
```

当前默认正式训练是六个连通分量 waveform partition；online full-shuffle 和 node-shared `/dev/shm` 是独立对照/实验路线。早期 64-shard mmap 不再是正式方案。

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/
rsmol_reasonaqa_train_component_partitions6_32k_10s_f32_v2
```

每次只加载一个 partition。每个 DDP rank 将该 partition 的全部唯一 waveform 克隆到自己的匿名 CPU RAM，训练完该 partition 后严格释放并核对每 rank RSS 与 cgroup anon 回落，通过后才加载下一 partition。

当前关键状态：

- 六分区物化：远程 `PASS`，共 968,059 条 QA、54,046 条唯一音频、64.42785263061523 GiB waveform；跨 partition 音频复制为 0。
- partition PERF20：p0、p2 均已远程跑通；完整预加载后稳态训练约 1.6–1.7 秒/step。
- 正式训练前的旧 v1（答案未追加 EOS）smoke：用户已确认两次均 `PASS`：
  - p2 加载 → 10 步 → 释放；p0 加载 → 10 步 → 释放；保存 step-20 checkpoint。
  - 从 step-20 checkpoint resume；p1 加载 → 2 步 → 释放；保存 step-22 checkpoint。
- 2026-09-18 已把训练合同升级为 `component_partitions6_rank_ram_compact_audio_answer_eos_v2`：每条 answer 最后恰好有一个受监督的 `<|endoftext|>`。由于输入/标签合同发生改变，旧 v1 smoke 不能放行 v2 正式训练，必须按相同 20+2 流程重跑。
- 正式训练：准备采用 10 epochs；EOS v2 smoke 和正式作业均尚未登记远程完成。
- 最新新增的 shared-store 路线已完成本地代码、package README 和静态合同测试，但远程 smoke20/resume2/formal 尚未运行；它不影响六分区或 online trainer。
- shared-store `/dev/shm` PERF20 已有远程 `PASS`：`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/perf20_shared_waveform_store_tmpfs_20260922_024229830446654-20/perf20_report.json`。该 report 证明共享 store 链路和稳态读取可运行；4090/pdgpu 变更属于当次性能作业，不能直接当作 5090 正式训练结果。
- 正式训练入口会重新读取两个 smoke report，验证训练合同、step 游标、resume 关联、MeSH 30 层轨迹和全部 8 rank 内存释放；验证不通过则拒绝启动。
- 原始 SmolLM2-135M 音频对比线已在本地改造成同一套六分区、compact-prefix、answer-EOS-v2、10-epoch 训练合同，训练 wrapper 使用 `pdgpu-3090`；当前状态为代码就绪，新的 20+2 smoke 与正式训练均尚未登记远程 PASS。
- 2026-09-23 新增原始 SmolLM2-135M 的独立 shared-store configurable-epochs 对比线：从本地 Hugging Face SmolLM2-135M 初始化，使用完整 v3 waveform store、全量 manifest shuffle、固定 260-token 双槽、单音频复用 audio1 HTSAT embedding、双音频分别编码，训练配置目标为 30 epochs（113,430 optimizer steps，5% warmup 为 5,672 steps）。该线使用独立 package/trainer/checkpoint/report/output root 和 `pdgpu-3090` 8-GPU wrapper，仍要求本路线真实 20+2 smoke PASS；当前仅本地代码与静态检查就绪，远程尚未运行。
- 固定 5-10-5 recursive 音频对比线也已改造成六分区、compact-prefix、answer-EOS-v2、10-epoch 合同；保持 20 个物理层和精确 `5-10-10-5` 逻辑轨迹、无 MeSH router/memory，训练 wrapper 使用 `pdgpu-5090`。用户已提供 `partition_formal_eos_v2_10epochs_20260919/checkpoint-037810` 作为完成训练产物；其独立 MMAU/MMAR full 评测代码已就绪，远程 artifact 审计与正式分数尚待评测作业确认。

路线隔离摘要：

| 路线 | 数据/模型定义 | 当前用途 |
|---|---|---|
| MeSH 六分区 | partition store、rank-local CPU RAM preload/release | 默认正式训练 |
| MeSH shared-store | 完整 v3 store 一次复制到节点 `/dev/shm`，8 rank 共享 mmap，全量 manifest shuffle | 新性能/数据生命周期实验 |
| MeSH online | 原始音频在线 decode/resample/crop/pad，完整 manifest shuffle | 速度和初始化后续训对照 |
| MeSH silence-slot | 六分区 + 固定 260，单音频第二槽 GPU 内 zero waveform | 独立槽位实验 |
| SmolLM2 shared-store | 原始 30 层 SmolLM2 + 完整 v3 `/dev/shm` store + 固定 260 双槽 | 30-epoch 同配置对比线 |
| SmolLM2 partition / fixed recursive | 各自文本 backbone 和独立 checkpoint contract | 历史/结构对比基线 |

文本模型默认初始化自第二轮低学习率 MeSH checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244
```

初始化边界必须说清楚：

- MeSH/SmolLM2 文本模型、routers 和 memory 参数：从上述 checkpoint 加载。
- HTSAT：从 AudioSet checkpoint 加载并完全冻结。
- `c2l(527->768)`：随机初始化并训练。
- bridge `768->576->576`：随机初始化并训练；线性层 Xavier uniform，LayerNorm weight=1/bias=0。
- `--mesh-checkpoint` 表示从文本模型开始一次新的音频训练。
- `--resume-from` 只接受本分区音频训练器生成的完整复合 checkpoint，会恢复 LLM、c2l、bridge、optimizer、scheduler、游标和所有 rank RNG，不会重新随机初始化 mapper。

## 1. 工作方式和不可违反的边界

本项目采用“本地代码、远程资产”工作流：

- 本地 Windows 工作区只保存 Python、Shell、测试和文档。
- 模型权重、原始数据、manifest、waveform store、日志和 checkpoint 都在远程 Linux。
- GitHub 只中转代码，不提交音频、权重、parquet、checkpoint 或大日志。
- 本地可做代码修改、`py_compile`、静态测试和 diff 审查；真实 CUDA、HTSAT、数据与内存释放必须以远程 report 为证据。
- 所有 GPU 作业必须通过仓库中的 `vc submit` wrapper；不要在登录节点直接运行 CUDA Python 或 `torchrun`。
- 保留各实验路线隔离。不要把 MeSH、固定 recursive、Parcae、Poisson recursion 或原始 SmolLM2 基线的模型/checkpoint/脚本互相替换。
- 不得因本地测试通过就声称远程 `PASS`。

状态术语：

- **远程 PASS**：用户提供过明确 PASS report、日志或真实 checkpoint。
- **代码就绪**：本地实现和检查通过，但远程尚未确认。
- **历史/对照**：保留用于复现或比较，不是默认主线。

## 2. 环境和远程路径

### 2.1 工作区、环境和资源

```text
本地仓库:
C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge

远程仓库:
/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM

项目代码目录（提交命令通常在这里执行）:
/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

conda 环境:
/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/rsmol

输出根目录:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol
```

已知远程环境：Python 3.10、PyTorch 2.11.0+cu128、Transformers 4.54.1。

当前正式 wrapper 资源：

```text
queue: route-specific; current MeSH text/audio and official Mellow use pdgpu-3090, while selected comparison/evaluation routes use pdgpu-5090
image: docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1
resources: -c 32 -m 256G -g 8 -n 1
```

### 2.2 模型与外部代码

```text
原始 SmolLM2:
/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2

当前文本 MeSH 初始化 checkpoint:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/
formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244

Mellow 官方代码（历史 checkout；当前 vendored 训练目录见上方 2026-10-03 补充）:
/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main

当前 vendored Mellow training:
/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol/mellow_official_training_c8204d8

HTSAT 官方代码:
/hpc_stor03/sjtu_home/jinwei.zhang/code/HTS-Audio-Transformer-main

HTSAT AudioSet checkpoint:
/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt
```

### 2.3 数据、manifest 和当前 partition store

```text
ReasonAQA metadata:
/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/{train,val,test}.json

AudioCaps:
/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2/{train,val,test}

Clotho v2.1:
/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1/{development,validation,evaluation}

ClothoAQA audio:
/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_audio/audio_files

最终 drop12 train manifest:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/
stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl

当前六分区 waveform store:
/hpc_stor03/sjtu_home/jinwei.zhang/data/
rsmol_reasonaqa_train_component_partitions6_32k_10s_f32_v2
```

完整 unique waveform store（shared-store 路线）：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/
rsmol_reasonaqa_train_unique_waveforms_32k_10s_f32_v3
```

v3 store 的固定文件是 `metadata.json`、`index.jsonl` 和 `waveforms.f32`；metadata 必须为 `status=PASS`、`format=manifest_unique_fixed_waveform_store_v1`，并记录 manifest/index/waveform identity。v3 是预处理 waveform，不是 HTSAT feature。shared-store 作业会把它连同 metadata/index 复制到本节点 `/dev/shm`，不会修改 persistent store。

## 3. 当前文本 backbone：5-10x2-5 MeSH

核心实现：

```text
code/RSmol/recursive_model_5_10x2_5_mesh.py
```

结构合同：

- SmolLM2-135M hidden size 576。
- 20 个物理 decoder layers：prefix 5 + shared middle 10 + suffix 5。
- 30 个逻辑执行位置：`5 + 10 + 10 + 5`。
- 中间 10 个物理层固定循环两次；不是 Poisson 动态深度。
- 5 个 memory slots。
- 三个阶段分别有 3 个 write routers 和 3 个 read routers，共 6 个独立线性 router。
- memory 是单次 forward 内的状态，不跨样本、batch 或 optimizer step 持久化。
- 训练 forward 使用 `use_cache=False`。

首个真实 optimizer step 会审计：

- 逻辑层轨迹精确匹配 `5-10-10-5`。
- 两次 middle loop 输入/输出都有有限梯度。
- 6 个 router 都有有限梯度。
- 10 个 shared middle 物理层都获得有限梯度。

PERF20 和当前 partition trainer 为避免 router 统计中的额外 GPU→CPU 同步，关闭逐 forward router 数值统计，但仍保留首步 router 参数梯度与完整逻辑轨迹审计。router/memory 参数本身仍正常训练。

## 4. 当前音频模型与精确序列合同

核心文件：

```text
code/RSmol/audio_5_10x2_5_mesh_mellow/model.py
code/RSmol/audio_5_10x2_5_mesh_mellow/data.py
```

### 4.1 Waveform 和 HTSAT

当前 partition 中保存的不是 HTSAT tokens，而是已经完成以下预处理的 waveform：

- mono；
- 32 kHz；
- 取前 10 秒，过短右侧补零；
- 固定 `[1, 320000]`；
- little-endian float32；
- 每条 waveform 1,280,000 bytes。

训练时 waveform 仍会进入 HTSAT。HTSAT backbone 全部 `requires_grad=False` 且保持 `eval()`；不能把 waveform store 描述成“已经离线算好 HTSAT feature”。

HTSAT/Mellow 路径输出 latent/CLS 等价特征和 527 类 framewise event-presence，Mellow `c2l` 将 527 维映射到 768 维。

### 4.2 Mapper

可训练 mapper 包括：

```text
c2l: Linear(527 -> 768)
bridge:
  Linear(768 -> 576, bias=False)
  + GELU
  + Dropout(0.5)
  + Linear(576 -> 576, bias=False)
  + residual
  + LayerNorm
  + CLS-preserving temporal avgpool8
```

每段音频输出精确 129 tokens：1 个 CLS/global token + 128 个下采样 frame tokens。

### 4.3 单/双音频 prefix

新的 partition 训练路线启用 `compact_single_audio_prefix=True`：

```text
真实单音频样本:
[audio1:129] [separator:1] [prompt] [answer]
prefix = 130

双音频样本:
[audio1:129] [separator:1] [audio2:129] [separator:1] [prompt] [answer]
prefix = 260
```

单槽/双槽按 manifest 的结构字段判断，不仅看两个路径是否相同：

- manifest 缺少第二音频、`audio2_reused=true`：真实单槽，prefix 130。
- manifest 明确存在两个槽，即使路径相同：仍是双槽，prefix 260；允许复用相同 waveform/embedding 计算，但不删除第二槽。

separator 从 tokenizer/model 词表解析，SmolLM2 当前通常使用 `!`，代码不硬编码词表 ID。

### 4.4 Prompt、answer、padding 和 loss

```text
prompt/question: 最多 129 tokens，add_special_tokens=True
answer 正文:     最多 249 tokens，add_special_tokens=False
answer 终止:     手动追加且只追加一个 <|endoftext|>
answer 总长:     最多 250 tokens（包含终止符）
```

代码要求 tokenizer 的 `<|endoftext|>` ID 与 `eos_token_id` 完全相同。若原始 answer 尾部已经含一个或多个 EOS，会先去掉尾部 EOS，再追加一个，保证每条 answer 恰好以一个 EOS 结束。EOS 位于 answer 区间内，`attention=1` 且 label 是真实 token ID，因此参与 causal-LM loss。即使 tokenizer 用 EOS 兼任 pad token，右侧 batch padding 仍为 `attention=0, label=-100`，不会进入 loss。

上下文上限：

```text
单音频最大: 130 + 129 + 250 = 509
双音频最大: 260 + 129 + 250 = 639
模型保护上限: 768
```

每条样本先独立构造真实 `prompt + answer`，之后才做 batch 右侧 padding。单/双音频可自然混在同一个 batch；每条样本先按自己的 130/260 prefix 拼接，再将完整多模态序列右侧补齐。

直接参与 loss 的区域严格如下：

| 区域 | attention | label |
|---|---:|---:|
| audio prefix | 1 | `-100` |
| separator | 1 | `-100` |
| prompt/question | 1 | `-100` |
| answer 正文及末尾 `<|endoftext|>` | 1 | 真实 token ID |
| batch padding | 0 | `-100` |

底层 Hugging Face causal-LM loss 做正常 one-token shift：第一个 answer token 由最后一个 prompt token预测，之后逐 token 自回归。audio 与 prompt 不作为监督目标，但它们是预测 answer 的上下文，因此 answer loss 会经由它们向 mapper 和 LLM 传播梯度；这是预期语义。

### 4.5 冻结和可训练参数

冻结：

- HTSAT backbone 全部参数；
- Mellow wrapper 中除 `c2l` 外的全部参数。

训练：

- `c2l(527->768)`；
- 768→576→576 bridge；
- 完整 MeSH/SmolLM2，包括 embeddings、20 个物理层、routers、memory 参数和 LM head。

optimizer 只接收 `requires_grad=True` 的参数。启动时 `trainable_parameter_audit()` 会硬检查 HTSAT frozen、wrapper 只有 c2l 可训练、c2l/bridge/MeSH 均处于训练模式。

## 5. ReasonAQA manifest 和六分区数据

最终用于分区规划的 train manifest SHA256：

```text
14cd324e4c78ba289c1d6b2ac26ba702f70d86ffc069536327bbd7cbd935c055
```

相关统计：

```text
QA rows:                    968,059
same-audio rows:            764,494（早期路径相同统计；结构单槽以 manifest 字段为准）
distinct two-audio rows:    203,565
unique audio:               54,046
total QA audio incidences:  1,171,624
```

构建方法：

1. 统计双音频关联图。
2. 对音频图做 union-find，完整连通分量不能拆开。
3. 将连通分量装箱到 6 个 partition。
4. 每条 QA 只属于一个 partition；其引用的所有音频都在本区。
5. 每个唯一音频只物化一次，跨 partition 复制为 0。
6. 从原唯一 waveform store 顺序复制成六个自包含 fixed-stride store。
7. 校验 manifest/index/waveform SHA256、payload 总字节数和抽样字节一致性。

远程 v2 物化结果：

```text
status: PASS
duplicated_audio: 0
total waveform: 64.42785263061523 GiB
source_payload_sha256_reverified: true

partition 0: 278,680 QA, 3,839 audio, 约 4.576 GiB
partition 1-5: 各约 137,876 QA、约 10,040 audio、约 11.97 GiB
```

partition 0 对应不可拆分的大型 Clotho 连通分量，因此在零复制条件下六份 QA 数量不可能严格均衡；这是已知且由用户接受的结果，不是物化错误。

规划与物化脚本保留用于审计/复现：

```text
code/RSmol/scripts/prepare_unique_audio_waveform_store.py
code/RSmol/scripts/plan_reasonaqa_component_partitions.py
code/RSmol/scripts/materialize_reasonaqa_component_partitions.py
```

### 5.1 三种数据读取链路（必须区分）

| 路线 | 完整 waveform 的位置 | rank 如何取样 | 当前用途 |
|---|---|---|---|
| online | 远程/本地文件系统的原始 wav | 完整 manifest 上的 `DistributedSampler` | 速度对照、初始化后在线训练 |
| six-partition | 每次加载一个 partition；每个 rank clone 到自己的匿名 CPU RAM | partition schedule + rank-local row slice | 当前默认正式训练 |
| shared-store | 完整 v3 store 一次复制到本节点 `/dev/shm`；8 rank mmap 同一文件 inode | 完整 manifest 上的 `DistributedSampler(shuffle=True)` | 新的共享内存/全量 shuffle 实验 |

shared-store 的完整链路是：

```text
远程 persistent v3 store + canonical manifest
  -> 作业启动时复制到 /dev/shm/rsmol_shared_train_<RUN_ID>
  -> metadata/index/manifest SHA256、格式、文件大小审计
  -> 8 rank mmap 同一个 waveforms.f32 inode
  -> 每个 epoch set_epoch，完整 manifest 全局打乱并分发互斥 row indices
  -> 每个 rank 当前 microbatch 读取 8 条 waveform，stack 成小型 CPU batch
  -> CPU batch 搬到本 rank GPU，进入 HTSAT/c2l/bridge/文本模型
```

这里“共享一份”指完整 `waveforms.f32` 的底层 `/dev/shm` 文件和 mmap inode 共享；每个 rank 仍有自己的 manifest/index Python 对象、DataLoader、当前 CPU batch 和 GPU tensor。shared-store 固定 `num_workers=0`，不会有 DataLoader worker 预取历史 batch；当前 microbatch 结束后 waveform batch 失去引用并由 allocator 复用。allocator 不保证 RSS 立刻下降，但不应随 step 线性增长。GA=4 累积的是 GPU 参数梯度，不是四份 CPU waveform。

## 6. 当前正式训练合同：MeSH 六分区默认主线

默认六分区正式入口：

```text
code/RSmol/scripts/train_audio_partitioned_5_10x2_5_mesh_mellow_ddp.py
code/RSmol/scripts/train_audio_partition_formal_5_10x2_5_mesh_mellow_ddp.sh
code/RSmol/run_audio_partition_formal_5_10x2_5_mesh_mellow_5090.sh
```

六分区训练器是默认主线。`train_audio_5_10x2_5_mesh_mellow_ddp.py --gate FORMAL` 是独立的 online full-shuffle 对照入口；它与六分区的 store、数据生命周期、smoke gate、checkpoint contract 不同，不要混用。

### 6.1 Batch、优化器与调度器

```text
world size:                  8
micro batch per GPU:         8
gradient accumulation:       4
effective global batch:      8 * 8 * 4 = 256
precision:                   BF16 autocast
optimizer:                   AdamW
betas:                       (0.9, 0.95)
weight decay:                0.1
max LR:                      1e-3
min LR:                      0
warmup:                      ceil(total optimizer steps * 5%)
after warmup:                cosine decay to min LR
gradient clipping:           0.5, error_if_nonfinite=True
DDP broadcast_buffers:       False
DDP find_unused_parameters:  False
NCCL timeout:                30 minutes
NCCL async error handling:   enabled in shell wrapper
```

当前计划的 10 epochs：

```text
optimizer steps/epoch:  floor(968059 / 256) = 3781
total optimizer steps:  3781 * 10 = 37810
warmup steps:           ceil(37810 * 0.05) = 1891
```

如果显式传 `--warmup-steps`，必须精确等于上述公式，否则硬失败。

当前 loss reduction 需要准确描述：每个 rank 的每个 microbatch 由 HF 对所有非 `-100` answer tokens 求 mean，再除以 GA=4 反传；DDP 对 rank 梯度平均。因此目标是各 microbatch token-mean 的等权组合，不是把整个 256 样本的全部 answer token 先求总和再除以全局 token 数。除非用户明确决定改变目标函数，不要擅自重写 reduction。

### 6.2 Epoch、partition 顺序和分区内 shuffle

每 epoch 固定 3781 optimizer steps：

- partition 0：`floor(278680/256)=1088` 步，即 278,528 条；shuffle 后丢弃 152 条，不重复。
- partition 1–5：合计 2693 步；用累计平衡分配 538/539 步。
- 如果某个小 partition 的配额比实际 QA 多，缺额只从该 partition 本 epoch 已打乱序列的开头补齐；绝不跨 partition。
- 每个 partition 内所有 QA 统一 shuffle；单槽和双槽自然混合，不分池。
- 每个 epoch 重新生成确定性 partition 顺序和分区内 shuffle。
- partition 0 只能出现在位置 0–4，绝不放在最后。
- shuffle plan 会计算 SHA256；中途 resume 必须重新生成相同 plan 并匹配 hash。

默认 `seed=0` 时前三个 epoch 的分区顺序：

```text
epoch 1: [2, 1, 4, 3, 0, 5]
epoch 2: [0, 2, 1, 5, 4, 3]
epoch 3: [4, 5, 3, 0, 2, 1]
```

### 6.3 一个 partition 的完整生命周期

1. 验证根 materialization report 和当前 partition metadata。
2. 验证 `PASS`、零复制、ID、manifest/index SHA256、waveform 文件字节数。
3. 每个 rank 打开当前 partition store。
4. 每个 rank 按本地 audio ID 顺序将所有 waveform `.clone()` 到本进程匿名 CPU RAM。
5. 检查 resident audio 数、总字节、miss 数且 eviction=0。
6. 8 rank barrier；打印 `[partition-train] loaded ...`。
7. 构造该 epoch/partition 的确定性全局 QA shuffle 和 optimizer-window plan。
8. 每个 global step 的 256 条按 rank 切成 32 条；每 rank 再运行 4 个 microbatch，每个 8 条。
9. BF16 forward、answer-only loss、GA backward、clip、AdamW、scheduler。
10. 分区训练期间 waveform cache 的 miss/clone/resident 数必须不变，即不再读 store。
11. 分区结束 CUDA synchronize，删除 batch/plan/cache/dataset/tensor 引用，关闭 mmap。
12. 执行 `gc.collect()` 和 Linux `malloc_trim(0)`。
13. barrier 后最多轮询 120 秒。
14. 每 rank RSS 至少下降 `partition waveform bytes * 70%`。
15. cgroup anon 至少下降 `partition waveform bytes * 8 * 70%`。
16. 任一释放检查失败即硬失败，绝不加载下一 partition。

### 6.4 日志

rank 0 在以下时机打印 step 日志：

- 全局 step 是 10 的倍数；
- 当前 partition 最后一步（即使不是 10 的倍数）。

格式：

```text
[partition-train] step=180/37810 p2 180/538 loss=1.234567 lr=0.000095238 1.993s
```

其中：

- `loss` 是 rank 0 上当前 optimizer step 四个 microbatch loss 的算术平均，不是 8 rank 聚合的全局 loss。
- `lr` 是该 optimizer step 在 `optimizer.step()` 时实际使用的 LR。
- JSON report 仍保存每一步的 loss、LR、耗时和单/双/混合槽型标记。
- partition 加载完成和释放完成另有日志；完整内存/cgroup 数据写入 `partition_training_report.json`。

### 6.5 Checkpoint、保留与 resume

```text
save_every:           500 optimizer steps
checkpoint_retention: 4
final step:           always saved
```

checkpoint 使用临时目录完整写入并检查 required files 后原子发布。保存内容：

- 完整 MeSH/SmolLM2；
- c2l 与 bridge；
- optimizer、scheduler；
- global step 和 segment/partition-step 游标；
- 当前 plan hash；
- 8 个 rank 的 Python/Torch/CUDA RNG；
- partition inventory、hash、完整 schedule；
- prefix、batch、LR、warmup、timeout、释放阈值；
- HTSAT/Mellow 路径与 Mellow 源码 SHA256。

可在 partition 中间 resume。resume 必须使用新的空 output dir，并保持 epochs、seed、batch、LR、保存策略、timeout、释放阈值、partition inventory 和 Mellow provenance 完全一致。旧固定-260 prefix 音频 checkpoint、旧 Stage 7 checkpoint、文本 checkpoint 均不能作为本训练器的 `--resume-from`。

### 6.6 正式训练前 smoke gate

Smoke 入口：

```text
code/RSmol/scripts/train_audio_partitioned_5_10x2_5_mesh_mellow_ddp.py --mode smoke
code/RSmol/run_audio_partition_smoke20_5_10x2_5_mesh_mellow_5090.sh
```

固定 smoke schedule：

```text
p2: preload -> train 10 -> strict release
p0: preload -> train 10 -> strict release
save checkpoint-000020

new job/output dir, resume checkpoint-000020
p1: preload -> train 2 -> strict release
save checkpoint-000022
```

旧 v1 的两次 smoke 已由用户确认远程通过，但它们没有 EOS 标签，不能用于当前 v2 正式 gate。必须用当前代码重新运行上述两次 smoke，并从新的输出目录传入 v2 report：

```text
--smoke20-report .../partition_training_report.json
--smoke-resume-report .../partition_training_report.json
```

正式 gate 会核对：`training_contract=component_partitions6_rank_ram_compact_audio_answer_eos_v2`、status、mode、inventory、seed、0→20 与 20→22 游标、segment 步数、30 层轨迹、每段 8-rank RSS/cgroup release，以及 resume report 是否确实指向第一次的 checkpoint-000020。

### 6.7 10 epochs 正式提交模板

必须将 smoke report 替换为远程真实 PASS 路径：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

SMOKE20_REPORT=/actual/eos_v2_smoke20/output/partition_training_report.json
SMOKE_RESUME_REPORT=/actual/eos_v2_resume2/output/partition_training_report.json
FORMAL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/partition_formal_eos_v2_10epochs_20260918

bash run_audio_partition_formal_5_10x2_5_mesh_mellow_5090.sh \
  --output-dir "$FORMAL_DIR" \
  --epochs 10 \
  --smoke20-report "$SMOKE20_REPORT" \
  --smoke-resume-report "$SMOKE_RESUME_REPORT"
```

默认 seed 为 0。如果 smoke 使用了其他 seed，正式命令必须传同一 `--seed`。

### 6.8 从六分区 checkpoint 仅初始化权重的在线训练对照

在线入口新增 `--init-from-audio-checkpoint`，它与 `--resume-from` 互斥。初始化源必须是完整的 `component_partitions6_rank_ram_compact_audio_answer_eos_v2` checkpoint；入口严格核对 completion marker、MeSH/mapper、130/260 prefix、answer EOS、HTSAT/Mellow 路径与 provenance、checkpoint 目录名/marker/training-state/cursor 的 global step 一致性。

该模式只加载 `mesh_model`、bridge 和 c2l；HTSAT 仍从当前命令指定的外部 checkpoint 加载并冻结，不加载源 checkpoint 的 optimizer、scheduler、RNG 或训练游标，新训练从 global step 0 开始。数据仍由 `ReasonAQADataset` 在线解码、mono、重采样到 32 kHz、截取前 10 秒并右侧补零；`DistributedSampler(shuffle=True)` 每个 epoch 对完整 manifest 全局重排，再为各 rank 分配互斥样本，不使用六分区 store。

FORMAL 的 8 GPU、microbatch 8、GA 4、save every 500 和 retention 4 仍固定；`--epochs`、`--max-lr`、`--min-lr` 现在是显式可传超参数，warmup 强制为 `ceil(actual_total_optimizer_steps * 0.05)`。这一入口生成的新 checkpoint 合同为 `online_reasonaqa_full_shuffle_compact_audio_answer_eos_v2`，可以用 `--resume-from` 完整恢复，但旧在线 checkpoint 不能冒充新合同 resume。

当前两轮对照训练的目标初始化源是：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/
partition_formal_answer_eos_v2_10epochs_20260918/checkpoint-037810
```

提交命令：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

INIT_CHECKPOINT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/partition_formal_answer_eos_v2_10epochs_20260918/checkpoint-037810
TRAIN_MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl
OUTPUT_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/online_fullshuffle_init037810_2epochs_lr2e-4_2e-5_20260920

bash run_audio_formal_5_10x2_5_mesh_mellow_5090.sh \
  --init-from-audio-checkpoint "$INIT_CHECKPOINT" \
  --train-manifest "$TRAIN_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 2 \
  --max-lr 2e-4 \
  --min-lr 2e-5
```

在当前 968,059 条 manifest、8×8×GA4 配置下，预期每 epoch 3,781 个 optimizer steps，两轮共 7,562 steps，warmup 自动计算为 379 steps。不要传 `--resume-from`，也无需手动传 `--warmup-steps`。

### 6.9 Shared-store 全量 shuffle 隔离训练线

当前实验直接复用这条独立路线，合同为 `node_shared_unique_store_fullshuffle_fixed260_audio_reuse_answer_eos_v2`。恢复旧 3-epoch 的固定 260-token 双槽，单音频复用 HTSAT embedding 后分别调用同一个 bridge（保留两次 dropout），并保留 answer-EOS 监督。完整 v3 unique store 每个作业只复制一次到节点 `/dev/shm`，8 个 rank mmap 同一个 `waveforms.f32` inode；不使用六分区 schedule，不给每个 rank 克隆完整 store，不重复解码/重采样/crop/pad。

实现和完整合同：

```text
code/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/README.md
code/RSmol/scripts/stage_audio_shared_store_5_10x2_5_mesh_mellow.sh
code/RSmol/scripts/train_audio_shared_store_5_10x2_5_mesh_mellow_ddp.py
code/RSmol/run_audio_shared_store_smoke20_5_10x2_5_mesh_mellow_5090.sh
code/RSmol/run_audio_shared_store_resume2_5_10x2_5_mesh_mellow_5090.sh
code/RSmol/run_audio_shared_store_formal_5_10x2_5_mesh_mellow_5090.sh
```

入口：

```text
code/RSmol/run_audio_shared_store_smoke20_5_10x2_5_mesh_mellow_5090.sh
code/RSmol/run_audio_shared_store_resume2_5_10x2_5_mesh_mellow_5090.sh
code/RSmol/run_audio_shared_store_formal_5_10x2_5_mesh_mellow_5090.sh
```

smoke 初始运行固定 step 0→20，发布 `checkpoint-000020`；resume 必须从 `epoch=0,batch_in_epoch=80,global_step=20` 开始，恢复 optimizer/scheduler/RNG，运行 20→22 并发布 `checkpoint-000022`。formal gate 检查两个 report 的合同、store identity、seed、初始化源、调度、MeSH 轨迹/梯度审计和 answer-only/EOS mask。三种运行均固定 3 epochs 调度、max LR `1e-3`、min LR `1e-4`、5% warmup 向上取整；当前 968,059 条数据对应 3,781 step/epoch、11,343 总步数、568 warmup steps。formal 从文本 MeSH 初始化重新训练，每 500 step 保存、保留 4 个 checkpoint。

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
SHARED=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow_shared_store

bash run_audio_shared_store_smoke20_5_10x2_5_mesh_mellow_5090.sh \
  --output-dir "$SHARED/smoke20_YYYYMMDD"

bash run_audio_shared_store_resume2_5_10x2_5_mesh_mellow_5090.sh \
  --resume-from "$SHARED/smoke20_YYYYMMDD/checkpoint-000020" \
  --output-dir "$SHARED/resume2_YYYYMMDD"

bash run_audio_shared_store_formal_5_10x2_5_mesh_mellow_5090.sh \
  --smoke20-report "$SHARED/smoke20_YYYYMMDD/shared_store_training_report.json" \
  --smoke-resume-report "$SHARED/resume2_YYYYMMDD/shared_store_training_report.json" \
  --output-dir "$SHARED/formal_3epochs_YYYYMMDD"
```

本路线本地已通过 Python 编译和独立静态合同测试；用户确认旧 compact 共享路线 smoke/resume 跑通，本次 fixed260 版本须用新输出目录重新运行，formal 从头初始化，不能延续旧 10-epoch 音频权重。staging 时间单独写入 report，不计入 optimizer step timing；退出时只删除本次精确的 `/dev/shm/rsmol_shared_train_<RUN_ID>` 目录。

## 7. 性能优化实验：保留的结论

详细历史 report 位于远程输出目录和旧聊天，本文只保留对当前设计有用的因果结论。

### 7.1 为什么不继续在线读取

原始在线读取具有严重且随机的 rank 长尾：data/forward/backward 的慢 rank 在不同 step/rank 间漂移，整个 DDP step 被最慢 rank 拖住。DataLoader `num_workers=2/rank`、64 CPU 核没有改善，反而受共享存储并发与调度影响。

四组 warm/cold 对照表明，主要损失来自 waveform 及时供应，不是文本 tokenization/collate：

- 在线路径稳态约 18.9 秒/step。
- waveform/full preload 可到约 1.6–1.7 秒/step。
- waveform preload 与 full preload 仅约 0.7 秒差异，优先解决 waveform I/O 是正确方向。

### 7.2 被放弃或仅保留复现的方案

- 64 个随机 waveform shard + mmap：约 27.5 秒/step，跨 shard 随机访问/page fault 很差；不用于当前正式训练。
- 一个 64.4 GiB 共享 mmap + rank0 page-cache warm：仍产生大量 major/minor faults与 rank 长尾；不用于正式训练。
- 小容量在线 LRU/prefetch：音频命中率低且 producer/consumer 争用，曾约 60 秒/step；不用于正式训练。
- 完整全量 waveform 每 rank 预加载：训练快但 8 rank 内存总量不可接受。
- 完整 v3 store 暂存到节点 `/dev/shm`、8 rank 共享同一 mmap inode 的 shared-store PERF20 已通过；这证明链路可运行，但不等于正式训练已 PASS，也不等于每 rank 有匿名 RAM 副本。

该 PERF20 report 的关键解读：初始化/首次访问开销集中在前几步，稳态 step time 约 0.537 s/step，rank 间平衡良好；它测量的是共享 store 输入链路，不包含正式 trainer 的 20+2 gate、长程 checkpoint 或 10-epoch 训练证明。

### 7.3 当前方案的依据

双音频关联图的完整连通分量允许把全部 64.4 GiB waveform 零复制切成六份。每个 rank 同时只持有一个 partition：

- p0 约 4.576 GiB/rank；
- 最大 partition 约 11.97 GiB/rank；
- 8 rank waveform 总驻留最大约 95.8 GiB，加模型/进程/caches 后在 256 GiB cgroup 中已通过 smoke。

partition PERF20 观测：第一步包含初始化开销，第二步下降，第三步起进入约 1.x 秒稳态。p0/p2 都证明完整分区加载后训练速度接近全量 preload 下限，因此转为当前正式生命周期设计。

### 7.4 PERF20 仍可用于诊断

历史性能入口仍保留：

```text
code/RSmol/scripts/train_audio_5_10x2_5_mesh_mellow_ddp.py --gate PERF20
code/RSmol/scripts/train_audio_perf20_5_10x2_5_mesh_mellow_ddp.sh
code/RSmol/run_audio_perf20_5_10x2_5_mesh_mellow_5090.sh
```

它支持 online、warm、waveform/full preload、shared store、rank RAM preload/prefetch、partition rank RAM 等隔离模式及 torch.profiler。它是诊断工具，不是当前正式训练入口。

## 8. 评测与生成

### 8.1 ReasonAQA samples generation 与 router CSV

MeSH 音频 checkpoint 的 ReasonAQA generation 入口：

```text
code/RSmol/scripts/generate_audio_checkpoint_reasonaqa.py
code/RSmol/scripts/generate_audio_checkpoint_reasonaqa.sh
code/RSmol/run_audio_checkpoint_reasonaqa_generation_3090.sh
```

支持默认前五个或 `--sample-indices` 指定 zero-based manifest 行号，并可导出所有 token 位置的 5 个 memory slots × 3 组 × write/read router 权重到 CSV。该入口现已兼容 partition-v2 checkpoint：严格核对 `component_partitions6_rank_ram_compact_audio_answer_eos_v2`、completion marker、130/260 prefix 和 answer-EOS 合同；结构单音频使用 `audio1 + separator1` 的 130-token prefix，显式双音频使用 260-token prefix，router CSV 也按每条样本的真实 prefix 标注 token region。当前默认 checkpoint 为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/partition_formal_answer_eos_v2_10epochs_20260918/checkpoint-037810
```

### 8.2 MMAU test mini

MeSH 音频 MMAU test mini 已升级到当前 partition-v2 checkpoint，并调用官方 `evaluation.py --input ...`。入口在运行前严格审计 `MMAU-v05.15.25` 的 1000 条 metadata、ID/任务/难度分布、metadata 与 scorer SHA256；模型侧严格核对 compact 130-token 单音频 prefix、768 context、checkpoint completion marker 与 Mellow/HTSAT provenance。正式结果固定保留官方 1000 条分母：完整遍历后的逐样本 skip 会写入空 `model_output` 并由官方 scorer 计错，同时在 report 中保留原因；只有 metadata/parquet 未完整遍历、终态不足 1000 条或其他全局故障才会阻止官方计分。

answer-EOS v2 模型的正常输出是 `c) It is plausible<|endoftext|>`；解码后的 `generated_text` 是 `c) It is plausible`。MMAU/MMAR prompt 已与 ReasonAQA 对齐为 `问题 a) ... b) ... c) ...`，不添加 `Choices:`；MMAR 的五、六选项自动使用 `e)`、`f)`。官方 scorer 使用词集合匹配而不是稳定的选择题字母解析，因此当前协议只在输出开头精确移除 `a)`/`b)`/`c)`/`d)`，再将剩余文本写入 MMAU 的 `model_output` 或 MMAR 自动探测出的官方预测字段；不猜选项、不改正文。原始 `generated_text` 与带特殊 token 的 `generated_text_raw` 均保存在 append-only 审计 JSONL 中。最大生成长度统一为 32 tokens。MMAU 的 parquet/metadata 选项核对只在 parquet 一侧按当前位置移除显式标签，避免把 `F. Scott Fitzgerald`、`J.D. Salinger`、`E-guitar`、`E-bass`、`B:maj/1` 等正文误判为选项标签。

当前 MeSH 文件：

```text
code/RSmol/scripts/evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.py
code/RSmol/scripts/evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.sh
code/RSmol/run_mmau_test_mini_5_10x2_5_mesh_mellow_5090.sh
```

默认 checkpoint 是本节 8.1 的 `checkpoint-037810`，队列为 `pdgpu-5090`。smoke 与正式评测必须复用同一 output dir；第二次提交会从前 5 条继续，绝不重复推理已完成样本：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
EVAL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/mmau_test_mini_checkpoint_037810_strip_label_max32_v2

RSMOL_MMAU_MODE=smoke RSMOL_MMAU_OUTPUT_DIR="$EVAL_DIR" \
  bash run_mmau_test_mini_5_10x2_5_mesh_mellow_5090.sh

# smoke PASS 后，以同一目录续跑全部 1000 条并执行官方计分
RSMOL_MMAU_MODE=full RSMOL_MMAU_OUTPUT_DIR="$EVAL_DIR" \
  bash run_mmau_test_mini_5_10x2_5_mesh_mellow_5090.sh
```

### 8.3 MMAR

MMAR 使用 Hugging Face 下载的 `MMAR-meta.json`、已解压的 `mmar-audio/audio/*.wav` 和下载包内官方 `code/evaluation.py`。HF JSON 与 GitHub JSONL 在顺序、canonical artifact 和预测字段上可能不同；入口会审计官方 ID 集合、modality/category 分布、音频存在性、scorer AST 语义和当前实际 output key，并把预测写成 scorer 可直接接受的 schema。不能手工把 `model_prediction` 或 `answer_prediction` 当作固定字段。音频统一为 32 kHz，短音频补零、长音频取开头 10 秒；选择顺序不打乱。输出保持完整记录，由官方 scorer 原样计分；逐样本 skip 保留在完整官方分母中，只有全局 pipeline failure 才阻断 scorer。

```text
code/RSmol/scripts/evaluate_mmar_5_10x2_5_mesh_mellow.py
code/RSmol/scripts/evaluate_mmar_5_10x2_5_mesh_mellow.sh
code/RSmol/run_mmar_5_10x2_5_mesh_mellow_5090.sh
```

同样使用 `pdgpu-5090`，smoke→full 复用同一个目录完成续跑测试和正式评测：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
EVAL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/mmar_checkpoint_037810_strip_label_max32_v2

RSMOL_MMAR_MODE=smoke RSMOL_MMAR_OUTPUT_DIR="$EVAL_DIR" \
  bash run_mmar_5_10x2_5_mesh_mellow_5090.sh

# smoke PASS 后，以同一目录续跑全部 1000 条并执行官方计分
RSMOL_MMAR_MODE=full RSMOL_MMAR_OUTPUT_DIR="$EVAL_DIR" \
  bash run_mmar_5_10x2_5_mesh_mellow_5090.sh
```

两套评测均写出 `run_config.json`、`progress.jsonl`、`raw_generations.jsonl`、`skipped.jsonl`、`smoke_first5.jsonl`、`official_evaluation.txt` 与 `evaluation_report.json`。MMAU 的官方输入是 `predictions_fixed_order.json`；MMAR 的官方输入是动态字段名的 `predictions_official.json`。若 full 在 scorer 前发生全局故障，代码会覆盖 smoke 留下的 `official_evaluation.txt` 并写入阻断原因，不再保留容易误认成 full 分数的 5 条旧日志。新会话仍应先用以下命令核实文件没有改名：

```bash
rg --files code/RSmol | rg 'mmau|MMAU|mmar|MMAR'
```

### 8.4 原始 SmolLM2 音频基线

当前 partition-v2 训练入口：

```text
code/RSmol/scripts/train_audio_partitioned_smollm2_135m_mellow_ddp.py
code/RSmol/run_audio_smollm2_135m_mellow_smoke20_3090.sh
code/RSmol/run_audio_smollm2_135m_mellow_resume2_3090.sh
code/RSmol/run_audio_smollm2_135m_mellow_formal_3090.sh
```

训练合同为 `smollm2_component_partitions6_rank_ram_compact_audio_answer_eos_v2`。除文本 backbone 是原始 30 个独立物理层的 SmolLM2-135M、没有 MeSH router/memory 外，六分区数据生命周期、130/260 compact prefix、answer EOS、batch/优化器/LR、20+2 smoke、严格内存释放、checkpoint/resume 和 10-epoch 正式配置均与当前 MeSH 主线一致。正式训练同样必须由本路线两个真实 PASS smoke report 放行。

新增的 shared-store configurable-epochs 线与上述 partition-v2 线完全隔离。它从
`/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2` 的标准 30 层模型初始化，复用当前
MeSH shared-store 的完整 v3 `/dev/shm` store、全量 manifest shuffle、8 GPU × microbatch 8 ×
GA 4、AdamW、`1e-3 -> 1e-4` cosine 和 save-500 配置，但固定使用 260-token 双槽：单音频只算
一次 audio1 HTSAT embedding 再分别通过两次 bridge，双音频分别计算 HTSAT embedding。30 epochs
对应 113,430 个 optimizer steps，warmup 严格为 `ceil(113430 * 0.05) = 5672`。独立合同为
`smollm2_node_shared_unique_store_fullshuffle_fixed260_audio_reuse_answer_eos_v2`，没有 MeSH
router/memory、递归或共享层，不能与 partition、MeSH 或 recursive checkpoint/report 混用。

```text
code/RSmol/audio_smollm2_135m_mellow_shared_store/README.md
code/RSmol/scripts/train_audio_smollm2_shared_store_135m_mellow_ddp.py
code/RSmol/scripts/train_audio_smollm2_shared_store_configurable_epochs_135m_mellow_ddp.py
code/RSmol/run_audio_smollm2_shared_store_smoke20_configurable_epochs_135m_mellow_5090.sh
code/RSmol/run_audio_smollm2_shared_store_resume2_configurable_epochs_135m_mellow_5090.sh
code/RSmol/run_audio_smollm2_shared_store_formal_configurable_epochs_135m_mellow_5090.sh
```

这条新线当前是“本地代码就绪”，远程 smoke20、resume2 和 30-epoch formal 均尚未登记 PASS。

当前 MMAU/MMAR 对比评测 checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/
partition_formal_eos_v2_10epochs_20260918/checkpoint-037810
```

基线评测与 MeSH 评测共享同一套官方数据审计、ReasonAQA 小写选项 prompt、开头 `a)`-`d)` 去标签、32-token 生成上限、完整官方分母和 append-only resume 协议；原始生成文本仍完整留在审计记录中。模型后端独立审计标准 30 层 SmolLM2 partition-v2 artifact，并构造 130-token 单音频 compact prefix。两个提交入口均使用 `pdgpu-5090`：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol

MMAU_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/mmau_test_mini_checkpoint_037810_strip_label_max32_v2
RSMOL_MMAU_MODE=smoke RSMOL_MMAU_SMOLLM2_OUTPUT_DIR="$MMAU_DIR" \
  bash run_mmau_test_mini_audio_smollm2_5090.sh
# smoke PASS 后复用同一目录续跑 1000 条
RSMOL_MMAU_MODE=full RSMOL_MMAU_SMOLLM2_OUTPUT_DIR="$MMAU_DIR" \
  bash run_mmau_test_mini_audio_smollm2_5090.sh

MMAR_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/mmar_checkpoint_037810_strip_label_max32_v2
RSMOL_MMAR_MODE=smoke RSMOL_MMAR_SMOLLM2_OUTPUT_DIR="$MMAR_DIR" \
  bash run_mmar_audio_smollm2_5090.sh
# smoke PASS 后复用同一目录续跑 1000 条
RSMOL_MMAR_MODE=full RSMOL_MMAR_SMOLLM2_OUTPUT_DIR="$MMAR_DIR" \
  bash run_mmar_audio_smollm2_5090.sh
```

旧的三 epoch、固定 260-prefix checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/
formal_20260911_v1/checkpoint-011343
```

该 checkpoint 只用于历史 generation，不能作为新 partition-v2 trainer 的 `--resume-from`，也不能交给当前 partition-v2 MMAU/MMAR evaluator：

```text
code/RSmol/scripts/generate_audio_smollm2_checkpoint_reasonaqa.py
code/RSmol/run_audio_smollm2_checkpoint_reasonaqa_generation_3090.sh
```

## 9. 历史与对照路线

以下内容保留为必要背景，但不是当前默认工作目标。

| 路线 | 定义 | 当前用途 |
|---|---|---|
| 原始 SmolLM2-135M | Hugging Face 原模型 | 音频基线与转换来源 |
| 15R recursive | 早期循环层实验 | 历史 |
| 固定 5-10-5 | 20 物理层，中间 10 层循环两次，无 MeSH | 独立音频对照 |
| 5-10-5 linear | 非递归线性对照 | 历史 |
| 5-10x7-5 | 中间层固定运行 7 次 | 历史 |
| 5-10xr-5 | 直接 recursion，可用 Poisson 深度 | 历史动态深度对照 |
| 5-10xpoisson-parcae | Parcae 注入递归 | 已完成历史路线 |
| 5-10x2-5 MeSH | 20 物理/30 逻辑、5 memory slots | 当前文本 backbone |
| Audio SmolLM2 | 不循环文本模型 + 同类音频 mapper | 当前对比基线 |

关键隔离规则：

- Parcae 的 `A_bar(h)+B_bar(u)`、特殊 `h0` 和 Poisson 深度不能带入 MeSH。
- 直接 recursion 的 `h0=e` 不能冒充 Parcae。
- 当前 MeSH 音频路线必须使用 `recursive_model_5_10x2_5_mesh.py`。
- 固定 5-10-5 音频对照使用 `recursive_model_5_10_5.py`，无 router/memory。
- 旧 `audio_5_10_5_mellow` 名称主要承载 manifest/HTSAT preflight，不代表当前 MeSH 正式 backbone 是固定 5-10-5。

固定 5-10-5 音频对照曾从以下 checkpoint 初始化并继续训练：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/
formal-epoch2-continue-20260902_184936/checkpoint-step-009244
```

当前该路线使用新的 partition-v2 trainer，但仍从上述文本 checkpoint 初始化：

```text
code/RSmol/scripts/train_audio_partitioned_5_10_5_recursive_mellow_ddp.py
code/RSmol/run_audio_5_10_5_recursive_mellow_smoke20_5090.sh
code/RSmol/run_audio_5_10_5_recursive_mellow_resume2_5090.sh
code/RSmol/run_audio_5_10_5_recursive_mellow_formal_5090.sh
```

训练合同为 `recursive_5_10_5_component_partitions6_rank_ram_compact_audio_answer_eos_v2`。它必须先完成本路线自己的 EOS-v2 20+2 smoke；旧三轮 composite checkpoint、MeSH/SmolLM2 smoke report 或文本初始化 checkpoint 都不能放行正式训练或作为新训练器的 resume source。

Parcae 历史完成 checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
stage4_5_10xpoisson_parcae_lr8e-4_mb2_ga64_20260903_172047/checkpoint-009244
```

Parcae 的历史运维合同仍被静态测试引用，保留以下最小信息：提交入口是 `run_stage4_5_10xpoisson_parcae_3090.sh`；转换时需要显式设置类似 `RSMOL_5_10XPOISSON_PARCAE_SOURCE_CHECKPOINT=/hpc_stor03/...` 的来源变量；Shell 批量审计范围包括 `scripts/*.sh`。正式训练 report 每个 optimizer step 保存 compact scalar metric per optimizer step，并用 `total_grad_norm_finite_nonzero` 记录梯度范数有限且非零。这些字段属于历史 Parcae 合同，不应复制进当前 partition 音频训练。

## 10. 当前主线文件地图

### 10.1 必读实现

```text
README.md
code/RSmol/recursive_model_5_10x2_5_mesh.py
code/RSmol/audio_5_10x2_5_mesh_mellow/model.py
code/RSmol/audio_5_10x2_5_mesh_mellow/data.py
code/RSmol/audio_5_10x2_5_mesh_mellow_silence_slot/README.md
code/RSmol/audio_5_10x2_5_mesh_mellow_silence_slot/data.py
code/RSmol/audio_5_10x2_5_mesh_mellow_silence_slot/model.py
code/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/README.md
code/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/data.py
code/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/model.py
code/RSmol/scripts/train_audio_partitioned_5_10x2_5_mesh_mellow_ddp.py
code/RSmol/scripts/train_audio_partition_smoke20_5_10x2_5_mesh_mellow_ddp.sh
code/RSmol/scripts/train_audio_partition_formal_5_10x2_5_mesh_mellow_ddp.sh
code/RSmol/run_audio_partition_smoke20_5_10x2_5_mesh_mellow_5090.sh
code/RSmol/run_audio_partition_formal_5_10x2_5_mesh_mellow_5090.sh
tests/test_audio_partition_training_static.py
tests/test_audio_5_10x2_5_mesh_mellow_static.py
```

### 10.2 数据和性能工具

```text
code/RSmol/scripts/prepare_unique_audio_waveform_store.py
code/RSmol/scripts/plan_reasonaqa_component_partitions.py
code/RSmol/scripts/materialize_reasonaqa_component_partitions.py
code/RSmol/scripts/train_audio_5_10x2_5_mesh_mellow_ddp.py
code/RSmol/scripts/stage_audio_shared_store_5_10x2_5_mesh_mellow.sh
code/RSmol/scripts/train_audio_shared_store_5_10x2_5_mesh_mellow_ddp.py
code/RSmol/scripts/stage_audio_shared_store_configurable_epochs_5_10x2_5_mesh_mellow.sh
code/RSmol/scripts/train_audio_shared_store_configurable_epochs_5_10x2_5_mesh_mellow_ddp.py
code/RSmol/scripts/train_audio_shared_store_smoke20_configurable_epochs_5_10x2_5_mesh_mellow_ddp.sh
code/RSmol/scripts/train_audio_shared_store_resume2_configurable_epochs_5_10x2_5_mesh_mellow_ddp.sh
code/RSmol/scripts/train_audio_shared_store_formal_configurable_epochs_5_10x2_5_mesh_mellow_ddp.sh
code/RSmol/run_audio_shared_store_smoke20_configurable_epochs_5_10x2_5_mesh_mellow_5090.sh
code/RSmol/run_audio_shared_store_resume2_configurable_epochs_5_10x2_5_mesh_mellow_5090.sh
code/RSmol/run_audio_shared_store_formal_configurable_epochs_5_10x2_5_mesh_mellow_5090.sh
code/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/README.md
code/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/data.py
code/RSmol/audio_5_10x2_5_mesh_mellow_shared_store/model.py
code/RSmol/scripts/stage_audio_smollm2_shared_store_configurable_epochs_135m_mellow.sh
code/RSmol/scripts/train_audio_smollm2_shared_store_135m_mellow_ddp.py
code/RSmol/scripts/train_audio_smollm2_shared_store_configurable_epochs_135m_mellow_ddp.py
code/RSmol/scripts/train_audio_smollm2_shared_store_smoke20_configurable_epochs_135m_mellow_ddp.sh
code/RSmol/scripts/train_audio_smollm2_shared_store_resume2_configurable_epochs_135m_mellow_ddp.sh
code/RSmol/scripts/train_audio_smollm2_shared_store_formal_configurable_epochs_135m_mellow_ddp.sh
code/RSmol/run_audio_smollm2_shared_store_smoke20_configurable_epochs_135m_mellow_5090.sh
code/RSmol/run_audio_smollm2_shared_store_resume2_configurable_epochs_135m_mellow_5090.sh
code/RSmol/run_audio_smollm2_shared_store_formal_configurable_epochs_135m_mellow_5090.sh
code/RSmol/audio_smollm2_135m_mellow_shared_store/README.md
code/RSmol/scripts/train_audio_perf20_5_10x2_5_mesh_mellow_ddp.sh
code/RSmol/run_audio_perf20_5_10x2_5_mesh_mellow_5090.sh
tests/test_audio_perf20_static.py
tests/test_audio_waveform_cache_perf20_static.py
tests/test_audio_shared_store_training_static.py
tests/test_audio_shared_store_configurable_epochs_static.py
tests/test_audio_smollm2_shared_store_configurable_epochs_static.py
tests/test_reasonaqa_component_partitions.py
tests/test_reasonaqa_partition_materialization.py
```

### 10.3 原始 SmolLM2 partition 对比线

```text
code/RSmol/audio_smollm2_135m_mellow/model.py
code/RSmol/audio_smollm2_135m_mellow/data.py
code/RSmol/scripts/train_audio_partitioned_smollm2_135m_mellow_ddp.py
code/RSmol/scripts/train_audio_smollm2_135m_mellow_smoke20_ddp.sh
code/RSmol/scripts/train_audio_smollm2_135m_mellow_resume2_ddp.sh
code/RSmol/scripts/train_audio_smollm2_135m_mellow_formal_ddp.sh
code/RSmol/run_audio_smollm2_135m_mellow_smoke20_3090.sh
code/RSmol/run_audio_smollm2_135m_mellow_resume2_3090.sh
code/RSmol/run_audio_smollm2_135m_mellow_formal_3090.sh
tests/test_audio_smollm2_partition_training_static.py
tests/test_audio_smollm2_135m_mellow_static.py
```

### 10.4 固定 5-10-5 recursive partition 对比线

```text
code/RSmol/audio_5_10_5_recursive_mellow/model.py
code/RSmol/audio_5_10_5_recursive_mellow/data.py
code/RSmol/scripts/train_audio_partitioned_5_10_5_recursive_mellow_ddp.py
code/RSmol/scripts/train_audio_5_10_5_recursive_mellow_smoke20_ddp.sh
code/RSmol/scripts/train_audio_5_10_5_recursive_mellow_resume2_ddp.sh
code/RSmol/scripts/train_audio_5_10_5_recursive_mellow_formal_ddp.sh
code/RSmol/run_audio_5_10_5_recursive_mellow_smoke20_5090.sh
code/RSmol/run_audio_5_10_5_recursive_mellow_resume2_5090.sh
code/RSmol/run_audio_5_10_5_recursive_mellow_formal_5090.sh
tests/test_audio_5_10_5_recursive_partition_training_static.py
tests/test_audio_5_10_5_recursive_mellow_static.py
```

该路线的 MMAU test-mini / MMAR 专用 full 评测入口也已单独新增，默认使用 `partition_formal_eos_v2_10epochs_20260919/checkpoint-037810`，不复用 MeSH 或 SmolLM2 的 checkpoint 加载器与输出目录。官方数据/scorer 层复用既有实现；每个生成 token 核对精确的 `0..14,5..14,15..19` 物理层轨迹，单音频为 130-token prefix、greedy、`use_cache=False`、最多 32 tokens。代码入口及提交命令见 `code/RSmol/audio_5_10_5_recursive_mellow/README.md`；远程评测结果尚未登记。

## 11. 已处理故障与不要重复的误诊

1. SmolLM2 tied embedding 导致转换时缺 `lm_head.weight`：转换器已按 tied-weight 语义处理。
2. MeSH meta tensor `.item()`：router 初始化已改为 meta-safe。
3. 文本 parquet 真正 shard 在 `SmolLM2-135M-10Bsubset/data`。
4. 某次 step 520 NCCL timeout 的根因是单 rank router-collapse 硬检查先退出，不是已经证明的网络故障；collapse 已改 diagnostic/warning。
5. prompt 和 answer 分别 padding 会把 pad 插到二者之间；当前已改为每条真实拼接后再 batch padding。
6. HTSAT checkpoint 必须加载 Mellow wrapper 对应的 child；曾经 official fallback 产生大量 missing/unexpected，当前严格拒绝 mismatch。
7. HTSAT frozen 不等于整条 wrapper `no_grad`：c2l 必须获得梯度，因此 forward 不能用全 wrapper `torch.no_grad()`。
8. 旧 mapper 不是严格 Mellow；当前已使用 c2l、CLS+frames、bias-free residual bridge、dropout 0.5、LayerNorm、CLS-preserving avgpool8。
9. 64-shard mmap 慢不是因为单纯 shard 数量，而是跨文件随机访问/page faults/共享存储长尾；当前正式路线已弃用。
10. shared mmap 被 rank0 warm 不等于每 rank 匿名 RAM；后者才在 PERF20 中达到理想稳态。
11. cgroup 路径可能是 v1/v2 混合；当前代码使用已有 hybrid-safe 探测，正式释放审计要求取得 anon 指标，否则拒绝下一 partition。
12. NCCL 日志停在 `Connected all rings` 不一定挂死：旧 smoke 在首次完整分区加载前没有进度打印，加载最大 partition 可持续数分钟；应先查作业状态与最早 Python exception，不要只看末尾 socket shutdown warning。
13. `loss≈1` 不能证明 prompt 被计入 loss；标签 mask 才是判断依据。模型是否真正利用音频应做正确/置零/打乱音频对照。

## 12. 当前风险与后续注意事项

- 旧 v1 两个 smoke 已由用户确认 PASS；当前 answer-EOS v2 合同必须重新跑 20+2 smoke，旧 report 会被正式 gate 拒绝。
- 原始 SmolLM2 partition 对比线同样必须先在 `pdgpu-3090` 重跑本路线 EOS-v2 的 20+2 smoke；旧三 epoch checkpoint、旧 STAGE7 report 或 MeSH smoke report 都不能放行该路线正式训练。
- 固定 5-10-5 recursive partition 路线已有用户提供的 10-epoch `checkpoint-037810`；新 MMAU/MMAR evaluator 会重新审计其本路线 EOS-v2 合同、37,810 step 完整游标、20-physical/30-logical 结构和 bridge/c2l hash，不会接受旧三轮 composite、文本或其他路线 checkpoint。
- MeSH/SmolLM2 等其余正式训练的远程完成状态仍以对应 checkpoint/report 为准；看到新结果后及时写入 job、output dir、最终 checkpoint 和 report 状态。
- 当前日志 loss 只是 rank 0 的四个 microbatch 平均，不是 8 rank 全局聚合 loss。
- 当前训练没有周期性 validation；模型选择需要另行设计只读评测，不要把 train report 当验证集表现。
- ReasonAQA samples generation、MMAU test-mini 与 MMAR 均已兼容 partition-v2 compact 130-token 单音频 prefix；远程正式分数仍以各自 `evaluation_report.json` 和 `official_evaluation.txt` 为准。
- HTSAT checkpoint 和 Mellow 源码不打包进复合 checkpoint；远程清理外部文件会导致 resume 失败。
- 10 epochs 很长；resume 时必须使用相同 epochs=10 和全部训练合同，否则 checkpoint 校验会拒绝。
- output dir 必须是新的空目录；脚本拒绝覆盖已有输出。
- 若正式训练某个分区 release 失败，不要放宽阈值后直接继续；先检查该 rank 是否仍持有 batch/cache/tensor 引用以及 cgroup anon/RSS 证据。
- shared-store 的 `/dev/shm` staging、mmap 和 persistent store identity 失败时不要退回到六分区或在线路径“顺便继续”；先保留独立 output/report，修复本路线问题后重新从 smoke20 开始。

## 13. 本地质量检查

修改当前主线后至少执行：

```powershell
python -m py_compile code/RSmol/recursive_model_5_10x2_5_mesh.py
python -m py_compile code/RSmol/audio_5_10x2_5_mesh_mellow/data.py
python -m py_compile code/RSmol/audio_5_10x2_5_mesh_mellow/model.py
python -m py_compile code/RSmol/scripts/train_audio_partitioned_5_10x2_5_mesh_mellow_ddp.py
python -m py_compile code/RSmol/scripts/train_audio_shared_store_5_10x2_5_mesh_mellow_ddp.py
python -m py_compile code/RSmol/scripts/train_audio_shared_store_configurable_epochs_5_10x2_5_mesh_mellow_ddp.py
python -m py_compile code/RSmol/scripts/train_audio_smollm2_shared_store_135m_mellow_ddp.py
python -m py_compile code/RSmol/scripts/train_audio_smollm2_shared_store_configurable_epochs_135m_mellow_ddp.py
python -m unittest discover -s tests -p "test_audio_partition_training_static.py"
python -m unittest discover -s tests -p "test_audio_5_10x2_5_mesh_mellow_static.py"
python -m unittest discover -s tests -p "test_audio_shared_store_training_static.py"
python -m unittest discover -s tests -p "test_audio_shared_store_configurable_epochs_static.py"
python -m unittest discover -s tests -p "test_audio_smollm2_shared_store_configurable_epochs_static.py"
python -m unittest discover -s tests -p "test_unique_audio_waveform_store_static.py"
git diff --check
```

本地缺少 Torch/Mellow/HTSAT 或远程权重时，只能说明相关运行测试未执行，不能写成 GPU PASS。

## 14. 新 Codex 会话接手清单

1. 完整阅读本文，先看第 0 节的路线状态和实验边界。
2. 执行 `git status --short`，保护用户已有修改；不要 reset/checkout 覆盖。
3. 阅读目标路线的 model/data/trainer/README/static test，不要只根据 README 猜代码。
4. 明确任务属于六分区正式训练、shared-store、online、silence-slot、历史 PERF20、评测，还是固定 5-10-5/SmolLM2 对照。
5. 旧六分区/固定 MeSH 音频训练默认文本初始化是第二轮 MeSH `checkpoint-009244`；新 7-slot/8-slot 文本转换和各音频路线必须按各自 README 指定的来源 checkpoint，已有音频 checkpoint 只能按其 contract 作为 init/resume。
6. 区分 `--model-path`、`--init-from-audio-checkpoint` 与 `--resume-from`。
7. 六分区仍是历史默认训练主线；shared-store、online、official Mellow 以及新的 7/8-slot 文本/音频路线都是隔离路线，不要交叉使用 smoke report 或 checkpoint。
8. shared-store 必须确认 v3 store、manifest SHA、`/dev/shm` 空间和 8 rank 同 inode 审计。
9. 任何正式提交都要核对两个同路线 smoke report 的实际绝对路径、seed、contract 和 `PASS`。
10. 远程失败时先找最早的 Python traceback/rank；末尾 NCCL `ChildFailedError` 通常只是连带结果。
11. 新的远程 PASS、job、output dir、checkpoint 或超参数确认后立即更新本文。

## 15. 参考资料

- MeSH：<https://arxiv.org/pdf/2510.07739>
- MeSH 官方仓库：<https://github.com/LivingFutureLab/MeSH/>
- Mellow：<https://arxiv.org/pdf/2503.08540>
- HTSAT：<https://github.com/RetroCirce/HTS-Audio-Transformer>
- Parcae：<https://arxiv.org/pdf/2604.12946>

外部论文用于解释设计；当前仓库代码、checkpoint config 和最新远程 report 决定实际实验合同。
