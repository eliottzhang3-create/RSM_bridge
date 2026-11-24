# HRM H2L3 BP5 + CED-small：PrefixLM音频训练，GBS128

本路线完整复制隔离自`mizar_s1_gbs128_text71427_mesh_ced_small`。本地Windows只维护代码，通过GitHub同步；权重、音频、索引、训练输出均位于远程Linux。既有71427音频训练、HRM文本训练和评测代码不改。

## 模型与初始化

文本来源固定默认为用户指定的raw最终checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/text_hrm_reinit3600_bp5/formal_20261122_200554_2644345/checkpoint-step-009244
```

登录CPU预检config/完整标记/9244步/原reinit3600 BP5 seed1来源身份及tokenizer库存；计算节点校验根部raw文件SHA，严格读取FP32 tensors并逐张量确认继承值、finite/形状/库存及embedding/head绑定。根部raw而非ema/作为初始化；不反序列化文本training_state.pt，不继承文本optimizer/EMA/RNG/游标，不转换模型、不折叠Norm、不清gate、不再次补偿o_proj。

保留P5/L5/H5/C5、20物理层、50次执行、H2L3、hidden576/MLP1536/GQA9:3/RoPE100000、MagicNorm、已训练gate及固定zL_init。文本LM独立参数102420288；显式新`AudioPrefixHRMForCausalLM`接入PrefixLM，不全局注册AutoModel，不将HRM伪装成普通20层Llama，也不增加MeSH/LoRA。

音频前端为原CED-small、原freq_merge_126 mapper及外置Identity：CED严格离线加载、全冻结eval，mapper随机初始化并训练。整段fixed20 waveform在CED前拼回，whole mel/BN/1012帧分段、逐样本dB及正确[split,freq,time]→[split,time,freq]重排保持；输出126×576。mapper全部频率/global/position/audio-type/refinement参数训练，dropout.05、resampleFalse。没有旧音频checkpoint初始化入口。

音频optimizer/scheduler/GradScaler/GradNorm/RNG/数据游标从step0建立。文本阶段Adam-atan2/EMA/BF16配置不带入本轮音频配方。

## PrefixLM、padding与loss

固定布局仍为`audio126 + SEP1 + prompt256 + answer300 =683`，prefix383。索引：audio0..125、SEP126、prompt127..382、answer383..682。

| Query / Key | Audio＋SEP＋prompt有效位置 | Answer有效位置 |
|---|---|---|
| Audio＋SEP＋prompt | 全双向 | 全部禁止 |
| Answer | 全可见 | 当前及以前可见 |

同一个显式四维加性mask贯穿Prelude、L的六次调用、H的两次调用和Coda，覆盖全部50次层执行；不允许任何prefix位置读取teacher-forced answer，因此递归不会形成间接答案泄漏。PrefixLM支持通过二维validity mask和显式prefix_length构造，不接受调用者随意传一个未经核对的四维mask；SDPA不再额外强加全程causal三角。

保留原S1文本构造、小写/模板/CoT规则；prompt与answer先追加` <|endoftext|>`，再按256/300截断和右padding，不强补被截掉的EOS。PAD使用已有`!`，EOS/SEP为0，不扩词表。prompt/answer padding按tokenizer attention_mask屏蔽K/V，不简单按token ID推断。

为保留旧683布局的首答案预测位置，query382无论prompt是否补齐都继续计算；当它对应PAD时，PAD仍不作为可读取的K/V，但该query可以读取有效prefix，用于预测第一个answer。不得把整行mask或输出清零；也不得在padding扰动测试中把这个固定answer-start query的输入当成无关位置随意更换。

外部answer-only shifted CE保持母本函数：logits382..681对应300个answer目标，tokenizer answer mask与target!=PAD共同确定有效项，未截断EOS参与loss。audio/prompt不另计loss；每rank每microbatch的有效answer token mean，再GA4平均和DDP平均，不改成全局token加权。

正式任务沿用原各source真实样本探针；第一个探针另核对完整683与仅prefix383的首答案logits/argmax一致，以及改变全部answer输入后prefix输出逐值相同。这是formal内部检查，不是单独GPU smoke或额外人工放行。

## BP5与Prelude任务梯度

全程固定BP5：H1、L4/L5/L6、H2记录梯度，L1/L2/L3不记录；Prelude和Coda正常求导，E不detach。H1直接使用E，保留loss→H2/H1→E→Prelude→音频mapper路径。固定L初值从checkpoint继承，每forward重建状态，无跨batch carry。

首micro反传审计实际50调用、BP集合、Prelude输出/E/H1/H2/L4/L5/L6边界及每个前五层参数梯度；首次完整GA4累积、unscale后且clip/optimizer/weight decay前再次逐层检查grad存在/finite、层聚合L2>0，并检查mapper非零梯度和CED无梯度。立即释放所有retain_grad引用。每更新继续全可训练参数缺失/非有限检查，任何rank失败集体报错，不以weight decay造成的变化冒充任务梯度。

报告为`first_recursive_backward_rank<N>.json`、`first_gradient_rank<N>.json`；新run及完整恢复后的首次更新都会执行。

## 固定训练配方

| 项目 | 值 |
|---|---|
| 数据 | 原hybrid1477492行、原resolved_paths.tsv、原source counts |
| Batch | 8GPU×micro4×GA4=GBS128 |
| Epochs / updates | 3；每epoch11542，总34626；每epoch尾部丢116行 |
| LM最大/最小LR | 6e-4 / 3e-5 |
| Mapper最大/最小LR | 8e-4 / 6e-5 |
| AdamW | beta(.9,.999)、eps1e-8、WD.01；bias和ndim≤1不衰减 |
| 精度 | FP32参数/forward，无autocast、TF32关闭；保留原GradScaler |
| 梯度裁剪 | 原GradNormTracker(1,10,2.5,.995)，保留异常终止和禁止静默skip |
| Seed | 1234；rank RNG1234+rank；原shuffle/sampler/epoch顺序 |
| 资源 | pdgpu-3090 / 64CPU / 256G / 8GPU / 1node |
| 保存 | 5771、11542、17313、23084、28855、34626及model--epo-3.ckpt |

LambdaLR在optimizer之后step。预热1731步，保留原startup最低0.1×max规则；LM起始6e-5、mapper起始8e-5。之后两个角色各自cosine floor：LM=.05×max、mapper=.075×max，不能共用原.1下限。为使用户指定min成为最后一次真实更新的LR，cosine分母使用`total_steps-1-warmup`。

令c为更新前已经完成的步数：预热c<1731使用`max(.1,(c+1)/1731)`倍最大LR；其后进度为`clamp((c-1731)/(34625-1731),0,1)`，LR=`min+(max-min)*(1+cos(pi*进度))/2`。第34626次真实更新使用LM3e-5/mapper6e-5，之后scheduler仍停在各自下限；没有lambda闭包串用角色问题。

沿用原CPU index及原始共享音频立即训练、默认4worker后台复制；全部rank在optimizer边界用MIN共同确认后切本地字节。tmpfs新后缀`/dev/shm/mizar_s1_gbs256_hrm_prefixlm_<token>`，`gbs256`仅母本安全守卫前缀，实际batch128。清理前先stop/reap producer，再校验owner/token；数据/索引/staging/CED等14份源码逐字节继承，Mellow两个类的AST保持。

## GitHub同步与正式提交

在本地仓库根PowerShell执行。sync_files.txt列出精确交付清单；测试依赖目录不会提交。

```powershell
git add -- code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small
git add -- code/RSmol/run_mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small_3090.sh
git add -- tests/test_audio_hrm_prefixlm_route.py tests/test_audio_hrm_prefixlm_tensors.py tests/test_audio_hrm_prefixlm_ddp.py tests/validate_audio_hrm_prefixlm.py
git add -f -- audit_reports/audio_hrm_prefixlm_validation.json
git add -p -- RM.md
git diff --cached --stat
git diff --cached --check
git commit -m "Add isolated HRM BP5 PrefixLM CED audio training"
git push
```

远程仓库根直接formal：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small_3090.sh
```

沿用原容器和`mellow_c8204d8`音频环境；HRM要求Transformers4.54.1，resolved config和训练入口均核对，不自动改共享环境/安装依赖。原`text_hrm_h2l3`与`text_hrm_reinit3600_bp5`作为模型/来源依赖保留在仓库中。

显式输出示例（新的空目录，脚本自动建立formal/）：

```bash
bash code/RSmol/run_mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small_3090.sh \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_hrm_bp5_prefixlm_ced_small_3090/formal_first_v1
```

默认run root为`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_hrm_bp5_prefixlm_ced_small_3090/formal_<timestamp>`，训练产物在其`formal/`；日志位于远程仓库`code/RSmol/log/mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small.*.JOB.log`。

允许覆盖`--text-model-dir`（仍须同文本路线raw最终9244）、`--ced-model-dir`、`--resolved-paths`、`--index-cache-root`、`--stage-workers`、`--output-dir`和`--resume-checkpoint`。不提供旧音频`--init-checkpoint`，不会误继承旧mapper/MeSH。

## 完整恢复和推理边界

音频完整checkpoint保存模型/已训练mapper/固定buffer、具名optimizer库存与moments、scheduler、Scaler、GradNorm、8rank RNG及下一个epoch/update游标。恢复严格核对本路线/PrefixLM/BP5、源码与共享HRM源码、文本和CED资产、数据/依赖/LR配方；校验tensor库存/shape/dtype/finite/绑定、冻结CED和固定zL_init、每参数步数、optimizer分组顺序/LR、scheduler/Scaler/GradNorm。其他路线或旧纯causalcheckpoint不能完整resume。

恢复沿用同一外层run root：

```bash
RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_hrm_bp5_prefixlm_ced_small_3090/formal_first_v1
bash code/RSmol/run_mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small_3090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt"
```

最终checkpoint为`<run-root>/formal/checkpoints/model--step-34626.ckpt`及`model--epo-3.ckpt`。vc入队不表示训练PASS，最终`formal/training_report.json`才表示34626次真实更新已完成。

Decoder提供PrefixLM一致的`inference_logits`：完整prefix383从query382预测首答案，随后拼已生成ID、answer继续causal，use_cacheFalse。通用`generate`/旧causal评测入口未适配，本次不改旧benchmark；后续评测须显式注册新模型和attention协议。

## 本地验证

在具备Torch/Transformers/Torchaudio/Soundfile/Safetensors的环境运行：

```bash
python -B tests/validate_audio_hrm_prefixlm.py --require-tensors
```

本次Windows测试使用已有`.codex_audio_cutoff_deps`的Torch2.9.0+cpu/Transformers4.54.1，加忽略目录`.hrm_test_runtime/audio_packages`中的torchaudio2.9.0+cpu/soundfile0.14.0；无修改共享Python。该Torch2.9 Windows Gloo在设备初始化时报unsupported，因此真实两进程Gloo另用已有`.hrm_test_runtime`的Torch2.15.0+cpu运行。对应命令：

```powershell
.\.hrm_test_runtime\Scripts\python.exe -B tests/validate_audio_hrm_prefixlm.py --dependency-dir .codex_audio_cutoff_deps --dependency-dir .hrm_test_runtime/audio_packages --ddp-python .hrm_test_runtime/Scripts/python.exe --require-tensors
```

报告`audit_reports/audio_hrm_prefixlm_validation.json`保留真实输出、依赖版本及失败历史。验证包括全部50次相同mask/双向影响/未来答案隔离/padding K/V和query382/训练-推理logits、逐层Prelude和mapper任务梯度、原始raw文本所有tensor/非零gate继承且不反序列化旧optimizer、两组34627个LR位置、具名optimizer/Scaler/GradNorm/RNG完整恢复后下一更新逐值一致、实际正式三轮训练及resume控制流。

2026-11-24最终22/22通过、无跳过：10项路线/脚本/staging、11项真实CPU张量/音频、1项真实双进程DDP。DDP梯度对照最大绝对差约2.98e-8；实际训练控制流的128行合成夹具每epoch1次更新，完成三epoch后再从step1完整恢复至step3，最终参数逐值一致，不冒充生产34626步或八卡执行。

双进程DDP执行实际HRM/Decoder与原mapper/loss的精确AST，比较两rank/两micro各自token mean后的梯度，包含不等答案长度；实际CUDA/八rank模拟训练夹具明确单列。合成WAV的真实解码/fixed20/whole mel/时频重排也执行；生产CED encoder权重前向由fixture替代，本地不读取服务器资产。真实Bash模拟vc和torchrun验证3090资源/含空格参数/失败清理；复制完成前训练可继续的线程回归执行原staging实现。

任务开始3701份tracked Python/Shell/JSON摘要保持，原音频数据/CED/mapper实现及answer_loss/GradNorm AST保持；原HRM recurrence从cos_sin之后的forward AST保持一致。Python3.10、两Shell、README命令、whitespace及Git可见性另核对。CPU通过不代表Torch2.11/CUDA、3090显存、八卡NCCL或生产文本来源已经验证；真实结果以远程formal报告为准。

`source_guard.json`是本次实际源码保护库存；继承的`parent_snapshot.json`只保留母本历史出处，不用于声称本新路线已通过旧审计。早期测试中的Bash夹具工作目录、极小CE reduction舍入断言及模拟encoder闭包deepcopy问题已修正，Gloo环境限制单列；不修改旧测试/历史快照来掩盖差异。
