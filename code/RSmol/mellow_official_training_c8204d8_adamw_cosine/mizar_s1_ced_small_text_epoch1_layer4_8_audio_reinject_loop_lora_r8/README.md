# Mizar 文本 epoch1 → CED4/8串行音频注入＋两轮独立LoRA

本目录完整复制隔离自
mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8。
正式入口为 code/RSmol/run_mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_5090.sh。

本地Windows只维护代码，通过GitHub同步；模型、数据、输出及GPU训练均在远程Linux。
本路线与母本、普通CED文本epoch1音频路线分别拥有独立的输出、log、job、index-cache、tmpfs和恢复身份。

## 初始化

默认从以下完整文本checkpoint继承全部LM权重和tokenizer：

~~~text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_gbs1024/epoch1/checkpoint-step-009244
~~~

显式使用本目录的非MeSH RecursiveLlamaForCausalLM，严格检查全部
missing/unexpected/mismatched/error后，才依次注册CED层注入、共享input音频再注入和LoRA。
20物理层执行前5→中10→中10→后5，共30次；保留继承权重、embedding/head绑定及原tokenizer合同。

这是新的音频S1训练：仅继承HF文本权重/tokenizer；不反序列化文本training_state.pt。
optimizer、scheduler、GradScaler、GradNormTracker、每rank RNG、step及数据游标重新建立。
无需再转换，也无需先训练文本epoch2。CED由原models/CED_small加载，冻结并保持所有子模块eval；
mapper、音频分支和LoRA按母本初始化；基础LM及新增可训练参数全量训练。

## 模型与训练保持母本一致

- CED每批编码一次，取final以及完整block4、block8的final norm前输出。
  原freq_merge_126 mapper生成E [B,126,576]作为输入音频前缀。
- A4/A8分别使用独立LN1536→Linear1536/576→GELU→Dropout0→Linear576/576→LN576。
  每次完整LM forward由原E只计算一次共享audio_linear(E)，不detach。
- 零基逻辑索引5：共享input门控残差→CED4残差；索引15：同一共享input残差→CED8残差。
  CED门控读取post-input hidden，直接残差只更新前126个位置，三个门控零初始化。
- 中间物理层5..14的q/k/v/o、gate/up/down七投影，按轮分别使用两套独立A/B。
  rank8、alpha16、scale2、dropout0，A Kaiming/B零，新增1,628,160参数。
  基础W/bias共享且全量训练；前后五层无LoRA。音频分支新增2,777,859参数。
- 保留初始化RNG隔离、完整序列LoRA、调用后及异常时清理轮次选择、
  use_cache=False、关闭decoder gradient checkpointing、首步及打开门控后的梯度审计。
- 原hybrid 1,477,492行、manifest/resolved_paths、fixed20在线处理；
  126audio＋1SEP＋256prompt＋300answer＝683token，prefix383。
  PAD为已有!/17、EOS/SEP0、attention_mask=None；
  answer-only rank-local有效token mean→GA4平均→DDP平均。
- 8GPU×microbatch8×GA4＝GBS256，3epochs，每epoch5771updates，总17313updates，
  warmup866，每epoch丢116行；seed1234、rank seed1234+rank，原采样顺序。
- 基础LM maxLR3e-4；mapper、音频投影/门控及两套LoRA maxLR6e-4。
  同一LambdaLR warmup/cosine floor0.1，AdamW betas(.9,.999)/eps1e-8，
  矩阵WD.01、bias/ndim≤1不衰减，原GradNormTracker。
- FP32 forward，无autocast/TF32，保留GradScaler。
  共享音频立即训练，后台默认4workers复制，optimizer边界全rank共同切换。
- 保存step2886/5771/8656/11542/14428/17313以及epoch3。
  原完整音频checkpoint恢复含模型、两套LoRA、音频分支、全部训练状态、8rank RNG及采样游标。

train.py、optim.py、全部models/data/staging/index/CED配置代码与指定母本逐字节一致。
contract.py只改ROUTE和默认文本路径；run.sh只改入口提示和tmpfs前缀；
提交器只改引用目录、初始化路径、输出/log/job/index-cache及提示。
audit_reference.py改为直接对照此次LoRA母本的精确差异检查；不参与正式训练。
35份原始参考文件的归一化LF SHA256记录于source_guard.json。
.gitignore仅增加放行source_guard.json，确保GitHub同步包含该清单。
母本历史reference_audit.json没有被复制为新路线的验证结果。

## 远程正式提交

先在本地提交并推送交付文件，再在远程仓库根执行：

~~~bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_5090.sh --stage-workers 4
~~~

默认已经指向上面的文本epoch1 checkpoint。可显式指定来源和新的输出目录：

~~~bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_gbs256_5090/formal_$(date +%Y%m%d_%H%M%S%N)
bash code/RSmol/run_mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_5090.sh \
  --text-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_gbs1024/epoch1/checkpoint-step-009244 \
  --output-dir "$RUN" \
  --stage-workers 4
~~~

正式入口保留原CED资产检查及CPU metadata索引准备后直接vc submit；
无自动smoke或CPU张量测试前置。登录节点只做CPU准备和提交，torchrun/CUDA在计算节点执行。
环境mellow_c8204d8；原镜像；pdgpu-5090/32CPU/256G/8GPU/1node。

保留--output-dir、--resume-checkpoint、--resolved-paths、--text-model-dir、
--ced-model-dir、--index-cache-root、--stage-workers参数。

## 输出与完整恢复

默认run root：

~~~text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_gbs256_5090/formal_<timestamp>/
~~~

沿用母本，run root下再建立formal/。主要产物：

~~~text
formal/runtime.json
formal/config_audit.json
formal/trainable_params.txt
formal/first_gradient_rank<N>.json
formal/opened_gate_gradient_rank<N>.json
formal/staging_report.json
formal/metrics.jsonl
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/training_report.json
~~~

独立route为ced_small_mizar_s1_fixed20_gbs256_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_v1。
index cache位于上述输出大目录下的index_cache/；
log为code/RSmol/log/mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_5090.<timestamp>.JOB.log；
tmpfs前缀为/dev/shm/mizar_s1_ced_text_e1_in48_loop_lora_r8_。

仅接受本路线同源码/源资产/包版本/数据/配方、原run root内的完整音频checkpoint。
恢复继续累计到17313updates，保留已经训练的全部LoRA及音频分支。
将RUN改为需要恢复的实际run root：

~~~bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_gbs256_5090/formal_实际时间戳
bash code/RSmol/run_mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_5090.sh \
  --output-dir "$RUN" \
  --resume-checkpoint "$RUN/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
~~~

--text-model-dir是新音频训练的LM初始化来源；
--resume-checkpoint是本路线训练生成的完整音频.ckpt文件。

## 本地验证与Git提交

仓库根运行：

~~~powershell
python -B tests/validate_mizar_s1_ced_text_epoch1_loop_lora.py
~~~

报告audit_reports/mizar_s1_ced_text_epoch1_loop_lora_validation.json：
40项本地检查通过，7项真实CPU tensor检查因缺Torch/Transformers依赖而跳过。
覆盖35份参考SHA、完整源码差异白名单、整个semantic recipe、全部17314个LR位置、
原GradNorm、文本epoch1配置兼容、step0初始化不读文本训练状态、跨路线恢复拒绝、
严格加载后才注册分支、非零分轮LoRA及scale2、post-input门控、梯度审计、
共享读取/后台复制/异常清理及Bash模拟正式vc提交资源和参数。
NumPy/fixture检查不等同于PyTorch autograd。
Windows沙箱拒绝临时目录和Bash信号管道后，经授权在沙箱外运行同一套本地检查。

有依赖时可在远程mellow_c8204d8环境手动运行合成CPU检查；不需要生产模型或数据，也不是提交前置：

~~~bash
python -B tests/validate_mizar_s1_ced_text_epoch1_loop_lora.py --require-tensors
~~~

7项tensor测试覆盖A/B独立存储、基础权重继承及RNG、零LoRA logits/基础梯度、
非零逐投影按轮路由、首步与打开门控后的梯度、完整CPU checkpoint下一步恢复、
严格文本加载及复制模型初值一致性。CPU collective/scaler为明确fixture，
本地报告不代表远程真实权重加载、GPU训练或八rank恢复已通过。

本地仅暂存本次新路线，避免包含工作区其他未提交实现：

~~~powershell
cd C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
git add -- code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8 code/RSmol/run_mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_5090.sh "tests/*mizar_s1_ced_text_epoch1_loop_lora*.py" audit_reports/mizar_s1_ced_text_epoch1_loop_lora_validation.json
git diff --cached --stat
git commit -m "Add epoch1 initialized CED audio training with loop LoRA"
git push
~~~

RM.md已前置本次记录，并保留之前的全部字节；它已有其他未提交记录，
如需随本次提交，可在commit前执行git add -p -- RM.md并仅选择本次记录。
本次未自动执行git commit/push或真实vc提交。

