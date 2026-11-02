# CED small / MeSH 5+10×4+5 / Mizar S1 GBS128

本路线完整复制 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small`，
将文本模型及其初始化源切换为现有四循环 MeSH，并将 microbatch 从8改为4。
文本加载器逐字复用 `mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_5_10x4_5`
的严格 x4 loader；该历史目录和入口名虽然包含 gbs256，实际 batch 配置也是8×4×4。
CED 音频处理、官方 mapper、损失函数、优化器及完整训练更新循环均沿用 CED 母本。

本地 Windows 维护核心代码，通过 GitHub 同步到远程 Linux；权重、数据与正式训练环境
均使用服务器现有文件。新 route、output、job、log、index-cache、tmpfs 和恢复身份独立。
原两条参考路线、原启动器和共享 MeSH 模型未修改。

## 训练合同

| 配置 | 本路线 |
|---|---|
| 文本模型 | hidden576；20个物理层；50次逻辑执行；中段10层循环4次 |
| 执行顺序 | `0..4, (5..14)×4, 15..19`，零基索引 |
| MeSH | 全序列7个瞬时memory槽；5组独立write/read，共10个router；每次forward重建memory |
| 训练角色 | CED冻结且所有子模块始终eval；mapper按母本随机初始化并全量训练；完整LM及router可训练，无LoRA |
| 数据 | hybrid 1,477,492行；保留MusicCaps和全部SpeechCraft splits |
| 输入 | 全通道平均、32kHz；单录音先中心裁剪/右补零至20秒，再按母本接口分两半 |
| CED特征 | 重新合并完整20秒后重采样16kHz；逐样本dB；冻结BN后按1012帧切分，保持频率/时间顺序 |
| mapper | 官方`freq_merge_126`；384维/4频率格→576维；126 tokens；dropout0.05；无额外projection |
| 序列 | audio126 + SEP1 + prompt256 + answer300；prefix383 / total683 |
| tokenizer | 文本checkpoint自带词表；注册已有`!`为PAD；不扩词表；EOS/SEP=0 |
| LM接口 | `attention_mask=None`、`labels=None`、`use_cache=False`；PAD仍参与上下文注意力 |
| loss | 外部answer-only CE；answer attention mask与非PAD共同选取有效目标，包含EOS；rank-local token mean |
| Batch | 8 GPU × microbatch4 × accumulation4 = GBS128；每rank每update消费16行 |
| 时长 | 3 epochs；11,542 updates/epoch；共34,626 updates；每epoch丢弃116行 |
| AdamW | LM/routers maxLR3e-4；mapper maxLR6e-4；betas0.9/0.999；eps1e-8；weight decay0.01，bias和ndim≤1不衰减 |
| LR | 保留母本LambdaLR；5% warmup + cosine；floor0.1；optimizer后scheduler；warmup=`round(34626×0.05)=1731` |
| GradNorm | initial L2=1、max=10、overdrive2.5、momentum0.995 |
| 数值 | FP32 forward；无autocast；GradScaler启用；TF32关闭 |
| 顺序 | seed1234；rank RNG=1234+rank；原shuffle/rank连续分片；DataLoader workers=0 |
| staging | 共享原始音频立即开始训练，后台字节复制；全量复制完成后在optimizer边界八卡一致切换；默认4个复制worker |

降低GBS后，保持3 epochs会使总更新数从17,313变为34,626；学习率峰值、warmup比例、
cosine公式与其他训练超参数保持母本值。`geometry()`统一派生训练、数据游标、保存步数和
作业完成检查，避免提前按旧17,313步判断完成。第一批梯度审计同时读取模型已有trace，
确认完整四循环及50次执行，不添加forward hook或随机运算。

## 服务器现有资产

四循环 MeSH 文本初始化默认路径与现有 x4 启动器一致：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/formal_2epochs_18488steps_20261007_004959_3090_v1/checkpoint-018488
```

使用共享 `recursive_model_5_10x4_5_mesh.py`；校验hidden576、logical50、physical20、
loops4、slots7、routers10和完整schedule。离线FP32全量加载；missing、unexpected、
mismatched或error任一非空即拒绝，不允许two-loop ablation。该文本checkpoint只用于初始化。

CED默认路径：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small
```

保留母本6个必需文件：`config.json`、`configuration_ced.py`、`modeling_ced.py`、
`feature_extraction_ced.py`、`preprocessor_config.json`、`model.safetensors`。
保留严格CED几何/权重检查和离线label metadata loader，直接使用本地527类标签配置。

数据与路径解析表保持 CED 母本默认值：

```text
manifest=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
resolved_paths=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
```

## GitHub同步与正式提交

本次只交付本地代码与命令，未执行git commit/push或远程提交。在本地仓库PowerShell中，
可仅提交本次文件；下面的文件清单不包含仓库中其他未跟踪文件：

```powershell
Set-Location C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
git add -- README.md code/RSmol/run_mizar_s1_gbs128_recursive_mesh_5_10x4_5_ced_small_5090.sh code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs128_3epochs_recursive_mesh_5_10x4_5_ced_small tests/test_mizar_s1_mesh_x4_ced_small_route.py tests/test_mizar_s1_mesh_x4_ced_small_shared_start.py tests/test_mizar_s1_mesh_x4_ced_small_behavior.py tests/test_mizar_s1_mesh_x4_ced_small_tensor.py tests/validate_mizar_s1_mesh_x4_ced_small.py audit_reports/mizar_s1_mesh_x4_ced_small_validation.json
git diff --cached --stat
git commit -m "Add GBS128 four-loop MeSH CED-small training route"
git push origin main
```

代码同步后，在远程登录节点直接提交formal：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
cd code/RSmol
bash run_mizar_s1_gbs128_recursive_mesh_5_10x4_5_ced_small_5090.sh --stage-workers 4
```

入口沿用 `vc submit`、`mellow_c8204d8`、`pdgpu-5090 / 32 CPU / 256G / 8 GPU / 1 node`。
直接提交正式训练，无自动smoke/tensor前置作业。登录节点先检查CED资产并准备/复用CPU
metadata索引，再申请GPU。GPU作业保留母本每来源样本的真实loader/model/loss探测，
探测后恢复RNG，再开始正式更新。

默认输出根目录为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_recursive_mesh_5_10x4_5_5090/formal_<timestamp>/
```

实际产物在该根目录的 `formal/` 子目录：

```text
formal/runtime.json
formal/config_audit.json
formal/first_gradient_rank<N>.json
formal/trainable_params.txt
formal/metrics.jsonl
formal/staging_report.json
formal/checkpoints/model--step-34626.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/training_report.json
```

中间step checkpoint为5771、11542、17313、23084、28855、34626，另外每epoch保存。
入口输出包含实际output路径；作业日志位于远程 `code/RSmol/log/` 的本路线独立文件。
训练器完成全部34,626次更新、检查冻结CED参数/buffer不变并严格重载最终完整checkpoint后，
才写入 `training_report.json` 的PASS；作业入口再检查该PASS和更新数。

可选参数沿用母本：`--output-dir`、`--resume-checkpoint`、`--resolved-paths`、
`--text-model-dir`、`--ced-model-dir`、`--index-cache-root`、`--stage-workers`。
新训练要求输出目录为空；推荐保持默认的独立时间戳输出。

## 完整断点恢复

仅接受本新路线、相同源码/共享模型/资产/数据/配方的完整formal checkpoint。
其中包含模型、optimizer、scheduler、GradScaler、GradNorm、每rank RNG与数据游标；
新GBS几何同时约束sampler和后台staging恢复位置。旧x2 CED、x4 HTSAT、标准CED
音频checkpoint以及文本初始化checkpoint均不能直接作为本路线resume。

恢复时保留原输出根目录，下面的时间戳和step改为实际存在的本路线文件：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
RUN_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_recursive_mesh_5_10x4_5_5090/formal_<原时间戳>"
bash run_mizar_s1_gbs128_recursive_mesh_5_10x4_5_ced_small_5090.sh \
  --stage-workers 4 \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-11542.ckpt"
```

若首次训练覆盖了资产/数据路径，恢复时也须传入相同路径。恢复不会重置调度或按新的
3 epochs续加训练；从保存游标继续完成原34,626步合同。

## 验证结果与边界

仓库根目录统一验证入口：

```bash
python -B tests/validate_mizar_s1_mesh_x4_ced_small.py
```

本地Python/Bash语法、63项实际测试通过；21项真实CPU tensor测试因缺少
torch/torchaudio/transformers/soundfile跳过，其中包括继承的16项CED检查与新增5项x4检查。
报告为 `audit_reports/mizar_s1_mesh_x4_ced_small_validation.json`。
具备依赖时可手动追加 `--require-tensors`，任一依赖缺失或测试skip都会令检查失败。
检查不下载资产、不使用CUDA、不提交作业，也不作为formal自动前置任务。

`audit_reference.py` / `reference_audit.json`记录53个原文件normalized-LF SHA保护，
CED数据/音频/mapper/优化器文件、完整训练更新循环及关键函数AST一致性，
并验证新loader与既有x4 loader完全一致。实际行为测试覆盖严格加载拒绝、四循环trace、
34,627个LR位置、GBS128 staging优先级、完整恢复身份和Bash模拟提交。
张量检查可验证683-token mapper→x4梯度、七槽瞬时memory、四轮边界梯度、严格save/load、
完整checkpoint恢复后的下一步一致性；CPU恢复用例模拟scaler和八rank RNG收集。

本轮未执行远程真实权重加载、八卡训练或真实分布式恢复，未登记远程PASS。
现有CED评测器绑定各自x2/17,313步训练合同；本新路线的评测需要另行适配。
