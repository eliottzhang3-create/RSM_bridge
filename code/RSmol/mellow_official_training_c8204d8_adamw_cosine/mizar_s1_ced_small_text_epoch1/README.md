# Mizar 文本 epoch1 → CED-Small 音频 S1，GBS256

## 2026-11-12 staging启动修复

原隔离tmpfs名称缺少stage_data.py要求的mizar_s1_gbs256_前缀，导致producer启动失败，
wait_for_start随后报未发布TRAINING_READY，退出清理再次被同一校验拒绝。
现仅修正run.sh的临时目录名称为：

~~~text
/dev/shm/mizar_s1_gbs256_ced_small_text_epoch1_<token>
~~~

保留原路径/软链接/所有权保护及全部训练设置。新增回归检查实际Bash路径→原validate_stage_root/cleanup，
重现旧名称失败，覆盖新名称通过、两路线隔离及错误路径/软链接/所有权拒绝。
同步修复后重新执行本页正式提交命令，默认生成新时间戳输出目录；本次staging失败尚未进入训练，无需resume。

本路线完整复制自 `mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small`，
对应原入口 `run_mizar_s1_gbs256_recursive_5_10_5_ced_small_5090.sh`。
本地 Windows 维护核心代码，经 GitHub 同步；权重、数据及训练全部在远程 Linux。

## 初始化与隔离范围

默认直接加载用户指定的文本 epoch1 最终 checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_gbs1024/epoch1/checkpoint-step-009244
```

无需再转换模型或完成文本 epoch2。显式使用原 `recursive_model_5_10_5.RecursiveLlamaForCausalLM`，
保留20物理层、前5→中10→中10→后5的30次执行、原选层映射、全部已训练LM权重、绑定embedding/head及对应tokenizer。
HF加载严格拒绝missing/unexpected/mismatched keys和error_msgs。

只加载LM权重和tokenizer；不读取文本 `training_state.pt`。音频训练从step0开始，
optimizer、scheduler、GradScaler、GradNormTracker、八rank RNG及数据进度按原音频S1方法重新初始化。
CED-Small仍从 `models/CED_small` 加载，完全冻结且始终eval；mapper按母本随机初始化；完整mapper与完整LM训练。

生产代码仅有以下差异：

- `contract.py`：新ROUTE身份、默认文本checkpoint路径。
- `run.sh`：新入口提示、独立tmpfs前缀。
- 独立提交器：新代码目录、默认文本路径、输出/log/job/index-cache名称及提交提示。

`train.py`、`optim.py`、所有模型/数据/音频预处理/离线CED配置/索引/staging代码与母本逐字节一致。
共享模型、母本及文本训练文件没有修改；35份参考源的归一化LF SHA256记录在 `source_guard.json`。
母本历史 `reference_audit.json` 不作为本路线验证报告复制。

## 保持不变的音频训练设置

| 项目 | 与母本相同的设置 |
|---|---|
| 数据 | `data/Mizar/s1_stage1_hybrid_filtered.json`，1,477,492行，原resolved_paths.tsv |
| 波形 | 原始音频单声道、重采样/中心裁剪/右补零为32kHz fixed20，拼回整段20秒后做CED前处理 |
| CED | 冻结eval，16kHz mel、1012-frame切片、原frequency/time顺序，在线FP32提取，无hidden cache |
| Mapper | 官方 `freq_merge_126`，dropout0.05、resample=False，输出126×576 |
| 序列 | audio126＋SEP1＋prompt256＋answer300；prefix383、总长683 |
| Tokenizer | 来自文本checkpoint；按S1把已有 `!` 注册为PAD，不扩词表；EOS/SEP=0 |
| LM forward | `attention_mask=None`、`labels=None`、`use_cache=False` |
| Loss | 原answer-only masked CE；rank-local有效token mean，除GA4后反传，再DDP平均 |
| Batch | 8 GPU × micro8 × GA4 = GBS256 |
| 训练长度 | 3 epochs，每epoch5771步，共17313步；每epoch丢弃116行 |
| LR | LM max3e-4、mapper max6e-4；原LambdaLR 866步warmup＋cosine、floor0.1 |
| AdamW | betas=(0.9,0.999)、eps1e-8、WD0.01，bias和ndim≤1不衰减 |
| 梯度控制 | 原GradNormTracker：initial L2=1/max=10、overdrive2.5、momentum0.995 |
| 精度 | FP32参数和forward，无autocast，保留GradScaler，TF32关闭 |
| 顺序与RNG | seed1234、rank seed1234+rank、原shuffle/sampler/drop-last；DataLoader workers0 |
| 原始音频复制 | 从共享路径立即训练，后台默认4 workers复制，完整复制后在optimizer边界全rank MIN切换 |
| 保存 | step2886/5771/8656/11542/14428/17313及最终epoch3完整checkpoint |
| 环境与资源 | `mellow_c8204d8`、原镜像、pdgpu-5090、32CPU、256G、8GPU、1node |

新身份为 `ced_small_mizar_s1_fixed20_gbs256_text_epoch1_5_10x2_5_v1`。
完整恢复合同仍包含代码/模型源码/源权重与tokenizer SHA、数据、配方及包版本；
旧音频路线、其他文本初始化或其他配方的完整checkpoint不能恢复到本路线。
使用父路线完全相同的训练状态保存、八rank RNG恢复和采样游标恢复流程。

## 正式提交

本地提交并push代码后，在远程登录节点同步，然后直接提交正式音频训练：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_text_epoch1_5090.sh --stage-workers 4
```

默认已经使用上述epoch1 checkpoint。若希望命令中显式写明初始化和输出位置：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/run_mizar_s1_ced_small_text_epoch1_5090.sh \
  --text-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_gbs1024/epoch1/checkpoint-step-009244 \
  --ced-model-dir /hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_text_epoch1_gbs256_5090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

保持原入口的CED配置检查和CPU索引准备，然后直接 `vc submit`；不增加smoke、CPU张量测试或文本数据扫描前置。
登录节点只执行原有资产/索引准备；CUDA训练在vc作业内执行。

默认输出根目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_text_epoch1_gbs256_5090/formal_<timestamp>/
```

沿用母本语义，指定的 `--output-dir` 下会再建立 `formal/`。主要产物：

```text
formal/runtime.json
formal/config_audit.json
formal/trainable_params.txt
formal/first_gradient_rank<N>.json
formal/staging_report.json
formal/metrics.jsonl
formal/checkpoints/model--step-17313.ckpt
formal/checkpoints/model--epo-3.ckpt
formal/training_report.json
```

log位于远程 `code/RSmol/log/mizar_s1_ced_small_text_epoch1_5090.<timestamp>.JOB.log`。
index cache为新输出根下的 `index_cache/`；tmpfs为 `/dev/shm/mizar_s1_gbs256_ced_small_text_epoch1_<token>`。
源码不保存模型、数据或训练产物。

## 同一音频run的完整恢复

保持原run的 `--output-dir` 和文本/CED/data来源，传入本路线的完整音频 `.ckpt`，
累计继续到step17313；完整恢复会读取音频模型、optimizer、scheduler、scaler、tracker、每rank RNG和采样进度：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_text_epoch1_gbs256_5090/formal_实际时间戳
bash code/RSmol/run_mizar_s1_ced_small_text_epoch1_5090.sh \
  --output-dir "$RUN" \
  --resume-checkpoint "$RUN/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

文本epoch1目录是新的音频训练的LM初始化来源；`--resume-checkpoint`专门用于本路线已生成的完整音频checkpoint文件。
恢复要求相同代码及包版本、配方、数据、源资产身份，与母本一致。

## 本地验证与Git提交

```powershell
cd C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
python -B tests/validate_mizar_s1_ced_text_epoch1.py
```

报告为 `audit_reports/mizar_s1_ced_text_epoch1_validation.json`。
49项本地检查通过，3项真实CPU tensor检查因缺Torch等依赖跳过。
检查包括35份母本SHA、全部未调整文件逐字节相等、完整semantic recipe、17314个LR位置、
原GradNorm策略、实际初始化/恢复分支、严格LM加载、tokenizer、共享读取/后台复制/异常清理、
Bash模拟正式提交资源和参数，以及原更新/数据/音频代码继承。
Windows沙箱无法访问测试临时目录或启动Bash信号管道后，在授权的沙箱外运行同一套本地检查。

有依赖时，可在远程 `mellow_c8204d8` 环境执行合成CPU检查，不需要生产模型/数据，也不是正式训练前置：

```bash
python -B tests/validate_mizar_s1_ced_text_epoch1.py --require-tensors
```

可选CPU tensor用例保留原mapper/LM反传、30次执行和完整checkpoint往返恢复检查。
本地结果不代表远程epoch1权重已经加载或真实八卡训练/恢复已经通过；未读取远程生产资产或提交真实作业。

本地提交两条路线的本次修复及回归检查：

```text
git add -- RM.md code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_text_epoch1 code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8 "tests/*mizar_s1_ced_text_epoch1*.py" audit_reports/mizar_s1_ced_text_epoch1_validation.json audit_reports/mizar_s1_ced_text_epoch1_loop_lora_validation.json
git diff --cached --stat
git commit -m "Fix staging paths for both Mizar text epoch1 audio routes"
git push
```

未自动执行git commit/push或真实vc提交。
