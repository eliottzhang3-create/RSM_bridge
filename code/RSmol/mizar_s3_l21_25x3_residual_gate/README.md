# S3：L21–25 三遍共享循环、独立残差门控、训练 L26

本路线从 `mizar_s3_l11_14x3_residual_gate` 复制隔离，仅迁移循环区间、循环宽度和后接可训练层。入口为 `code/RSmol/run_mizar_s3_l21_25x3_residual_gate_5090.sh`。原 L11–14 参考入口及旧转发入口保留。代码在本地 Windows 维护，经 GitHub 同步；正式资产、数据和训练输出位于远程 Linux。

## 模型和初始化

```text
L1–20 → [L21–25 第一遍] → [L21–25 第二遍] → [L21–25 第三遍] → L26 → L27–30 → 原 final norm/head
```

三遍为总执行次数；五层共享同一组 Parameter，保留全部30个物理层，共40次 block 调用。L26 只执行一次。第一遍普通执行，第二、第三遍各使用独立 gate 和循环偏置：

```text
H1 = F(H0)
Xr = Hprev + pr
Gr = sigmoid(Linear_r(RMSNorm(Xr)))
Hr = Xr + Gr * (F(Xr) - Xr)          r=2,3
```

`F` 是完整 L21–25 五层块；gate 在五层块末尾混合整个更新，不在各层单独混合。RMSNorm 无 affine，epsilon 沿原 LM；两个 `Linear(576,1)` 互相独立，weight=0、bias=-4.595，初始 gate≈0.01000119。gate 为逐 token 标量，广播576通道。两个576维循环偏置全零初始化，分别在第二、第三遍之前加到所有 token，同时进入 gate、F 和残差直通分支。音频、SEP、prompt、answer、PAD 全部适用；第一遍没有 gate 或偏置。没有音频锚点或额外音频重注入。

原样 strict 加载 `/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt` 的454个 FP32 状态，逐值验证后再注册两个 gate 和偏置。18个 release 文件大小/SHA检查保持。此次从官方 final 新训练，不从其它训练 checkpoint 初始化。

仅 L21–26 全部 attention/MLP/norm 和两个 gate/两个偏置训练。共享 L21–25 为17,700,480参数，L26为3,540,096，新增参数2,306，合计 **21,242,882可训练**；原冻结参数 **138,059,264**。状态字典459项。CED、已训练 projector、embedding/tied head、其它层和 final norm 冻结并保持 eval。前20层 no_grad；三遍完整 BPTT、不 detach，冻结后缀保留输入梯度。无 KV cache、decoder gradient checkpointing 或跨 forward 隐状态；final norm 仅最后一次。

## 数据和训练配方

直接复用已经封存的完整 S3：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s3_full_filtered_v1
```

读取 `train.json` 和 `resolved_paths.tsv`，每个 epoch 全部272,796行：AVQA26481、AudioCaps49486、Clotho1991、CompA-R48045、LP-MusicCaps-MTT7871、MusicCaps631、SpeechCraft113975、TACOS24316。只读校验原 seal、计数、映射与文本；不重新生成数据。新的不可变 CPU index 绑定本路线代码与 tokenizer 身份，预检失败不申请 GPU。

沿用参考的运行时音频处理：SoundFile → resample32k → mono → 超长取中心20秒、短音频右补零至640000采样 → 10+10秒重拼整段在线 CED/projector。中心起点为 `(n-640000)//2`。原封存元数据里的历史 first20 字段保留，本路线合同明确 center20 运行时覆盖，数值行为与参考一致。

原 S3 question/原序小写选项和答案保持：prompt256无EOS，answer96加空格EOS，禁止截断。126audio+SEP1+prompt256=prefix383，总479；PAD17、SEP/EOS0、原因果 PAD 可见、attention_mask=None，logits382..477监督96个答案位置并忽略答案 PAD。

| 参数组 | max LR | min LR |
|---|---:|---:|
| 两个 gate＋两个循环偏置 | 6e-5 | 6e-6 |
| 共享 L21–25 | 1.5e-5 | 1.5e-6 |
| 单次 L26 | 5e-6 | 1e-6 |

两完整 epochs，8GPU×micro4×GA4=GBS128，每轮2,132步，总 **4,264次实际更新**。每轮末批28行、GA1，前4rank各4行、后4rank各3行；无丢尾、无补齐重复，每轮seed1234+epoch确定性打乱。原 rank-local有效answer token mean → 实际GA mean → active rank mean 保持，空rank为零任务loss通信。

前100次更新 warmup，启动下限0.1×各组max，第100次更新到峰值；随后各组 cosine，末次实际更新到各自min，epoch边界不重置。AdamW(.9,.999)/eps1e-8/WD.01，一维、bias和循环偏置不衰减。GradScaler、GradNormTracker、非有限检查和1e6应急阈值保持。FP32、无 autocast、TF32关闭。

在 **1066、2132、3198、4264** 保存完整 checkpoint，keep4。模型、具名AdamW moments、scheduler、scaler、tracker、8rank RNG和epoch/update游标全部保存恢复。

资源为 **pdgpu-5090 / 8GPU / 32CPU / 256G / 1node**，原镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`、`mellow_c8204d8`、Transformers4.54.1及conda覆盖变量保持；无额外GPU smoke。

shared-first后台staging沿用：单producer复制去重原音频，全部文件/字节/源stat核验完成后，8rank在optimizer边界以collective MIN一致切到本地；两轮复用，恢复重新stage。退出先stop/reap producer，再按owner/token清理。本路线tmpfs为 `/dev/shm/mizar_s3_l21_25x3_gate_<token>`。

## 代码同步和正式提交

本地PowerShell执行以下命令将本次交付同步至GitHub。本次实现不会自动commit/push：

```powershell
Set-Location C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
git add -- RM.md code/RSmol/run_mizar_s3_l21_25x3_residual_gate_5090.sh code/RSmol/mizar_s3_l21_25x3_residual_gate tests/test_mizar_s3_l21_25x3_residual_gate.py tests/validate_mizar_s3_l21_25x3_residual_gate.py audit_reports/mizar_s3_l21_25x3_residual_gate_validation.json
git diff --cached --check
git diff --cached --stat
git commit -m "Add isolated S3 L21-25 three-pass gated training with trainable L26"
git push origin HEAD
```

远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# 仅打印配置，无环境激活/数据读取/作业申请
bash code/RSmol/run_mizar_s3_l21_25x3_residual_gate_5090.sh --check-args

# 可选：只做登录CPU预检，不提交GPU
bash code/RSmol/run_mizar_s3_l21_25x3_residual_gate_5090.sh --preflight-only

# 正式提交一次；自动CPU预检并复用有效index
bash code/RSmol/run_mizar_s3_l21_25x3_residual_gate_5090.sh
```

已有S3封存数据无需重新prepare；正式命令本身会做预检，不要求先单独运行preflight。支持原 `--data-dir`、`--model-dir`、`--output-dir`、`--index-cache-root`、`--stage-workers`。`--output-dir`指外层root，实际训练在其 `formal/` 子目录；fresh运行须用新空目录。

## 输出和恢复

默认输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
  mizar_s3_l21_25x3_residual_gate_2epochs_gbs128_5090/formal_<tag>/formal/

log: code/RSmol/log/mizar-s3-l21-25x3-gate.<tag>.JOB.log
final: formal/checkpoints/model--step-4264.ckpt
```

提交时打印确切output/log_path/admission；`metrics.jsonl`含三组实际/下一LR、loss、梯度、有效批量和shared/local读盘状态，stdout每25步及末步打印。每来源真实音频 `SOURCE_PROBE_PASS` 后才进入训练。以最终 `training_report.json` PASS、两轮完整覆盖、checkpoint重读及shell成功退出判定完成；staging进度和提交成功不等于训练完成。

仅本路线相同代码、资产、数据、环境和配方的最新完整未完成checkpoint可恢复；L11–14原实验和其它checkpoint拒绝跨路线resume。本次中断后沿原run恢复，例如：

```bash
bash code/RSmol/run_mizar_s3_l21_25x3_residual_gate_5090.sh \
  --resume-checkpoint /完整原输出目录/formal/checkpoints/model--step-2132.ckpt
```

自动推导原外层root，epoch边界从下一轮开始；日志较新行/partial tail先备份再回退。已完成4264步、较旧checkpoint回退或身份不匹配均拒绝。

## 本地验证

本地最终 **24/24项CPU测试通过，无skip**；真实官方18文件资产的454→459加载、完整479位置前反向及AdamW检查通过。可训练参数21,242,882，全部训练角色更新，原冻结状态逐值保持。

报告为 `audit_reports/mizar_s3_l21_25x3_residual_gate_validation.json`。从 `tests/` 执行，避免仓库 `code/` 遮蔽Python标准库：

```bash
cd tests
python -B validate_mizar_s3_l21_25x3_residual_gate.py --official-bundle /本地可选/Mizar159M
```

测试覆盖真实小30层Llama/native/eager/SDPA、独立非零gate公式/全位置偏置/40调用/L26和final norm单次、15次核心输出完整BPTT及冻结保持、479答案CE/推理前两logits/因果性；真实音频处理、sealed S3/index只读复用、两轮分片/LR/保存、实际trainer中断恢复、checkpoint库存和旧route拒绝、两进程Gloo及Bash单次模拟提交/失败阻断/producer生命周期。

可选官方检查使用本地真实18文件资产及454→459 strict继承，真实CED/projector/LM合成静音479位置前反向和AdamW，核验全部训练角色有任务梯度并更新、原冻结状态不变。生产代码无测试兼容补丁；已有torchaudio未用native扩展仅在本地测试进程显式适配。CPU测试不代表远程音频完整性、八卡NCCL、5090显存吞吐或正式收敛通过。

`parent_sources.json`固定49份原参考和S3依赖文件SHA；`reference_dependencies.json`说明已有只读S3验证依赖。`sync_files.txt`列本次交付和需随仓库已有的依赖。新运行源码身份只绑定自身包/入口/release及S3验证依赖，无需通过原训练入口转发。
