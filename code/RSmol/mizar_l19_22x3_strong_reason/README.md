# 原入口：Mizar L19–22纯循环四次，完整AudioMCQ，3epochs，3090

原地更新 `run_mizar_l19_22x3_strong_reason_5090.sh` 及本包。入口/包名保留历史x3/strong_reason/5090字符串，当前实际合同为 **L19–22×4 / AudioMCQ only / 3epochs / pdgpu-3090 / 64CPU**。本地Windows维护代码，经GitHub同步；正式权重、数据和训练输出仍在远程Linux。

## 模型与训练合同

重新从官方 `/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt` 初始化，严格核验18个release文件SHA/大小，完整继承454个状态，包括已训练CED/projector及30层LM。无模型转换；不加载旧三循环训练checkpoint作为新起点，optimizer/scheduler/RNG/游标从零新建。

```text
第1–18层 → [第19–22层] × 4 → 第23–30层 → final norm / head
```

四次是总执行次数，四层共享同一组Parameter，共30物理层/42次block调用。每次完整输出直接进入下一次，所有683位置参与循环；无额外音频注入、anchor、audio_gate、LoRA或MeSH，无新增参数。原Transformer层内残差和MLP gate_proj保留。

仅第19–22层全部attention/MLP/norm参数训练，每层3,540,096，共14,160,384；其余145,139,456参数冻结且eval。完整模型159,299,840个独立参数/454状态，embedding与head绑定保持。前18层no_grad；四次完整BPTT不detach，冻结后缀23–30及最终norm/head保留输入梯度，final norm只在第30层后一次；无KV/decoder checkpointing/跨forward隐状态。

训练/数据实现对齐当前 `run_mizar_l9_12x3_strong_reason_5090.sh` 对应的四循环AudioMCQ配方：

| 设置 | 当前值 |
|---|---|
| 数据 | 完整历史AudioMCQ 720,524行/epoch，ReasonAQA=0 |
| epochs | 3 |
| Batch | 8GPU × micro4 × GA4，最大GBS128 |
| 每epoch更新 | 5,630：5,629完整128更新+12条尾批 |
| 总更新 | 16,890 |
| max/min LR | 2e-5 / 2e-6 |
| Warmup | round(16890×3%)=507，随后全局cosine |
| 保存 | 每0.25epoch后首个完整optimizer step，共12点，保留最新4份 |

LR在epoch边界不重置，最后真实update使用minLR。AdamW betas=(.9,.999)、eps1e-8、WD.01，bias/一维免decay；原GradScaler/GradNormTracker保持。FP32、无autocast、TF32off，seed1234及rank RNG1234+rank。每epoch sampler seed1234+epoch，无drop/padding重复，每个原训练行各用一次；尾批前4rank各2条、后4rank各1条、GA1。保留每卡micro有效answer token mean→实际GA mean→活跃rank mean；空rank仅零任务loss通信，不增加消费记录或任务梯度。

音频SoundFile解码、mono32k、fixed20中心裁剪/右补零后10+10，在线CED/projector。126audio+SEP1+prompt256+answer300=683，prefix383，PAD17/EOS0；原S1文本模板、空格EOT后截断且不补回EOS、可见PAD因果attention及answer-only shifted CE保持。

## 复用已生成的manifest

用户回传已成功复用（本地未独立访问服务器）：

```text
/mnt/cloudstorfs/sjtu_home/jinwei.zhang/data/Mizar/l9_12x4_audiomcq_only_v1
```

新入口和合同默认指向上述路径。已有manifest时，入口执行 **verify-only** 校验seal、720524七来源计数及四个文件SHA/大小，直接继续CPU index，不扫描旧hybrid、不重抽样、不重建/改写数据。数据中的l9_12名称是原生成路线名称，不影响L19–22训练。

数据从旧完整训练 `s1_stage1_hybrid_filtered.json` 保留全部AudioMCQ，含strong/weak、直接答案/CoT，字段/答案/重复视图/原顺序保持，排除756968 ReasonAQA。来源AudioCaps113844、Clotho4798、MusicCaps3221、CompA-R249161、LP-MusicCaps-MTT19632、SpeechCraft287727、TACOS42141。沿旧hybrid audit exists路径，旧已用音频缺失/空/冲突报错；不猜映射、不静默过滤。

包内prepare_manifest.py与当前L9–12参考逐字节保持，使用相同数据format和preparer SHA，支持复用同一sealed结果。若显式使用全新不存在的数据目录，入口仍可按相同历史source/audit生成；本次已生成数据无需这一步。`--prepare-only` 对已有数据仅核验后退出，不申请GPU。

## 正式提交

先将本地修改提交推送到GitHub，再在远程仓库根：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# 参数预览：无环境激活/数据读取/GPU申请
bash code/RSmol/run_mizar_l19_22x3_strong_reason_5090.sh --check-args

# 复用已生成manifest，经CPU index/release/admission校验后单次正式提交
bash code/RSmol/run_mizar_l19_22x3_strong_reason_5090.sh
```

无需重新运行旧数据准备命令。也可显式指定已生成路径：

```bash
bash code/RSmol/run_mizar_l19_22x3_strong_reason_5090.sh \
  --data-dir /mnt/cloudstorfs/sjtu_home/jinwei.zhang/data/Mizar/l9_12x4_audiomcq_only_v1
```

实际资源 **pdgpu-3090 / 8GPU / 64CPU / 256G / 1node**，原镜像 `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`、`mellow_c8204d8`与Transformers4.54.1。沿原Miniconda和RSMOL_CONDA_BASE/MELLOW_CONDA_BASE覆盖，不改共享环境，无额外GPU smoke。

默认目录：

```text
output root: /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/
             mizar_final_l19_22x4_audiomcq_gbs128_3epochs_3090/formal_<tag>/
training:    <output root>/formal/
index:       .../mizar_final_l19_22x4_audiomcq_gbs128_3epochs_3090/index_cache/<signature>/
log:         code/RSmol/log/mizar-audiomcq19-x4.<tag>.<JOB>.log
tmpfs:       /dev/shm/mizar_l19_22x4_mcq_<token>
```

旧三循环输出不覆盖。`--output-dir`指外层root，内部追加formal。提交会打印确切log_path、manifest、admission和output。默认4worker单后台producer仅复制当前AudioMCQ去重原文件，先shared立即训练、完整文件/字节验证后8rank在optimizer边界MIN共同切本地；3epochs复用一次staging，恢复时重建。退出先stop/wait回收producer再按owner/token清理。CPU READY_FOR_GPU或vc提交成功不是训练完成。

保存点：

```text
1408,2815,4223,5630,
7038,8445,9853,11260,
12668,14075,15483,16890
```

最终保留12668/14075/15483/16890；最终checkpoint `formal/checkpoints/model--step-16890.ckpt`。完成依据为`training_report.json` PASS、epochs3/optimizer_steps16890/三epoch完整覆盖与最终重读一致，加shell成功退出。metrics含epoch/epoch_update/epoch_fraction、实际LR、loss、尾批与读盘状态，epochN_rankR记录覆盖。

## 完整恢复与历史checkpoint

仅接受本L19–22×4/AudioMCQ/3epochs相同代码/数据/环境/来源与配方的完整checkpoint。旧L19–22×3及L9–12、L14–17 checkpoint均拒绝跨路线恢复。此次从官方final新训练；本次中断后使用最新完整checkpoint，沿用原外层root：

```bash
bash code/RSmol/run_mizar_l19_22x3_strong_reason_5090.sh \
  --resume-checkpoint /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/my_l19_22x4_run/formal/checkpoints/model--step-5630.ckpt
```

恢复全部模型/具名AdamW moments/scheduler/Scaler/GradNormTracker/8rank RNG和下一个epoch/update游标；epoch边界从下一epoch0开始，LR不重置。日志较新行/partial tail原样备份后回退，完整坏行拒绝；已完成16890步不续训。

已有mizar_core3_eval固定的是三循环历史源码库存；本次不将新四循环合同注册到该旧评测器。旧三循环checkpoint评测须沿原匹配Git源码/缓存快照，新四循环评测另行适配，不放宽旧身份。

## 验证与代码同步

当前报告 `audit_reports/mizar_l19_22x4_audiomcq_validation.json`，旧 `mizar_l19_22x3_strong_reason_validation.json` 保留历史意义。测试沿原tests文件名，覆盖真实小Llama/native/独立42次递推与16次core输出梯度、冻结后缀梯度和参数保持、三epoch完整小模型训练/epoch边界resume、生产720524分片/尾批/LR/保存、真实Gloo GA4/尾GA1/空rank、seal/index/已有manifest只读复用、实际CPU admission/错旧route拒绝、Bash3090/64CPU单提交/先reap后cleanup与实际tmpfs守卫。

最终本地21/21项CPU测试通过、无skip。已有本地官方release实际通过18文件SHA/454继承、原576维30层native逐值对照、真实CED/projector+683token四循环合成静音CPU前反向与AdamW后冻结保持。CPU测试不代表远程CUDA/NCCL、3090显存吞吐或正式收敛；远程已生成manifest信息来自用户日志，不冒充本地独立远程校验。生产代码无测试适配，Windows本地已有torchaudio未用native扩展仅测试进程适配。

```bash
cd tests
python -B validate_mizar_l19_22x3_strong_reason.py --official-bundle /path/to/Mizar159M
```

`source_guard_x4.json`记录本任务范围外4258份文件基线，并登记另一会话L14–17路线的精确并发变更；旧`source_guard.json`完整保留，`sync_files.txt`列交付文件；README/RM更新保留旧RM全文，原L9–12参考及其他路线不改。已移除不再使用的旧strong+Reason previous_training.py。LICENSE/NOTICE保留。

本地PowerShell提交命令（未自动执行）：

```powershell
Set-Location C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
git add RM.md code/RSmol/run_mizar_l19_22x3_strong_reason_5090.sh code/RSmol/mizar_l19_22x3_strong_reason tests/test_mizar_l19_22x3_strong_reason.py tests/test_mizar_l19_22x3_previous_training.py tests/validate_mizar_l19_22x3_strong_reason.py audit_reports/mizar_l19_22x4_audiomcq_validation.json
git diff --cached --check
git diff --cached --stat
git commit -m "Train Mizar L19-22 with four shared passes on AudioMCQ using 3090"
git push origin HEAD
```
