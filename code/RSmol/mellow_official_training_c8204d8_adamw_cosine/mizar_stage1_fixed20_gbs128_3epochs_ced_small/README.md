# Mizar S1 / 原始 SmolLM2-135M + CED-small / GBS128 / 3090

本路线从 `mizar_stage1_fixed20_gbs256_3epochs_ced_small` 完整复制隔离。
唯一训练配方变化是 microbatch 8→4，GBS256→128，以及由此派生的步数与warmup。
按用户补充，提交资源使用 pdgpu-3090、64CPU、256G、8GPU、单节点。
原GBS256训练、其入口、模型及全部既有评测源码保持。
本地Windows维护核心代码，经GitHub同步到远程Linux；权重、数据和训练产物留在服务器。

## 正式配置

| 项目 | 配置 |
| --- | --- |
| 文本初始化 | `/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2`；原始30个独立层，hidden576，全量训练 |
| 音频 | 原CED-small资产离线严格加载、冻结且始终eval；在线FP32编码 |
| mapper | 原官方freq_merge_126、dropout0.05、外置Identity，随机初始化并训练 |
| batch | 8 GPU × 每卡micro4 × GA4 = GBS128；前三次no_sync，第四次DDP同步 |
| 数据 | 原hybrid 1,477,492行；原manifest、路径映射、来源范围和shuffle规则 |
| 更新 | 3 epochs；每epoch11,542更新；总34,626；warmup1,731，沿用原round(5%)公式 |
| 每epoch样本 | 消耗1,477,376行，丢弃116行 |
| LR | LM max3e-4 / min3e-5；mapper max6e-4 / min6e-5；原LambdaLR、初始factor0.1 |
| optimizer | 原AdamW betas(0.9,0.999)、eps1e-8、矩阵WD0.01、bias/ndim≤1 WD0 |
| 梯度及精度 | 原GradNormTracker、GradScaler；全FP32、无autocast、TF32关闭 |
| 序列 | audio126 + SEP1 + prompt256 + answer300 = 683；prefix383 |
| 文本和loss | 原S1模板、EOS追加后截断、PAD为! / ID17、SEP/EOS0、attention_mask=None、answer-only shifted CE |
| loss归约 | rank内microbatch有效token均值 → 四次GA均值 → DDP均值 |
| GPU提交 | pdgpu-3090 / 64CPU / 256G / 8GPU / 1node；原镜像和mellow_c8204d8 |

不加入MeSH、递归、LoRA或音频checkpoint权重初始化。batch变化会改变梯度统计与训练轨迹；
其余模型、音频处理、数据读取和优化计算保持母本。不能把新训练描述成原GBS256的数值等价运行。

音频保持mono/32k、单录音fixed20中心裁剪/右补零、真实双录音各10秒；
重拼完整20秒后计算原CED整段mel、冻结BN、1012帧分段及逐样本dB。
mapper的正确时频排列、冻结CED与可训练mapper之间的梯度边界原样保留。
数据读取仍为登录CPU index → GPU共享原始音频立即训练 → 默认4workers后台复制原始字节 →
全部完成后optimizer边界八rank MIN collective一致切换到本节点`/dev/shm`。
退出仍先停止/reap producer，再校验owner/token清理。
新临时目录使用`/dev/shm/mizar_s1_gbs256_ced_small_gbs128_<token>`，保留母本守卫要求的历史前缀；
这里的gbs256是安全路径前缀，实际训练geometry由contract明确记录为8/4/4/128。
不新增smoke、额外短训练或人工报告放行条件；原资产检查和CPU索引准备保持。

## GitHub同步

本地仓库根目录可只暂存本任务文件，先查看diff再按正常流程commit/push：

```powershell
git add -- code/RSmol/run_mizar_s1_ced_small_gbs128_3090.sh code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs128_3epochs_ced_small tests/test_mizar_s1_ced_gbs128_route.py tests/test_mizar_s1_ced_gbs128_shared_start.py tests/test_mizar_s1_ced_gbs128_offline_config.py tests/test_mizar_s1_ced_gbs128_staging_roots.py tests/test_mizar_s1_ced_gbs128_behavior.py tests/test_mizar_s1_ced_gbs128_tensor.py tests/validate_mizar_s1_ced_gbs128.py
git add -p -- RM.md
git diff --cached --check
git diff --cached --stat
git commit -m "Add isolated SmolLM2 CED GBS128 training on 3090"
git push
```

包内`.gitignore`显式放行`reference_audit.json`、`parent_reference_audit.json`和
`original_files_sha256.json`，确保普通Git同步携带审计来源。
`parent_reference_audit.json`只表示母本历史审计；本新路线当前审计为`reference_audit.json`。
本地验收报告写入仓库根`audit_reports/mizar_s1_ced_gbs128_validation.json`。

## 正式提交命令

代码推送GitHub后，在远程登录节点的仓库根目录执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_ced_small_gbs128_3090.sh
```

入口直接提交一个formal作业。默认读取原数据与资产：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2
/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
```

默认输出根为：
`/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_3090/formal_<timestamp>`。
运行产物位于其下`formal/`；index-cache、job及log命名也独立。
单次明确指定新输出目录可用：

```bash
bash code/RSmol/run_mizar_s1_ced_small_gbs128_3090.sh \
  --stage-workers 4 \
  --output-dir "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_3090/formal_$(date +%Y%m%d_%H%M%S%N)"
```

入口仍支持`--resolved-paths`、`--text-model-dir`、`--ced-model-dir`、`--index-cache-root`与
`--stage-workers`。默认文本来源保持原始SmolLM2，不会自动寻找已有音频权重或latest checkpoint。
日志位于远程仓库`code/RSmol/log/mizar_s1_ced_small_gbs128.<timestamp>.JOB.log`。

## Checkpoint和完整恢复

按母本相同的六个等进度保存点，保存5771、11542、17313、23084、28855、34626步，
最终另存`formal/checkpoints/model--epo-3.ckpt`。
checkpoint完整保存模型（含冻结CED）、AdamW、scheduler、GradScaler、GradNormTracker、
8rank RNG、epoch与下一optimizer数据游标。恢复不会重新初始化已训练mapper。

仅接受本新路线`ced_small_mizar_s1_fixed20_gbs128_v1`、相同源码/数据/资产/依赖/配方的未完成正式checkpoint。
旧GBS256、MeSH或其他音频checkpoint均不能作为本路线完整resume。
中断后沿用原run root（不要额外追加`formal`）并传对应checkpoint，例如：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_3090/formal_实际时间戳
bash code/RSmol/run_mizar_s1_ced_small_gbs128_3090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt" \
  --stage-workers 4
```

这会继续至总计34,626步，不是额外训练34,626步。每次作业重新staging并从shared启动。
已训练完成的checkpoint无需训练resume；原GBS256专用评测身份与新合同不同，不能直接套用。
本任务交付正式训练，不更改任何既有评测器或历史结果。

## 本地验证与远程证据边界

```bash
python -B tests/validate_mizar_s1_ced_gbs128.py
```

验证包含母本/其他tracked源码LF SHA保护、复制文件差异白名单、完整trainer计算继承、
实际runtime配置生成、完整LR区间及原官方文本/tokenizer/mapper/GradNorm对照、
新geometry/保存点、实际Bash模拟vc参数、后台复制/集体切换/失败清理、真实路径守卫与完整恢复接口。
验证器缺少真实Torch/音频依赖时逐项记录跳过，不把fixture当作真实autograd或GPU通过。
有依赖时可额外使用`--require-tensors`要求真实CPU tensor检查无跳过；不作为formal前置。
本地官方源码目录可用`--reference-root`覆盖；未提供源码时只运行独立母本审计，不冒称官方对照已执行。
远程环境可在本包目录直接运行`python runtime_checks.py`做原toy CPU张量诊断，无需生产资产或CUDA。
正式权重加载、3090显存、真实8GPU训练与完整恢复以远程日志、checkpoint和training_report为准。

本轮本地验收：56项执行通过、15项真实CPU张量检查因缺少Torch/Torchaudio/Transformers/SoundFile/Safetensors明确跳过；
Python3.10语法、2份Bash及4块文档命令语法、真实Bash模拟提交和Git同步清单通过。
21份母本文件normalized-LF一致，3,166份既有tracked源码SHA保持。官方对照包含mapper AST、
文本/RNG 1,344组、tokenizer 8组、LR 34,627个位置（最大误差0）、GradNorm 10,000组。
Windows沙箱阻止测试临时目录后，经批准在沙箱外完成同一本地离线回归；未读取远程资产或提交真实作业。
