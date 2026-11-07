# 官方 Mizar final：5 + 20×2 + 5，仅 input embeddings 重注入

本目录从 `mizar_official_recursive_5_20x2_5` 复制隔离，提供独立的审计、CPU 转换与重载实现。原 serial84 + LoRA 路线不变。本地 Windows 只维护代码，经 GitHub 同步；权重、生产转换和训练均留在远程 Linux。

## CPU 转换命令

本地代码提交并推送后，在远程登录节点执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5_input_only.sh convert --seed 0 --threads 8
```

该脚本直接在终端运行，激活既有 `mellow_c8204d8`，在导入 Torch 前清空 `CUDA_VISIBLE_DEVICES` 并设置离线模式。全程 CPU-only，不申请 GPU、不提交 `vc` 作业、不读取训练数据。需要既有 Torch/torchaudio/Transformers 4.54.1 环境及完整本地模型资产，不自动安装依赖或下载文件。CPU 转换包含完整模型前向，耗时以远程实际运行情况为准。

默认来源是官方完整发行目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M
```

其中读取 `checkpoints/final/stage3_q4_seed20260905.ckpt` 和 `assets/`。直接从官方 final 初始化，不从另一条已转换或已训练的 LoRA 路线拆权重。默认新产物为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-5-20x2-5-input-reinject
```

`convert` 可用 `--source DIR --output DIR --seed 0 --threads 8` 指定独立来源/输出；已有输出目录会拒绝覆盖。可选只读审计与转换后检查：

```bash
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5_input_only.sh audit
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5_input_only.sh verify
# 可选：重新在 CPU 跑完整 tensor/前向检查；convert 本身已经强制执行保存重载检查。
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5_input_only.sh verify --tensors --threads 8
```

自定义产物的 `verify` 也需传相同 `--output DIR`。`audit` 默认只校验文件，可加 `--tensors` 做 CPU 严格加载和单遍对齐。`--report PATH` 只允许创建全新报告文件。环境根目录可通过 `MIZAR_520X2_INPUT_ONLY_CONDA_BASE` 覆盖。

## 模型与初始化

- 完整保留官方 30 个物理层和原 LM、mapper、CED 权重。prelude 为 0–4，循环模块为 5–24，coda 为 25–29；实际执行 `0..24, 5..24, 25..29`，共 50 次。循环中段两轮共享基础权重。
- 保留一套共享 `Linear(576,576)` input projection 和一套共享 `Linear(576,1)` gate。gate 对当前音频 hidden 逐 token 计算，两轮使用相同参数。
- 没有 LoRA、encoder 中间层 MLP/gate/特征捕获、memory 或 router。CED 只执行一次最终特征提取，冻结且始终 eval；原 final 已训练的 `freq_merge_126` mapper 继续训练。
- 新投影 weight 沿原 `nn.Linear` 初始化，bias 为零；gate weight/bias 均为零。保留母本的 input 分支分配顺序和隔离 CPU RNG。转换 seed 默认为 0。

每次 LM forward，先从原 mapper 输出取得音频 input embeddings `A0`，只计算一次投影：

```text
A0 = original_final_mapper(CED_final_features)       # [B,126,576]
R  = audio_linear(A0)                               # 每个 LM forward 一次

# 逻辑位置 5、25，均位于对应中段循环之前：
H_audio = H_audio + tanh(audio_gate(H_audio)) * R
```

残差只直接写入音频位置 0–125；SEP、prompt、answer 仍通过后续 decoder 层与音频交互。`R` 不 detach，梯度可回到 mapper。零 gate 关闭的是新增残差；中段仍执行两次，因此不会要求整个循环模型的输出等于原 30 层模型。

| 项目 | 数量 |
| --- | ---: |
| 官方基础独立参数 | 159,299,840 |
| 新增 projection + gate 参数 | 332,929 |
| 完整模型独立参数 | 159,632,769 |
| 官方 / 转换后 state tensor | 454 / 458 |

embedding/lm_head 保持 tied。无 KV cache、无 gradient checkpointing，参数及前向均 FP32。

## 六步转换与产物

1. 按固定官方 release manifest 校验 checkpoint、tokenizer、CED 构造代码及权重的完整 SHA256。
2. 先严格加载全部 454 份官方 state，逐 tensor 核对继承值，再注册四份新增 tensor；核对层数、参数总数、tied 权重和 trainability。
3. 关闭新增分支并执行单遍 30 层，与原 LM 数值对齐；再以合成 20 秒音频执行完整 683-token、50 次 decoder 调用，检查两次 input 注入、一次投影和 CED 布局。
4. 保存模型 state、完整配置和精确复制的 tokenizer/CED 构造资产。无训练 optimizer、scheduler 或游标。
5. 从暂存产物重建模型，严格加载全部 458 份 tensor，逐 tensor 精确相等；完整 683-token logits 在保存前后精确相等。
6. 所有检查成功后发布完整目录。包含 CloudStor 不支持 `RENAME_NOREPLACE` 时的兼容路径；已有目标始终拒绝覆盖。

产物包含 `model.ckpt`、`config.json`、`tokenizer/`、`ced-small/`、`manifest.json`、`source_audit.json`、`conversion_report.json` 和许可证。模型 Python 源码、官方 release manifest 和产物字节均绑定到审计身份；模型/合同源码变化后需重新转换。该目录是训练初始化，不是 full resume checkpoint；`training_intent.implemented=False` 延续转换元数据的语义，正式训练由下方独立训练器负责。

失败保留 `*.staging-*` 和其内 `failure.json`。若只在第六步失败，且第五步的 458 tensor、strict load、683-token logits 记录完整，可用 CPU 的续发布入口：

```bash
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5_input_only.sh publish \
  --staging /path/to/Mizar159M-5-20x2-5-input-reinject.staging-XXXX
```

自定义目标需追加原 `--output DIR`。续发布只做文件及完成记录校验，不重跑 tensor，不复制模型权重；不得拿原 serial84 的 staging 复用。诊断备份独立保留。进程异常终止留下的 `.conversion.lock` 需要确认无对应转换进程后人工处理。

## 训练和本地验证

正式入口及完整恢复见 [训练说明](../mellow_official_training_c8204d8_adamw_cosine/mizar_official_final_s1_gbs128_5_20x2_5_input_only/README.md)。转换完成后即可直接提交一个正式训练任务，不自动提交前置诊断作业。

仓库根目录可手动运行：

```bash
python -B tests/validate_mizar_input_only.py
# 在已有完整依赖的环境要求真实小尺寸 CPU tensor 检查，任何缺依赖/skip 均失败：
python -B tests/validate_mizar_input_only.py --require-tensors
```

当前本地 63 项实际检查通过，12 项真实 tensor 检查因缺 Torch/Transformers/audio 依赖明确跳过；报告为 `audit_reports/mizar_input_only_validation.json`。检查覆盖文件校验/配置保留/发布恢复、父路线 45 文件 SHA 保护、数据及更新循环对照、staging 线程协议和真实 Bash 夹具。可选真实 tensor 测试覆盖完整 30 物理/50 逻辑小模型、保留 input 分支初始化一致性、公式和梯度、严格重载及 optimizer/RNG 恢复。没有读取远程生产权重或执行真实 GPU/DDP，远程转换结果以实际报告为准。
