# 四个指定 checkpoint 的 MMAU / MMAR / MMSU 完整评测

本地 Windows 只维护代码，通过 GitHub 同步到远程 Linux；权重、数据和 GPU 推理均在远程。
提交入口为 `code/RSmol/run_mizar_selected4_benchmarks_5090.sh`。
每个目标固定训练路线、checkpoint、runtime、step17313 和训练 geometry `8,8,4`。

## 目标及 checkpoint 对应

以下 run 均位于 `/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/`。
最终权重为 `<run>/formal/checkpoints/model--step-17313.ckpt`，配置为 `<run>/formal/runtime.json`。

| TARGET | 训练入口（均在 code/RSmol） | run |
|---|---|---|
| `init0904` | `run_mizar_s1_gbs256_recursive_5_10_5_ced_small_init0904_5090.sh` | `mizar_s1_ced_small_gbs256_recursive_5_10_5_init0904_5090/formal_20261111_130037935014521` |
| `text_epoch1` | `run_mizar_s1_ced_small_text_epoch1_5090.sh` | `mizar_s1_ced_small_text_epoch1_gbs256_5090/formal_20261112_103104688616376` |
| `text_epoch1_serial48_lora_r8` | `run_mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_5090.sh` | `mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_gbs256_5090/formal_20261112_103717262469368` |
| `text_lora_epoch1_serial48_r8` | `run_mizar_s1_ced48_text_epoch1_loop_lora_r8_5090.sh` | `mizar_s1_ced48_text_epoch1_loop_lora_r8_5090/formal_20261112_111357380584730` |

四条均为非 MeSH、20 个物理层、`5 + 10×2 + 5` 共30次执行。前两条复用普通递归 runtime，
后两条复用串行 CED4→8 / audio reinjection / 双 loop LoRA r8 runtime。
`ced48` 表示 CED4→8。

第三条在音频 step0 新建 LoRA；第四条继承文本 epoch1 已训练的 A/B，使用其专属模型类和合同，
构造时先注册 A/B 再严格加载文本权重，随后注册音频模块，禁止再次启用/重置 LoRA。
评测严格恢复最终音频 checkpoint 的完整 state_dict，包括训练后的 LM、CED、mapper、各音频分支和 A/B。
文本初始化的来源核对不会被用于覆盖最终音频权重。
第四条报告将 4,406,019 个分支参数分为 2,777,859 个新音频参数和 1,628,160 个继承的 LoRA 参数；
第三条的 4,406,019 个参数均在音频训练起点新建。这些字段描述初始化来源，不代表最终参数仍是初始值。

## 推理和评分

沿用已有 FP32、关闭 TF32、seed1234、eval/frozen、单卡 batch1；32kHz 单声道，整段居中裁剪或右补零
至20秒再分成两个10秒；prompt256、prefix383、greedy300、完整重算、无 KV cache/attention mask。
MMAU test mini 和 MMAR 各1000题，保留既有小写题面及各自官方 scorer；MMSU5000题，使用已确认的
官方大小写题面和原样官方 scorer。保留原始生成内容，不增加字母提取或 EOS 清洗。
MMSU 实际评分分母不足5000时保留官方分数并判定整轮未完整通过。

沿用 `pdgpu-5090`、1GPU / 8CPU / 32G / 1node、原镜像和 `mellow_c8204d8` conda 环境及离线设置。
默认数据目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini
/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR
/hpc_stor03/sjtu_home/jinwei.zhang/data/MMSU/MMSU data
/hpc_stor03/sjtu_home/jinwei.zhang/data/MMSU/MMSU code/mmsu_evaluation.py
```

原始文本 checkpoint、tokenizer 和 CED 资产也须继续位于各自训练合同记录的路径，正式 loader 会检查其身份。

## 远程提交

先将下面“本地 Git 同步”的改动推到 GitHub，再在远程执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# 每条命令提交三个独立的完整评测作业，共12个。
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh init0904 all
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh text_epoch1 all
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh text_epoch1_serial48_lora_r8 all
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh text_lora_epoch1_serial48_r8 all
```

`all` 会依次尝试提交三个独立作业；某次提交失败仍尝试剩余两项，最终以非零状态提示失败。
也可以分别运行以下12条命令，**与上面的4条二选一，避免重复提交**：

```bash
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh init0904 mmau
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh init0904 mmar
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh init0904 mmsu
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh text_epoch1 mmau
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh text_epoch1 mmar
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh text_epoch1 mmsu
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh text_epoch1_serial48_lora_r8 mmau
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh text_epoch1_serial48_lora_r8 mmar
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh text_epoch1_serial48_lora_r8 mmsu
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh text_lora_epoch1_serial48_r8 mmau
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh text_lora_epoch1_serial48_r8 mmar
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh text_lora_epoch1_serial48_r8 mmsu
```

仅打印目标绑定及 shell 引用后的命令，不提交、不读取权重/数据，也不验证路径存在性：

```bash
bash code/RSmol/run_mizar_selected4_benchmarks_5090.sh init0904 all --check-args
```

单 benchmark 可传 `--dataset-dir`、`--evaluation-script`、`--output-dir`，MMSU另可传 `--mmsu-root`；
`all` 仅接受 `--check-args` 和 `--parquet-batch-size`。checkpoint/runtime/route/step/geometry 固定，
预设会覆盖残留的模型环境变量。其他目标仍可使用原通用提交入口。

每个 run 的输出独立写入：

```text
<run>/eval/mmau_test_mini_step17313/
<run>/eval/mmar_step17313/
<run>/eval/mmsu_step17313/
```

断点只在 checkpoint、数据、代码和协议身份一致时恢复。此次扩展改变共享 runtime 的代码指纹；
已有未完成的旧评测应沿用原代码版本继续。保留旧结果和身份检查，不混合不同代码版本的预测。

## 本地验证与 Git 同步

```powershell
cd C:\Xlance\GZ_bridge\Recursive_SALM\RSM_bridge
python -B tests/validate_mizar_s1_ced_mmsu_eval.py --report audit_reports/mizar_selected4_eval_validation.json
```

验证报告明确区分本地 fixtures、真实依赖检查的跳过项与生产推理；不代表远程权重加载或GPU评测已通过。
本次本地验证：142项中133通过，9项因缺少 Torch / Transformers / Safetensors、SoundFile 或 PyArrow
等依赖明确跳过。报告为 `audit_reports/mizar_selected4_eval_validation.json`。
`reference_audit.json` 保存105份训练源码摘要和4份获准扩展的评测文件前后摘要；
原 MMSU 历史基线保持原样，旧默认分支、registry/constants、推理函数和原结构检查另做 AST/源码校验。

确认本地验证报告后，在本地执行以下命令同步代码：

```powershell
git add -- RM.md code/RSmol/run_mizar_selected4_benchmarks_5090.sh code/RSmol/mizar_selected4_eval code/RSmol/run_mizar_s1_ced_small_mmau_eval_5090.sh code/RSmol/run_mizar_s1_ced_small_mmar_eval_5090.sh code/RSmol/scripts/mizar_s1_ced_recursive_eval_runtime.py code/RSmol/scripts/mizar_s1_ced_multilayer_eval_runtime.py tests/test_mizar_selected4_eval.py tests/test_mizar_selected4_eval_tensor.py tests/test_mizar_s1_ced_mmsu_eval.py tests/test_mizar_s1_ced_recursive_eval.py tests/test_mizar_s1_ced_multilayer_eval.py tests/test_mizar_s1_ced_eval_extensions.py tests/test_mizar_s1_ced_input_ca_eval.py tests/validate_mizar_s1_ced_mmsu_eval.py audit_reports/mizar_selected4_eval_validation.json
git commit -m "Add MMAU MMAR MMSU evaluation for four Mizar checkpoints"
git push
```
