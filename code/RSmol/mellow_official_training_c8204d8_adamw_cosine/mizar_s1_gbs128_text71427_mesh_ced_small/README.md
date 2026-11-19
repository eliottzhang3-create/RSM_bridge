# CED-small + 已训练 Mizar 7–14×2–7 MeSH：S1 / GBS128

2026-11-19新增的隔离路线，复制自
`mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small`。
原20物理层/30执行音频路线、7–14×2–7文本模型/训练器及旧提交入口不改。
本地Windows只有代码；下面的权重、数据、环境及训练输出均在远程Linux。
通过GitHub同步代码后执行，不需要把远程资产下载到本地。

## 正式提交

默认精确读取用户指定的已完成文本checkpoint：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/formal_reinit1000_lr6e4_20261118_112042_3963870/checkpoint-009244
```

在代码已经commit/push到GitHub后，于远程仓库根执行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced_small_3090.sh
```

需要指定输出目录时，必须使用新的空目录；目录为run root，脚本会再建立`formal/`：

```bash
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced_small_3090.sh \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_text71427_mesh_3090/formal_$(date +%Y%m%d_%H%M%S)
```

入口在登录节点只做CPU JSON/完成标记/CED配置校验及索引准备，然后通过`vc submit`申请
`pdgpu-3090 / 单节点8 GPU / CPU64 / 256G`，使用原项目镜像、`mellow_c8204d8`环境。
GPU任务在计算节点通过torchrun启动。正式入口直接formal，无独立smoke或审查报告前置要求。
完整job名含后缀小于60字符。支持与母本相同的路径覆盖和`--stage-workers`参数。

## 模型与初始化

- 显式加载`MizarText7_14x2_7MeshForCausalLM`，不注册全局AutoModel覆盖。
- 28物理层，7前段 + 14共享中段×2 + 7后段，共42次执行；零基schedule为
  `0..6, 7..20, 7..20, 21..27`。hidden576、vocab49152、embedding/head绑定。
- 五槽MeSH覆盖完整683位置。前段后、每轮中段后各一组write/read，共六模块、12参数张量、17,310参数。
  保留原模型的查询位置、memory初始化/写读规则；已训练routers完整继承，不重新初始化。
- 9244是来源步数。仅加载其模型/config/tokenizer，音频optimizer step0起重新训练。
  不读取`training_state.pt`，不继承文本AdamW、scheduler、Scaler、RNG或数据游标，不需重新转换。
- 校验来源JSON和完成标记；HF loading_info不得有missing/unexpected/mismatched/errors。
  加载后逐张量核对全部safetensors与模型值精确一致，记录12个来源router SHA256。
  只允许safetensors因embedding/head绑定省略一个别名。
- CED-small资产固定`/hpc_stor03/sjtu_home/jinwei.zhang/models/CED_small`，严格离线加载，冻结全部参数且保持eval。
- mapper从头初始化并训练。结构保留官方`freq_merge_126`：384维×4频率格，输出126×576；
  dropout0.05、resample=False，外置projection为Identity。完整LM和routers全量训练。

## 输入与数据协议

**按用户最终确认，严格保留S1训练端的EOS与中心裁剪。**
公开推理入口的无prompt EOS、取前20秒规则不用于本训练。
只读核对来源为本地Mizar源码的`runtime/stage1/data/audiotext_dataset.py`、
`pipeline/build_ced_cache.py`和`pipeline/prepare_inference.py`，参考文件SHA记录在`reference_audit.json`。

- prompt与answer均先追加` <|endoftext|>`，再按256/300截断、固定padding；不强行补回截断的EOS。
  保持原text_pair随机模板、小写及CoT保留规则。
- PAD为已有`!`，注册后词表大小/ID不变；EOS与音频文本SEP均为ID0。
- 单原音频mono、resample32k，长音频中心裁剪20秒，短音频右补零。
  前后10秒只是原接口容器，CED前重新拼为完整20秒，保留原整段mel/BN/1012帧分段和逐样本dB处理。
- 明确的两条原录音才分别中心裁剪/补齐10秒再拼接。历史`filepath2`可能是预处理后半段，不能据此判为双原录音。
- `audio126 + SEP1 + prompt256 + answer300 = 683`，prefix383。
  LM的attention_mask/labels仍为None、use_cache=False，固定prompt/answer PAD仍可见。
- 外部answer-only CE，logits382..681对应300个answer目标，包含未截断EOS，排除PAD。
  每rank每microbatch有效answer-token mean，再对GA4和DDP取均值；不改成全局token加权。

数据保持原hybrid共1,477,492行及原source counts：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv
```

保留MusicCaps、全部SpeechCraft splits、原raw resolution、seed1234及rank连续分片。

## 训练合同

| 项目 | 值 |
|---|---|
| Batch | 8 GPU × micro4 × GA4 = GBS128 |
| Epochs / steps | 3 / 每epoch11,542 / 总34,626 |
| 每epoch丢弃尾部 | 116行 |
| 最大LR | 全LM/router3e-4，mapper6e-4 |
| Scheduler | 原LambdaLR；warmup1,731步；cosine floor0.1；step0至少0.1×max LR |
| AdamW | betas(0.9,0.999)，eps1e-8，WD0.01；bias和ndim≤1不衰减 |
| 精度 | FP32 forward/参数；原GradScaler；无autocast、无TF32 |
| 梯度 | 原GradNormTracker(1,10,2.5,0.995)，保留原异常终止 |
| Seed | 1234；各rank1234+rank |

来源文本run名中的`lr6e4`仅描述文本训练，音频LM最大LR仍按母本为3e-4。
本次不继承文本run的BF16/GBS1024/数据管线。

CPU索引完成后，计算节点立即通过共享路径读音频，同时默认4线程后台复制原始字节；
全部完成后，在optimizer边界以8rank MIN共同切换到本地。保留producer健康检查、精确游标及失败处理。
新暂存目录以`/dev/shm/mizar_s1_gbs256_text71427_mesh_ced_small_`开头：
其中`gbs256`仅为母本安全守卫前缀，实际batch合同为GBS128。
清理前先停止/reap producer，并校验owner/token；不把暂存路径纳入恢复身份。

## 保存、恢复与诊断

默认输出run root：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_text71427_mesh_3090/formal_<timestamp>
```

checkpoint位于`<run-root>/formal/checkpoints/`，六等分保存：
`5771 / 11542 / 17313 / 23084 / 28855 / 34626`；最后另存`model--epo-3.ckpt`。
checkpoint保留模型、optimizer、scheduler、GradScaler、GradNorm、8rank RNG及下一个数据游标。

中断后用**同一个run root**和这条音频路线的完整checkpoint恢复，例如：

```bash
RUN_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs128_text71427_mesh_3090/formal_REPLACE_WITH_ACTUAL_RUN
bash code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced_small_3090.sh \
  --output-dir "$RUN_ROOT" \
  --resume-checkpoint "$RUN_ROOT/formal/checkpoints/model--step-5771.ckpt"
```

`--text-model-dir`始终表示文本权重初始化来源；`--resume-checkpoint`只接受本音频路线的完整formal状态。
恢复严格匹配架构、代码及共享依赖、来源资产、数据和训练合同，拒绝旧GBS256/旧骨干checkpoint。

正式任务内部自动记录来源继承/布局/实际42次执行和完整memory形状；首次反传记录两轮中段边界梯度、
12个router梯度、冻结CED及mapper/LM更新。边界梯度仅首次microbatch保留，随后释放。
报告包括`runtime.json`、`config_audit.json`、`first_recursive_backward_rank<N>.json`、
`first_gradient_rank<N>.json`、`metrics.jsonl`和最终`training_report.json`。
这些诊断是正式任务内部检查，不是额外smoke作业。

## 本地核查

```bash
python -B tests/validate_mizar_s1_text71427_ced_small.py
```

本地结果见`audit_reports/mizar_s1_text71427_ced_small_validation.json`。
58项已执行通过；31项真实张量/音频依赖检查因本机缺Torch等明确跳过。
通过项包含实际Bash+模拟vc参数流、后台复制/失败清理、CPU index及原音频/文本路线回归。
官方函数对照包含1,344次文本/RNG、8组tokenize参数、34,627个scheduler位置、10,000次GradNorm标量比较；
mapper类AST与原实现一致，35份母本/共享模型文件SHA保持。

在有完整ML依赖的环境可加`--require-tensors`，缺依赖或跳过即失败；此选项不接入formal入口。
新增真实小模型测试覆盖逐张量继承/拒绝缺router、42次执行/683位置、两轮梯度、answer shift及完整状态恢复。
本地代码检查不能证明远程权重已加载成功、GPU显存足够或34,626步已完成；实际结果由远程formal报告确认。
