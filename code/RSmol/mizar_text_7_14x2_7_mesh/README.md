# Mizar final文本权重：7–14×2–7、五槽、三组MeSH

独立路线，复制并修改既有MeSH模型/训练基础和Mizar文本转换逻辑。
原模型、转换器、训练器和提交入口保持不变；新代码不导入旧路线运行模块，
只读取共同的官方release manifest。不注册全局AutoModel覆盖；使用本路线显式类加载。

## 结构与初始化

源文件固定为：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt
```

权重仅提取`caption_decoder.lm.*`，文本assets仅提供config/tokenizer/generation配置。
先严格加载完整30层文本模型，再按**一基编号删除第3、18层**：

| 部分 | 目标物理层（一基） | 来源层（一基） |
|---|---|---|
| 前段 | 1–7 | 1, 2, 4, 5, 6, 7, 8 |
| 中段 | 8–21 | 9–17, 19–23 |
| 后段 | 22–28 | 24–30 |

28个独立DecoderLayer；中14层调用两次，共42次执行。
零基schedule：`0..6, 7..20, 7..20, 21..27`。
按项目约定，config的`num_hidden_layers=42`，`recursive_layer_count=28`。
四种schedule字段及零基/一基来源映射同步写入并严格校验。

五槽memory形状`[B,5,L,576]`，slot0初始化为原始embedding，其余为零。
前段后、两次中段后各一组write/read，共六个独立`Linear(576,5,bias=True)`，
12个新增参数张量、17,310个参数。router softmax为FP32；加法写入、加权读取，
全路径反传。memory为单次forward激活，不持久化，不跨forward累积。

过渡router查询沿用项目MeSH合同：前段输出；循环router查询为该轮core输入，
写入值为该轮core输出。官方实现的过渡查询是embedding，本路线保留项目的查询约定。
embedding不乘sqrt(hidden)，不新增norm、残差缩放或辅助loss。

**router初始化采用官方真截断正态**：`nn.init.trunc_normal_`，
`std=sqrt(2/(5*hidden_size))`，边界`[-3std,+3std]`，bias零，不做深度缩放。
转换时在HF所有post_init之后，用独立seed0重新初始化六router，避免构造器随机数消耗影响。
这不同于旧本地模型的normal+clamp；旧实现保持不变。加载checkpoint时保留已保存router。
参考：[官方代码](https://github.com/LivingFutureLab/MeSH/blob/main/src/llamafactory/model/modeling/modeling_gpt_neox.py)、
[论文](https://arxiv.org/html/2510.07739v2#S3.SS2)。

## 远程转换命令

本地只有代码。经GitHub同步到远程后，在远程CPU运行：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/scripts/convert_mizar_text_7_14x2_7_mesh.sh
```

默认输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-text-7-14x2-7-mesh-5slot
```

已有目录会被拒绝。需要新名称时显式传入`--output-dir /absolute/new/path`。
转换在CPU、离线、8线程完成，核对官方checkpoint SHA256、大小、454 tensor inventory，
以及文本assets哈希。全部继承参数逐张量精确相等；新增随机参数只允许12个router tensor。
同时核对embedding/lm_head源值一致和目标Parameter绑定。

私有staging中保存FP32 HF safetensors、config和tokenizer，严格重载后检查全部tensor、
42次轨迹、logits及tokenizer一致，再原子发布。失败保留staging/failure.json，
不会覆盖源或已有产物。`recursive_conversion_metadata.json`和`conversion_complete.json`
记录来源、层映射、router初始化及验收结果。删层后模型与原30层模型不要求logits相同。

## 正式文本训练

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
bash code/RSmol/run_stage4_mizar_text_7_14x2_7_mesh_training_5090.sh
```

该入口通过`vc submit`申请`pdgpu-5090`、单节点8 GPU、CPU32、内存256G，
沿用项目镜像和远程`rsmol`环境。GPU任务必须经该入口提交，不能在登录节点直接torchrun。
目标运行环境为Python3.10、Torch2.11.0+cu128、Transformers4.54.1；实际版本写入诊断记录。
正式入口可直接提交，不要求其它路线的smoke报告。纯预览命令不提交：

```bash
bash code/RSmol/run_stage4_mizar_text_7_14x2_7_mesh_training_5090.sh --dry-run
```

| 项目 | 固定配置 |
|---|---|
| 初始状态 | 本路线转换产物；新optimizer/scheduler/RNG/数据游标 |
| Batch | 8 GPU × micro8 × GA16 = GBS1024 |
| 精度 | FP32参数/AdamW状态；BF16 autocast仅覆盖forward；FP32 CE/softmax；无Scaler/TF32 |
| 数据 | `data/SmolLM2-135M-10Bsubset`或其`data`下精确85片Parquet，直接流读persistent存储 |
| 分片 | 排序后seed打乱，按rank连续分配，分片数11/11/11/11/11/10/10/10 |
| 分片内 | buffer4096打乱，过滤空文本；独立样本、不packing、不补EOS或special tokens |
| Context | 1024，动态右padding，padding label=-100，分片不足microbatch的尾部丢弃 |
| 一轮定义 | 9244次optimizer更新；不是每条数据恰好一次；允许rank内数据轮转 |
| LR | max8e-4、min8e-5、warmup463，沿用Mizar completed-step cosine公式 |
| AdamW | betas(0.9,0.95)、eps1e-8；矩阵WD0.1，bias/norm/低维WD0 |
| Loss | 全文本causal CE；按所有rank和整个GA窗口的有效shifted token数归一化 |
| 梯度 | 每micro反传CE sum，DDP平均后乘world/global_tokens；clip L2=1 |
| DDP | broadcast_buffers=False、find_unused_parameters=False；前GA-1次no_sync |
| 保存 | 每500步及最终9244步；本次输出仅保留最近三个完整checkpoint |
| Seed | 默认0；各rank为seed+rank |

CUDA必须支持BF16，不能静默回退。前7、中14、后7、embedding/head/norm及routers全部训练。
数据顺序采用Mizar文本路线；WD分组采用MeSH路线。残差/memory保留FP32语义，
BF16前向不代表所有激活都减半，显存以远程实测为准。

显存不足时显式选择另一套GBS1024配置（禁止自动切换）：

```bash
bash code/RSmol/run_stage4_mizar_text_7_14x2_7_mesh_training_5090.sh \
  --micro-batch-size 4 --gradient-accumulation-steps 32
```

micro变化会改变动态padding和分片尾部处理，不保证数值轨迹相同。
所有入口参数、CLI覆盖及配置写入`run_manifest.json`。默认输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/formal_<timestamp_pid>/
```

其中`train_metrics.jsonl`记录每步全局loss、有效tokens、LR、梯度范数和显存；
`ddp_diagnostics`记录各rank心跳/失败；router collapse仅作非致命诊断。
最终checkpoint为`checkpoint-009244`，正常完整运行最后保留8500、9000、9244。
job日志独立位于该路线的`job_logs`，不写入Git checkout。

## 中断恢复及可选短验证

完整恢复仅接受本路线、相同batch/精度/训练超参、未完成的一轮checkpoint。
恢复模型、optimizer、schedule步数、每rank Python/CPU/CUDA RNG和数据游标。
游标保存当前分片入口的专用shuffle RNG及已消费batch数，重放当前分片即可准确恢复
shuffle buffer和后续样本；不保存整段文本buffer，不需要重读此前分片。
重放大分片可能耗时；跨设备/不同CUDA内核不承诺浮点bitwise一致。

```bash
bash code/RSmol/run_stage4_mizar_text_7_14x2_7_mesh_training_5090.sh \
  --resume-from /absolute/formal_run/checkpoint-005000
```

恢复会建立新的输出目录，不覆盖旧run。micro4来源恢复时也必须显式指定micro4/GA32。
可选短验证使用相同初始化、数据和9244步scheduler：

```bash
# 10 updates；输出中checkpoint-000010
bash code/RSmol/run_stage4_mizar_text_7_14x2_7_mesh_training_5090.sh --gate D
# 从短验证第10步完整恢复，再做2步；不能用于放行或恢复正式训练
bash code/RSmol/run_stage4_mizar_text_7_14x2_7_mesh_training_5090.sh \
  --gate E --resume-from /absolute/short_run/checkpoint-000010
```

## 验证与文件地图

```bash
# 纯本地/CPU检查；tensor依赖缺失会明确skip
python -B -m unittest discover -s tests -p 'test_mizar_text_7_14x2_7_mesh*.py' -v
# 远程rsmol环境可单独执行真实小维度CPU张量测试，不需要下载权重
CUDA_VISIBLE_DEVICES="" python -B -m unittest discover -s tests -p 'test_mizar_text_7_14x2_7_mesh_tensors.py' -v
```

真实tensor测试覆盖官方router初始化/meta-safe、最终LM namespace/严格加载、
28层继承、42次轨迹、三组query/记忆初态、两轮梯度、全量保存重载、42逻辑KV cache、
全局token梯度归一化、optimizer分组、RNG及checkpoint保存/恢复/保留策略。
纯Python检查覆盖来源校验、全部配置别名、层映射、调度、真实shuffle循环的准确重放。
Bash测试用stub vc验证参数传递，不触碰真实队列。CPU或模拟通过不能记作远程GPU PASS。

- 模型：`../recursive_model_mizar_text_7_14x2_7_mesh.py`
- 来源/架构/转换/数据/配方：本目录的`contract.py`、`conversion.py`、`data.py`、`recipe.py`
- 转换入口：`../scripts/convert_mizar_text_7_14x2_7_mesh.{py,sh}`
- 训练入口：`../scripts/train_stage4_mizar_text_7_14x2_7_mesh_ddp.{py,sh}`
- 提交入口：`../run_stage4_mizar_text_7_14x2_7_mesh_training_5090.sh`
