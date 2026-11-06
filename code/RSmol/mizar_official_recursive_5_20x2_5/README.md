# 官方 Mizar final → 5 + 20×2 + 5：审计与转换

本次交付是独立的模型定义、只读审计、转换及严格重载接口。代码在本地修改，通过 GitHub 同步到远程 Linux；权重、转换产物和数据留在远程。本入口不提交 GPU 作业，不启动训练。原训练路线和评测器无需变动。

## 固定模型合同

- 来源：`/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M/checkpoints/final/stage3_q4_seed20260905.ckpt`，官方 Stage3 Q4 seed20260905 / step200。使用完整已训练的 LM、mapper、CED；`assets/smollm2-135m/model.safetensors` 仅作为来源资产审计，不用它替代 final LM。
- 完整保留物理层 0–29。前五层 prelude，中间 5–24 层共享基础权重执行两次，最后 25–29 层 coda。50 次调用顺序为 `0..24, 5..24, 25..29`；不做旧路线的 30→20 抽层。
- 中间每层 q/k/v/o、gate/up/down 加两套独立 LoRA，每轮一套，rank8 / alpha16 / scale2 / dropout0，A Kaiming、B零；基础 W 继续训练。LoRA 覆盖完整序列。
- 在逻辑索引 5 和 25 注入音频：共享 input projection/gate 的残差，然后分别为独立 CED8 / CED4 分支残差。CED gate 读取 **post-input hidden**。直接注入仅覆盖音频位置 0–125；共享 input projection 每个 LM forward 只执行一次，输入来自原 final mapper 输出。
- CED 每批只提取一次，保留最终特征及完整第4/8块输出；第4/8块特征使用各自的新 MLP。CED 冻结并始终 eval；mapper、LM 及新增参数可训练。无 memory/router。
- 新参数：LoRA **3,256,320**，音频分支 **2,777,859**，合计 **6,034,179**。完整模型独立参数 **165,334,019**；原454份 state tensor保留，新增584份，共1038份。输入 embedding 与 lm_head 保持同一 Parameter。
- 参数及前向 FP32，attention_mask=None，禁用 KV cache 和 gradient checkpointing。tokenizer 保留 vocab49152、PAD17、EOS/SEP0；完整训练前向支持 `126 audio + 1 SEP + 256 prompt + 300 answer = 683`。

后续训练意图已经写入产物，但 **trainer 尚未实现**：S1 hybrid 1,477,492条、3epochs，8GPU × microbatch4 × GA4 = GBS128。每epoch 11,542次更新、舍弃116条，合计34,626步、warmup1,731步。循环基础权重（含两种 norm）和全部新增参数 maxLR=6e-4；继承的 mapper、prelude/coda、embedding、最终norm/lm_head maxLR=3e-4。`trainability_audit()` 枚举独立参数、LR角色和参考 AdamW decay 分组，tied embedding 只计一次。answer-only CE、原数据/staging/GradNorm/调度等在下一阶段接入 trainer。

## 远程运行

代码提交/推送并在远程拉取后，在登录节点或有足够 CPU 内存的交互环境运行。以下路径以现有远程 checkout 为准：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only

# 只读：18个官方文件逐个核对大小/SHA256，不导入Torch。
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5.sh audit

# 转换：所有张量、数值前向与保存重载关卡都是必需的。
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5.sh convert --seed 0 --threads 8

# 独立复验已发布产物；包含实际完整音频和683-token前向。
bash code/RSmol/scripts/mizar_official_recursive_5_20x2_5.sh verify --tensors --threads 8
```

wrapper 激活 `mellow_c8204d8`，设置完全离线与空 `CUDA_VISIBLE_DEVICES`；转换固定使用 CPU / Transformers **4.54.1** / eager attention。需要该环境现有的 Torch、torchaudio、Transformers、safetensors 和 CED依赖，不自动联网安装。完整 CPU 前向可能需要数分钟，建议至少8GB可用内存；不会调用调度器。`MIZAR_520X2_CONDA_BASE` 可指定 Conda 根目录。

默认输出：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-5-20x2-5-serial84-looplora-r8
```

所有命令支持 `--help`。`audit --tensors` 增加源 state、严格加载、新参数结构、tokenizer 与同30层单遍数值检查；普通 `verify` 只校验产物文件，不运行张量。用 `--source /path/Mizar159M`、`--output /path/new-directory` 指定位置。源文件必须与仓库已有官方 [release_manifest.json](../mizar_official_eval/release_manifest.json) 完全一致。可用 `--report /path/new-report.json` 保存额外报告，拒绝覆盖已有报告。

## 转换的强制验收

1. 校验完整官方 checkpoint 和构造/tokenizer资产共18个文件的大小及 SHA256，再执行固定的本地 CED 构造代码。不会改写源文件，不使用网络。
2. 构造未扩展模型，要求恰好454份state和159,299,840独立参数；`weights_only=True` 在CPU读取官方 plain state_dict，拒绝 training wrapper、DDP前缀及低精度浮点。
3. **strict=True 加载全部官方权重之后**，才注册新分支。逐一 `torch.equal` 校验每份继承张量；检查新增参数数量、独立LoRA存储、零B/零gate、tied embedding、CED冻结与LR分组。
4. 在 eval/no_grad 下，使用原标准 Llama30层和相同的继承权重、输入执行单遍对齐（rtol/atol=2e-5）。这是同30层计算的检查；**零gate/B不代表50次调用等于原30次调用**。
5. 合成20秒波形完整走 CED→mapper/MLPs→683-token LM，通过实际hooks核对物理层50次调用、每轮140次LoRA投影、CED特征一次和输入投影一次。不是数据集质量或训练收敛验收。
6. 保存后，从产物重新构造模型，严格加载1038份state；逐张量及完整683-token logits要求精确相同。重新核对源文件与实现身份后，原子发布目录。

产物是完整 **model-only initialization**，不是训练恢复点：

```text
model.ckpt                    # 全量FP32 plain state_dict，含原权重、新模块、tied别名与buffers
config.json                   # 50逻辑 / 30物理层的递归配置，保留原text配置其他字段
tokenizer/                    # 原始7个text配置/tokenizer文件，逐字节复制
ced-small/                    # 固定的原CED构造代码/配置/基础资产，含README，共7文件
source_audit.json              # 官方来源与18文件哈希
conversion_report.json         # 继承、计数、LR角色、数值前向及重载检查
manifest.json                 # 文件清单/哈希、架构、训练意图、源码身份
LICENSE / NOTICE
licenses/Mizar-ClearBSD.txt
```

不存 optimizer、scheduler、训练RNG或数据游标。构造资产一并打包，重载无需原 Mizar159M 目录。只能通过本路线的显式 loader 加载，不能直接用标准 `AutoModelForCausalLM.from_pretrained()`；现有MMAU/MMAR evaluator尚未接入此架构。

```python
# 从 code/RSmol 下执行，或将该目录加入 PYTHONPATH。
from mizar_official_recursive_5_20x2_5.tensor_runtime import reload_verified
model, audit = reload_verified("/path/Mizar159M-5-20x2-5-serial84-looplora-r8", threads=8)
# 返回可训练CPU FP32模型，CED仍冻结/eval；推理时调用 model.eval()。
# 同一进程中，此loader会设置CPU/offline环境与Torch线程数。
```

默认拒绝已有输出、源/目标嵌套及Git checkout内的模型产物，且没有overwrite选项。独占锁防止同名转换同时执行；Linux用 `renameat2(RENAME_NOREPLACE)` 原子发布。失败时保留 `*.staging-*/failure.json`，最终目录不发布；普通verify拒绝未完成重载或含额外文件的产物。进程被强制杀死可能留下 `.conversion.lock`，确认无对应进程后再人工处置。实现的Python源码及官方manifest哈希均进入产物身份；源码变化会拒绝旧产物，需审查变动并重新转换。

## 本地验证与边界

```bash
python -B tests/validate_mizar_official_recursive_conversion.py
# 在具备真实Torch/Transformers的环境，禁止任何tensor skip：
python -B tests/validate_mizar_official_recursive_conversion.py --require-tensors
```

文件测试使用合成小文件验证实际SHA256、缺失/篡改/路径穿越/覆盖拒绝、原子发布、产物身份、完整配置保留及官方mapper AST；不读取远程权重。十一项可选真实CPU测试覆盖配置构造/嵌套字典隔离，以及小hidden维度但完整30物理/50逻辑层的独立LoRA、串行注入边界和post-input门控、梯度解锁、异常scope复位、strict重载及数值对齐。

本地缺少Torch/Transformers时，这十一项必须明确记为 **SKIPPED**；不会以mock张量登记PASS。统一报告为 `audit_reports/mizar_official_recursive_conversion_validation.json`。远程实际来源审计、生产转换及数值PASS以远程命令产生的报告为准。

## 第5步配置校验失败的修复

旧版可能在第5步报 `recursive config must preserve every original source config field`。
构造模型时直接把将要保存的字典传给 `LlamaConfig.from_dict()`，上游[4.54系列配置实现](https://github.com/huggingface/transformers/blob/v4.54.0/src/transformers/configuration_utils.py#L712)会原地写入 `attn_implementation`。
因此运行时字段进入了待保存配置，与由原始tokenizer配置生成的严格预期不同。该错误发生在重载的文件校验阶段，最终目录尚未发布。

修复为：配置构造入口使用深拷贝，隔离源配置、待保存配置与运行时对象的嵌套字段；保存权重前提前校验完整配置；失败时列出extra/missing/changed字段。原来的完整字段一致性检查保留。
本地新增3项配置回归测试（其中一项实际走构造→JSON保存→产物文件审计；配置库用可变行为替身，不模拟张量），另有2项真实Transformers配置测试。本地27项检查通过，11项真实依赖检查因缺依赖跳过。

将修复经GitHub同步到远程后，重跑原 `convert --seed 0 --threads 8` 命令即可。
原失败的 `.staging-*` 目录可保留，不会作为正式产物复用，也不会阻止新转换；不需要删除原始Mizar159M。
此次代码改变了实现身份，因此应重新转换，而非手动修改旧staging的配置或manifest来绕过检查。远程完整重载与最终发布仍需本次重跑确认。
