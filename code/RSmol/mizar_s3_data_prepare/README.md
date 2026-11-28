# S3 全候选池：CPU 数据审计、原音频映射、过滤后训练 manifest

本阶段以 `s3_candidates.json` 的 metadata 为唯一训练监督，审计完整候选池，保留所有可靠映射且可正常解码的样本。没有采样、Q4 选择、选项轮换、CoT 增补或模型训练。Windows 本地维护源码，经 GitHub 同步后在远程 Linux 执行；生产音频和生成的数据集留在远程。

## 1. 输入与身份

默认输入均位于 `/hpc_stor03/sjtu_home/jinwei.zhang`：

| 参数 | 默认路径（相对上述根目录） |
| --- | --- |
| `--source` | `data/Mizar/s3_candidates.json` |
| `--avqa-metadata` | `data/AVQA/train_r1aqa_line.json` |
| `--avqa-root` | `data/AVQA`，含四个 `VGG*` 子目录 |
| `--historical-manifest` | `data/Mizar/s1_stage1_hybrid_filtered.json` |
| `--historical-audit` | `outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv` |
| `--output-dir` | `data/Mizar/s3_full_filtered_v1` |

CLI 固定官方 S3 SHA256：`c4b1862f58148d2728dfad10ee8346974924badca6de6b82181ffe605225cc65`。289,952 条中 AVQA 33,875、AudioMCQ 256,077（AudioCaps 49,687、Clotho 1,992、CompA-R 48,045、LP-MusicCaps-MTT 7,871、MusicCaps 637、SpeechCraft 123,529、TACOS 24,316）。不同内容或来源配额会拒绝，避免把 S1、S2 chunk-pair、Q4 子集误当本次输入。

过滤后实际 N 必须由服务器审计得出，不以本地参考标注匹配数预设最终训练规模。

## 2. 匹配与过滤规则

### AVQA

1. `source_sample_key=avqa_train_XXXXXXXX` 按原标注整数 `id` 连接，禁止把 id 当文件行号。
2. 检查原四选项顺序及零基 `answer` 与 S3 的 gold index/content 一致；只规范化大小写和空白。重复原 ID、答案/选项冲突以及原 `View` 类记录排除。
3. 核对 `video_name` 与原 `audio_path` 的 YouTube ID、起始秒数一致。ID 大小写敏感，从末尾拆分 `_秒数`，保留 ID 内的 `_`/`-`。
4. 在四个 `VGG*` 目录中按这个完整录音标识唯一匹配 WAV。`_000011` 和 `_11` 可表示同一起始秒；同 ID 不同时间不合并，不做模糊 stem/相似题目匹配。多份候选即排除，不任意挑一个。越界路径、越界软链接拒绝。
5. 只有“原标注缺少该 ID”的题，才允许复用另一个已验证题的**完全相同 S3 filepath1** 所证明的录音映射，并保存 anchor ID/SHA。标签冲突不能走此回退。同一 S3 path 被不同原录音认领时，该组全部排除。

S3 题面已音频化改写，继续完整使用 S3 `input/question/choices/answer`；原 AVQA 问题不写入训练监督。确实找不到依据的样本记录原因后过滤，不补抽。

### AudioMCQ

只从旧实际训练 hybrid 的 `source=audiomcq / audio_contribution=strong / stageE_variant=audiomcq_mcq` 行建立身份索引，按来源、原 `id`、规范化后四选项顺序连接 S3。再核对原音频文件名（SpeechCraft 额外核对 split），使用旧 TSV 中该 `audio_path` 的唯一成功路径；不重新跑旧映射规则、不按同名文件全盘寻找、不使用 S3 中作者的绝对路径作回退。

**S3 会修正部分 S1 的 gold 标签。旧答案不参与录音连接，也不覆盖 S3 答案。** 逐条记录历史 gold 与 S3 是否不同，报告 `retained_audiomcq_historical_gold_differences`。历史同录音的 CoT 行不加入新训练。

无法确认历史成员、旧映射缺失/冲突/失效的 AudioMCQ 也会明确列入排除清单。此举不改变原审计或原训练数据；报告按来源给出精确损失，不能把“映射缺失”解读为“数据本身不存在”。

### 音频与 metadata

每份去重音频通过 SoundFile 分块解码完整文件，检查实际帧数、声道、采样率和 NaN/Inf，记录时长、峰值/RMS、静音、header 帧数差异及 native float32 波形 SHA。坏/空/不可解码音频排除；短音频及全零音频只统计，不自动删掉。权限故障整体失败，缺少解码依赖在过滤开始前整体失败。

本工具不修改、复制或重采样音频。后续训练必须保留 S3 的 `mono -> resample32k -> first20 -> right-pad`，不能套用旧 S1 的中心裁剪。native hash 不等同于规范化 first20 hash，不宣称完成近重复或未见评测集审计。

S3 字段整体原样保留，包括未知字段；只改变 JSON 序列化格式，保留原相对顺序、原选项、原答案和原路径。重复 S3 主 ID、非法 gold、题面/选项/答案矛盾、chunk-pair 合同和截断 JSON 会整体拒绝。

## 3. 远程执行

代码完整经 GitHub 同步后，在登录节点运行 CPU 命令，无 `vc submit` 或 CUDA：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM
git pull --ff-only
bash code/RSmol/prepare_mizar_s3_data.sh --check-args
bash code/RSmol/prepare_mizar_s3_data.sh
```

wrapper 使用已有 `mellow_c8204d8` 环境，依赖 NumPy、SoundFile/libsndfile，清空 `CUDA_VISIBLE_DEVICES`，不安装依赖或修改共享环境。`RSMOL_CONDA_BASE` / `MELLOW_CONDA_BASE` 可覆盖 Miniconda 根目录。

自定义输入位置或新输出目录：

```bash
bash code/RSmol/prepare_mizar_s3_data.sh \
  --source /hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s3_candidates.json \
  --avqa-root /hpc_stor03/sjtu_home/jinwei.zhang/data/AVQA \
  --avqa-metadata /hpc_stor03/sjtu_home/jinwei.zhang/data/AVQA/train_r1aqa_line.json \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s3_full_filtered_v1
```

先打印输入哈希阶段和 metadata/hybrid 扫描进度，解码每增加 200 份唯一音频打印进度；每 10,000 候选打印总体进度，`--progress-every` 可调。完整解码会读取所有被映射音频，耗时由共享盘和文件总量决定，流式读取不把音频库加载到 RAM。metadata 使用有界流式解析；索引按样本/引用驻留。输出临时决策包含原 metadata，预留约数 GiB 磁盘空间，音频不另占一份。

只有最终 `S3_DATASET_PASS` 和有效 `manifest.json` 才表示数据集发布成功。所有排除原因见报告，不会以原始 289,952 冒充保留数量。

## 4. 交付数据与后续训练读取

```text
data/Mizar/s3_full_filtered_v1/
  train.json            过滤后的完整 S3 metadata JSON 数组，后续训练数据
  audio_path_map.json   原 filepath1 -> 远程实际音频绝对路径（精确键）
  resolved_paths.tsv    同一映射，field=filepath1/status=exists
  selection.jsonl      保留样本的源行号/SHA/选择后序号/映射证据
  excluded.jsonl       排除样本的完整原 metadata、源行号/SHA、原因和细节
  audio_audit.jsonl    每个保留的唯一音频的解码结果/内容指纹/stat
  dataset_report.json  全部来源/保留/排除/标签差异/容量/音频统计
  manifest.json        输入/源码/decoder 身份、全部产物 SHA/大小和 seal
```

**后续训练必须同时使用 `train.json` 和 `audio_path_map.json`。** `train.json` 的作者原路径保持不变，不能直接 `open(row['filepath1'])`。可以使用本包接口：

```python
from mizar_s3_data_prepare.prepare import verify_dataset, resolve_audio
from mizar_s3_data_prepare.io_utils import iter_rows, loads
from pathlib import Path

root = Path('/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s3_full_filtered_v1')
identity = verify_dataset(root)
paths = loads((root / 'audio_path_map.json').read_text(encoding='utf-8'))
for metadata in iter_rows(root / 'train.json'):
    audio_file = resolve_audio(metadata, paths)
    # 用 metadata['input']/['answer'] 作监督，audio_file 只决定文件读取。
```

不在本阶段划分新的 train/probe，也不按行随机划分同录音的多个 QA。旧 S1/S2/S3 暴露情况不据此消除。初始化官方 Stage3 final 后属于继续微调。

## 5. 复用、失败与核验

相同命令再次执行会核对输入、源码、decoder、AVQA 文件库存及历史映射音频 stat，再核验已有全部产物和保留音频 stat；同身份只读复用，不再解码。音频 stat 不替代重新解码，需要强制重查内容时使用新输出目录。

```bash
bash code/RSmol/prepare_mizar_s3_data.sh --verify-only
python -c 'import json; p="/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s3_full_filtered_v1/dataset_report.json"; d=json.load(open(p)); print(json.dumps({k:d[k] for k in ("input_rows","kept_rows","kept_by_source","excluded_rows","excluded_by_reason","retained_audiomcq_historical_gold_differences","unique_retained_audio")},ensure_ascii=False,indent=2))'
```

独占锁仅在新输出旁创建；在同级 `.building-*` 中写入/校验后原子 no-replace 发布，支持 CloudStor 不实现 `RENAME_NOREPLACE` 的兼容路径。失败保留决策文件及 `failure.json`，不发布训练目录。已有数据不覆盖。完整原行按“保留＋排除”恰好覆盖，保留 manifest 与 selection 逐行核对。

## 6. 本地验证与同步

本次 19 项 CPU 回归通过，无跳过。另对本地参考副本全量核验：S3 289,952 行通过结构检查；256,077 条 AudioMCQ 均可连接参考 S1 的身份、选项和原录音文件名，其中 43,562 条 gold 与 S1 不同且保持 S3 为准。AVQA 参考标注共 31,978 行，26,842 条 S3 AVQA 的 ID/选项/gold 一致，7,033 条缺原 ID；这个数字不包含服务器 WAV 核验或同 S3 录音 anchor 的后续恢复，不能作为最终过滤配额。

测试使用真实合成 WAV 和 JSON/TSV、实际 SoundFile 解码/过滤/发布/复用、错误答案/冲突/损坏拒绝、真实 Bash wrapper（conda/python 为接口夹具）。本地参考 S3/S1/AVQA 只验证 metadata；生产 hybrid/TSV/WAV、远程最终 N 尚未核验，CPU 结果不代表远程数据已准备好。

```bash
python -B tests/validate_mizar_s3_data_prepare.py
```

精确同步文件见 `sync_files.txt`。仅源码、测试、文档和小型审计报告进入 GitHub。示例本地 PowerShell 命令（用户自行执行 commit/push）：

```powershell
git add -- code/RSmol/mizar_s3_data_prepare code/RSmol/prepare_mizar_s3_data.sh tests/test_mizar_s3_data_prepare.py tests/validate_mizar_s3_data_prepare.py audit_reports/mizar_s3_data_prepare_validation.json RM.md
git commit -m "Prepare full S3 candidate audio mapping and filtered manifest"
git push
```
