# 已完成 MMAU 消融结果的 perception / semantic / reasoning CPU 汇总

直接读取现有29个条件的 `conditions/*/summary.json`，核对 `condition_scores.json`、`baseline.json`、运行身份和最终 `PASS` 报告。按用户指定的8/8/11个子类合并，主分数为组内正确题数之和除以题数之和，不平均子类 accuracy。不重跑推理，不读取权重、音频、原始逐题 journal 或 GPU 缓存。

## 终端执行

将新增 `mizar_ablation_regroup/` 和 `run_mizar_ablation_regroup_cpu.sh` 经 GitHub 同步到远程后：

```bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM/code/RSmol
bash run_mizar_ablation_regroup_cpu.sh
```

入口自动 source 原 conda 初始化脚本并 `conda activate mellow_c8204d8`，复用该环境中已经能画图的 Matplotlib。使用 Agg 后端，无需桌面，`CUDA_VISIBLE_DEVICES` 设为空；不导入 Torch/Transformers，不调用 vc 或 torchrun。可直接在CPU终端运行。

默认来源：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_ablation_bf16_8gpu3090_v1/
```

默认输出为来源旁边的独立目录：

```text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_ablation_bf16_8gpu3090_v1_psr_v1/
```

可显式指定路径，来源目录与输出目录不能重叠或互为父子：

```bash
bash run_mizar_ablation_regroup_cpu.sh \
  --source-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_ablation_bf16_8gpu3090_v1 \
  --output-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_ablation_psr_custom
```

若当前终端已经激活环境，也可直接：

```bash
python -B -m mizar_ablation_regroup.regroup \
  --source-dir /hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_ablation_bf16_8gpu3090_v1
```

## 分组与图

精确映射保存在 `mapping.json`，区分大小写，覆盖全部27个官方子类。按已核对的MMAU元数据，perception为331题、semantic为230题、reasoning为439题；运行时以原结果中的实际子类计数为准，并要求29个条件各子类的分母与baseline完全一致、合计1000题。

主图 `psr_accuracy_by_layer.png` 是两个面板：左侧三组绝对accuracy与各自同色baseline虚线；右侧相对各自baseline的变化，单位为百分点。横轴是从1编号的原始消融层2～29。baseline保存在表格首行，作为曲线参考线，不虚构成第0层或第1层消融。

图例使用英文能力组名及题数，避免服务器中文字体缺失；不平滑曲线，不修改原likelihood预测或正确性判定。分组是用户指定的实验分类，不预先宣称浅层/中层/深层已经分别承担这些职责。

输出文件：

```text
psr_accuracy_by_layer.png    3000×1100主图，两个面板
psr_accuracy_by_layer.svg    同图矢量版本
psr_scores.csv              baseline＋28层共29行；各组正确数、分母、accuracy、百分点差值
psr_scores.json             完整分组数值，含各自baseline
subcategory_mapping.csv     27个子类的归属、分母与baseline正确数
group_mapping.json          本次用户指定映射的副本
run_config.json             原结果身份、输入文件指纹、脚本与映射身份
regroup_report.json         完整性检查、分组题数、baseline及输出文件哈希
```

终端打印每个条件的 `PSR_SCORE` 以及最终 `PASS`。原结果保持只读；同一输入和脚本重跑可重新生成同身份输出。若源汇总、映射或脚本改变，请使用新的输出目录。校验只依赖已完成且内容哈希正确的汇总，不重新运行原GPU审计，也不要求旧训练/模型/数据文件仍存在。

## 本地验证

```bash
# 仓库根，CPU环境需要Matplotlib
python -B tests/validate_mizar_ablation_regroup.py
```

测试用明确标记的合成29组汇总，检查加权计分、baseline差值、完整覆盖、未知/漏分子类、来源身份/内容损坏拒绝、源文件字节不变及真实PNG/SVG生成。测试分数不是用户远程实测结果。
