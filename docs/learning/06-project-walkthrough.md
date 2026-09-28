# 06 项目 walkthrough 与复现

## 一条命令链

~~~text
原始 CSV
  ↓ python data_process.py
train/val/test parquet + pipeline_meta + sidecar
  ↓ python main.py --smoke
单模型训练/保存/快速验证
  ↓ python baselines.py --models ... --seeds ...
验证集候选比较
  ↓ python evaluation.py --model ... --split val
用户级曝光内排序
  ↓ 配置锁定后才运行 test final confirmation
最终确认与发布资产
~~~

## 第一步：预处理

~~~powershell
python data_process.py
~~~

检查 pipeline_meta.json 中的行数、字段和 split 元数据。它是后续输入 schema 的来源。

## 第二步：跑通模型链路

~~~powershell
python main.py --smoke
~~~

--smoke 的目标是验证数据加载、模型构建、训练、保存和加载，不用于解释性能。单模型 main.py 训练结束会读取 test 并写入结果，因此候选比较应使用 baselines.py 的默认 val 流程。

## 第三步：验证集候选比较

~~~powershell
python baselines.py --models "logistic,shared_bottom+mlp,single_task+mlp,mmoe+mlp,mmoe+dcn,mmoe+senet" --seeds 2025,2026,2027
~~~

候选选择只看验证集，报告多 seed 的均值与样本标准差。当前可信教学案例是 [E8 基线对照](../assets/baselines_v2.md)；单层 PLE-CGC 对照是 [E10](../assets/ple_cgc_val_3seed.md)。

## 第四步：排序评估

~~~powershell
python evaluation.py --model KuaiRand-Pure/saved/runs/<run>/model.keras --split val --k 10 --bootstrap 1000
~~~

输出位于 run 目录下的 ranking_eval/val/。它要求模型输出、标签和 processed_ids_val.parquet 按 row_id 严格对齐。[E11 用户级排序报告](../assets/ranking_metrics_val_ple_mmoe.md) 展示了如何阅读这类结果。

## 第五步：最终确认

只有配置锁定后，才运行：

~~~powershell
python evaluation.py --model <run>/model.keras --split test --final-confirmation
~~~

test 指标不是排行榜，不得回流到候选选择或调参。

## 从命令跳到源码

| 学习问题 | 入口 |
| --- | --- |
| 数据怎样合并和切分？ | [data_process.py](../../data_process.py) 的读取、日期和转换函数 |
| 输入张量怎样构建？ | [models/inputs.py](../../models/inputs.py) |
| encoder 和 MTL 怎样组合？ | [models/builders.py](../../models/builders.py) |
| 早停和 loss 怎样配置？ | [main.py](../../main.py) 的训练主流程 |
| GAUC/Top-K/ECE 怎样计算？ | [evaluation.py](../../evaluation.py) |
| 多 seed 怎样汇总？ | [aggregate_ranking.py](../../aggregate_ranking.py) |
| 置换重要度和泄漏审计怎样做？ | [feature_importance.py](../../feature_importance.py), [leakage_audit.py](../../leakage_audit.py) |

## 当前有效证据链

教程只引用验证集协议下的有效发布资产：

1. [E8：六配置 × 三 seed 基线](../assets/baselines_v2.md)；
2. [E10：单层 PLE-CGC 对照](../assets/ple_cgc_val_3seed.md)；
3. [E11：用户级曝光内排序](../assets/ranking_metrics_val_ple_mmoe.md)。

这些资产的详细运行目录、配置和结论以 [实验台账](../experiments.md) 为准。旧协议、单 seed 或消耗 test 的历史结果只用于审计，不用于当前模型结论。

## 练习顺序

1. 先运行 examples/ 中的三个 Python toy 示例，并阅读 sample_exposures.csv，不依赖 KuaiRand 原始数据；
2. 再运行 data_process.py 和 main.py --smoke；
3. 最后阅读 E8/E10/E11 的发布资产，写出“指标、split、seed、结论边界”四项实验摘要。

