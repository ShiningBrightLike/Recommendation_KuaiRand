# 10 ADR 导读：为什么项目这样设计

ADR 不是教程的替代品，而是记录“当时有哪些选择、为什么选这个、代价是什么”。建议先读主线，再按问题进入对应 ADR。

| 你遇到的问题 | ADR | 决策摘要 |
| --- | --- | --- |
| 为什么不能用 test 做特征筛选？ | [0001](../../adr/0001-feature-decision-set-discipline.md) | 验证集是特征决策集，test 只做最终确认 |
| 为什么采用置换重要度和两阶段门控？ | [0002](../../adr/0002-permutation-importance-and-two-stage-gate.md) | 与 AUC 口径一致，先批量过滤，再同种子重训确认 |
| 为什么早停看 val_auc_mean？ | [0003](../../adr/0003-model-selection-metric.md) | 避免加权 loss 被点击任务完全主导 |
| 为什么保留统计特征却承认风险？ | [0004](../../adr/0004-leakage-audit-default.md) | 维持基线可比性，同时等待严格窗口重算 |
| 为什么把模型拆成两条轴？ | [0005](../../adr/0005-pluggable-encoder-and-multi-task-structure.md) | 特征表示和任务共享是不同建模问题 |
| 为什么 PLE-CGC 的 shared 输出不接最后 tower？ | [0006](../../adr/0006-ple-cgc-multi-task-structure.md) | 保持任务分化语义，单层时 shared gate 无下游梯度是已知取舍 |
| 为什么排序指标叫“曝光内”？ | [0007](../../adr/0007-user-level-ranking-evaluation.md) | 候选集合是用户在 split 内的全部曝光行 |

## 读 ADR 的方法

1. 先找 Context：当时真正要解决的困难是什么；
2. 再找 Decision：选择了哪条协议；
3. 最后读 Consequences：哪些能力得到、哪些能力放弃；
4. 回到源码和发布资产，检查决策是否仍然成立。

如果代码、教程和 ADR 不一致，不应悄悄选择其中一个；应先确认当前协议，再更新术语、实现和参考文档。

