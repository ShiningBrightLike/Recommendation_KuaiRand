# 07 置换重要度与两阶段门控

## 它回答什么问题

置换重要度回答：

> 在固定模型和固定评估集上，把一列输入打乱后，模型的 AUC 下降了多少？

对特征 j 和任务 t，可以写成：

~~~text
importance(j, t) = AUC_baseline(t) - AUC_after_shuffle(j, t)
~~~

它衡量的是预测依赖，不是因果效应、业务价值或“模型权重大小”。

## 当前项目的协议

1. 模型固定，不在每个特征上重新训练；
2. 决策集固定为验证集；
3. 每次只打乱一个输入列；
4. 重复置换并报告 mean ± std；
5. 默认门控任务是点击和点赞，关注与评论仍然报告；
6. shadow_0 作为真实输入中的噪声参考；
7. 批量阶段用 cutoff 过滤候选；
8. 确认阶段用相同 seed 重训有/无候选的模型。

默认 cutoff 是 0.001。它是当前项目的实验约定，不是跨数据集的普适阈值。

## 为什么不直接用 SHAP 或前向选择

当前选择置换重要度，是因为它与 AUC 评估口径直接对应、可以作用于不同模型结构，且不依赖某一种可解释模型。它的代价也必须显式保留：

- 强相关特征会分摊重要度；
- 一次置换可能偶然让 AUC 上升，出现负值；
- 重要度依赖当前模型、当前 split 和当前特征集合；
- 重要度变化不能单独证明新特征应该被采纳。

## 如何读当前报告

参考 [feature_importance_v2_report.md](../../assets/feature_importance_v2_report.md) 时，先看四项元数据：模型、split、决策集行数、门控任务。再看 shadow_0 的均值和标准差，最后看相关特征提示。

当前发布资产：

- [Markdown 报告](../../assets/feature_importance_v2_report.md)；
- [JSON 结果](../../assets/feature_importance_v2.json)；
- [CSV 结果](../../assets/feature_importance_v2.csv)；
- [Top 特征图](../../assets/feature_importance_v2_top.png)；
- [ADR-0002：置换重要度与两阶段门控](../../adr/0002-permutation-importance-and-two-stage-gate.md)。

## 代码导航

- [feature_importance.py](../../../feature_importance.py)：参数解析、逐列置换、判定、报告和绘图；
- [data_process.py](../../../data_process.py)：shadow feature 必须在预处理时真实加入；
- [CONTEXT.md](../../../CONTEXT.md)：总体重要度、决策集和影子特征的术语边界。

## 不要这样下结论

- “通过 cutoff”不等于“具有因果作用”；
- “重要度最高”不等于“业务最重要”；
- “shadow 很低”不等于所有统计泄漏都不存在；
- 旧模型上的重要度不能直接迁移到新编码器或新多任务结构。

