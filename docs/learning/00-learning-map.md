# 推荐精排学习地图

> 面向有 Python 和机器学习基础、但还不熟悉推荐系统精排的读者。本文档层在工程实现之上提供一条循序渐进的学习路径；代码、实验和术语的最终依据仍以仓库源码、CONTEXT.md、ADR 与发布资产为准。

## 你将学会什么

完成主线后，你应该能够：

- 说清召回、粗排、精排和重排在推荐链路中的职责；
- 把一条曝光日志解释成训练样本、反馈标签和用户候选集合；
- 理解类别 embedding、数值标准化、特征编码器和多任务结构；
- 区分 AUC 与用户级曝光内排序指标，并解释它们的边界；
- 跑通本项目的数据处理、模型 smoke、验证集比较和排序评估；
- 读懂 [data_process.py](../../data_process.py)、[models/](../../models/)、[main.py](../../main.py) 和 [evaluation.py](../../evaluation.py) 的关键入口。

本教程不承诺实现召回、全量候选生成、线上服务、序列模型或因果效果评估。它讨论完整推荐链路，但动手部分聚焦本项目已经实现的离线精排。

## 双轨阅读路线

### 90 分钟快速导览

按以下顺序阅读，目标是建立全局地图并跑通最小链路：

~~~text
README
  → 01 推荐链路
  → 02 样本与时间切分
  → 04 多任务模型
  → 05 排序评估
  → 06 代码 walkthrough 与 smoke
~~~

### 深入学习与实验线

完整阅读 01–06，运行 examples/ 中的 toy 示例，再对照：

- [docs/experiments.md](../experiments.md)：实验台账与 E8/E10/E11 教学案例；
- [docs/project_overview.md](../project_overview.md)：工程架构与结果详述；
- [CONTEXT.md](../../CONTEXT.md)：项目术语和评估边界；
- [docs/adr/](../adr/)：为什么采用当前数据、模型和评估协议。

## 章节地图

| 章节 | 核心问题 | 主要代码入口 |
| --- | --- | --- |
| [01 推荐系统全链路](01-recsys-pipeline.md) | 本项目在推荐系统的哪一段？ | [README.md](../../README.md), [main.py](../../main.py), [evaluation.py](../../evaluation.py) |
| [02 样本、标签与时间切分](02-samples-and-splits.md) | 什么是一个样本，为什么不能随机切分？ | [data_process.py](../../data_process.py) |
| [03 特征与输入张量](03-features-and-inputs.md) | 字段如何变成模型可以处理的向量？ | [data_process.py](../../data_process.py), [models/inputs.py](../../models/inputs.py) |
| [04 多任务精排模型](04-multitask-models.md) | 多个反馈任务应该共享什么、分开什么？ | [models/encoders/](../../models/encoders/), [models/mtl/](../../models/mtl/) |
| [05 用户级排序评估](05-ranking-evaluation.md) | 指标到底回答了什么问题？ | [evaluation.py](../../evaluation.py), [aggregate_ranking.py](../../aggregate_ranking.py) |
| [06 项目 walkthrough](06-project-walkthrough.md) | 如何从命令走到源码和实验产物？ | [data_process.py](../../data_process.py) → [evaluation.py](../../evaluation.py) |

## 每章怎么读

每章都遵循同一模板：

~~~text
为什么 → 直觉 → 最小公式 → 图 → 本项目代码 → toy 示例/命令 → 常见误读 → 练习
~~~

图中的颜色约定：蓝色表示数据，紫色表示模型，绿色表示评估，橙色表示实验决策，红色表示风险或禁止操作。实线表示当前实现，虚线表示概念边界或未来扩展。

## 深入参考层

完成六章主线后，进入 [深入参考层](reference/README.md)：

- 07–08：置换重要度、shadow feature、point-in-time 风险；
- 09–10：bootstrap、实验治理和 ADR 设计决策；
- 11：已实现、待实现和明确非目标的方法地图；
- FAQ、限制与阅读清单：用于复盘、写实验结论和继续学习。

## 版本与证据

本学习层按 2026-09-28 的仓库状态编写。教程中的实验数字只引用 [docs/assets/](../assets/) 的发布产物；候选模型只使用验证集，最终确认集（test）不能回流到选型或调参。

