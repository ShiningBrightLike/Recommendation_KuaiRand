# 阅读清单

以下材料按“先理解本项目，再扩展方法”排序。外部论文中的方法不代表已经在本仓库实现。

## 先读本仓库

1. [KuaiRand 项目总览](../../project_overview.md)：数据、模型、实验结果；
2. [实验台账](../../experiments.md)：有效、作废和工程验证运行；
3. [CONTEXT.md](../../../CONTEXT.md)：术语边界；
4. [ADR 目录](../../adr/)：关键设计取舍。

## 数据与推荐链路

- Gao et al., *KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos*。本项目数据集论文，DOI：10.1145/3511808.3557624；
- 检索关键词：candidate generation、pre-ranking、ranking、re-ranking、offline evaluation。

## 多任务与排序模型

- MMoE：Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts；
- PLE：Progressive Layered Extraction for Multi-Task Learning；
- DCN / DCN-v2：特征交叉与低秩交叉网络；
- SENet：通过 squeeze-and-excitation 做字段级重标定；
- 检索关键词：multi-task recommendation、expert gating、task conflict、ranking loss。

## 评估与可信实验

- ROC AUC、NDCG、MAP、Recall 的定义与适用场景；
- calibration、reliability diagram、ECE；
- cluster bootstrap / grouped bootstrap；
- data leakage、temporal split、point-in-time correctness。

阅读外部方法时，始终追问三件事：候选集合是什么、信息截止时间是什么、指标回答什么问题。

