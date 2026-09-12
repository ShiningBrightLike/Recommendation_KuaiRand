# 精排架构审查（2026-09-13）

> 触发背景：在进入精排深度优化前，对当前代码快照做一次架构级静态审查。审查方式沿用 `code-review` 技能的两轴结构（Standards / Spec），并补充一层 ML 架构正确性核查。本文档同时作为后续修复工作的 spec 基线。

## 审查范围

代码快照：`bc2d771` + 工作区未提交改动（`ROADMAP.md` 等）。范围文件：`MMoE_model.py`、`main.py`、`data_process.py`、`data_loading.py`、`config.py`、`feature_importance.py`。

## 总结论

精排架构骨架没有致命设计错误：共享 Embedding + MMoE + 任务塔、时间切分、动态 vocab、置换重要度与影子特征对照构成了自洽的离线研究闭环。但存在 1 个已确认的实现 bug、3 个高危口径/正确性问题，以及若干中优先级改进项；建议先完成 P0 修复再进入模型结构升级。

## Standards（规范轴）

硬性违规：

1. **术语表与代码口径冲突**：`CONTEXT.md` 定义“总体重要度 = 四个任务的绝对 AUC 下降平均值”，代码只对 `gate_tasks`（默认点击/点赞）取平均，README 又写成“门控任务平均绝对 AUC 下降”。
2. **`_Avoid_` 词被使用**：README 使用笼统的“特征重要性”，而 CONTEXT 明确避免该说法。
3. **失效交叉引用**：`docs/project_overview.md` 引用 `docs/adr/0001-0002`，实际为两个独立文件。

判断性问题（smell baseline）：

4. 门控对每任务下降取 `abs()`，与 ADR-0002“负值不隐式修正”存在张力，需要显式说明“报告保留符号、门控取绝对值”。
5. `analyze()` 参数达 12 个（Data Clumps）。
6. `_add_shadow_features` 三个 split 共用 `seed=777`，val/test 噪声是 train 序列前缀，与 docstring 的 “independent” 不符。
7. `spec.kind == "categorical"` 在两处重复判断（Repeated Switches）。

## Spec（规格轴）

1. **缺失**：ADR-0002/CONTEXT 的“确认阶段”（同种子重训对比有/无候选）未实现，只有批量过滤阶段。
2. **规格外扩展**：相关性提示、`--shadow-features`、三态判定词、四类产物、README 真实结果表——相对 ADR 属扩展，但均为此前设计问答确认过，应回写 ADR-0002 而非删除。
3. **实现有错**：`--candidate-cols` 把逗号字符串直接 `set()`，得到字符集合，文档示例必然失败。
4. 若以 `ROADMAP.md` 为 spec，则 P0 全部未实现（本审查正是在此之前完成）。

## 架构正确性补充（ML 层面）

1. **统计特征 point-in-time 泄漏风险（最高优先）**：`video_features_statistic_pure.csv` 的 52 列为视频级全期聚合（`like_cnt`、`comment_cnt`、`follow_cnt` 等结果计数），直接作为特征。已核查：合并无行膨胀（1,141,112 → 1,141,112），单特征 AUC 0.50–0.68，属“视频侧先验”而非直接泄漏单个曝光标签；但若聚合窗口覆盖被预测曝光，仍违反 point-in-time 原则，需做“含/不含统计特征”对照审计。
2. **CVR 语义漂移**：项目名为“多目标 CVR 预测”，标签是曝光级 click/like/follow/comment 直接二分类，不是 click→conversion 条件概率。
3. **早停指标与精排目标脱节**：`monitor="val_loss"` 加权总损失中点击占 87.0%（best epoch 9），关注+评论合计 0.8%，早停实质只在选“点击模型”。
4. **稀疏任务噪声**：val 集关注/评论正样本仅 251 / 464 个，AUC 波动大，不宜作为默认门控任务（现已如此）。
5. **表征能力受限**：35 个类别特征共享一张 8 维 Embedding，`tag`/`onehot_feat*` 等高基数特征表达不足；无分域 Embedding、无特征交叉。
6. **MMoE 层序列化脆弱**：缺 `build()`/`get_config()`，Keras 3 需靠 `register_keras_serializable` + `custom_objects` 才能加载。
7. **缺对照模型**：无 Logistic / Shared-Bottom / 单任务基线，无法证明 MMoE 的增量价值。

## 修复计划（本次执行）

| 编号 | 问题 | 修复方案 | 状态 |
| --- | --- | --- | --- |
| FIX-1 | `--candidate-cols` 解析 bug | 抽出 `parse_candidate_cols()` 并补单测 | 已完成（单测覆盖） |
| FIX-2 | “总体重要度”口径冲突 | 以“门控任务”为准统一代码/README/CONTEXT，并注明报告仍含四任务明细 | 已完成 |
| FIX-3 | 早停只看加权总损失 | 新增 `val_auc_mean` 回调，默认按该指标早停，`--monitor` 可切换 | 已完成（ADR-0003） |
| FIX-4 | 统计特征泄漏风险 | 新增 `--drop-features` / `--drop-stat-features` 与 `leakage_audit.py` 对照审计 | 首轮完成，严格重算待做 |
| FIX-5 | 缺对照模型 | 新增 Logistic / Shared-Bottom / 单任务模型与 `baselines.py` 对比脚本 | 首轮完成（结果见 docs/assets/baselines_v1.md） |

## 修复验证结果

- 全量单测 19/19 通过（含新增的候选特征解析、特征过滤、模型构建器、早停辅助测试）。
- 泄漏审计（门控任务早停口径）：去掉 51 列全期统计特征后四任务平均测试 AUC 0.7143 → 0.6929（−0.0214），门控均值 −0.0072（`docs/leakage_audit.md`）。
- 对照模型（门控任务早停口径）：Logistic 0.6898、Shared-Bottom 0.7050、单任务 0.7186、MMoE 0.7143（四任务平均，`docs/assets/baselines_v1.md`）。

### 复审发现与修正（2026-09-13，复核本轮修复）

- **已修正**：`val_auc_mean` 原先取四任务平均，与 ADR-0003/CONTEXT 的“门控任务（点击/点赞）平均”不符；现已改为门控任务口径，并重跑泄漏审计与对照模型使所有结果与新口径一致。
- **已补齐**：新增 ADR-0004 记录泄漏审计的默认取舍；ROADMAP P0 出口标准补充进度说明，并修正“默认改用训练期重算版本”的措辞（改为待重算对照后决定）。

## 相关文档

- [ROADMAP.md](../../ROADMAP.md)
- [ADR-0001 特征决策集纪律](../adr/0001-feature-decision-set-discipline.md)
- [ADR-0002 置换重要度与两阶段门控](../adr/0002-permutation-importance-and-two-stage-gate.md)
- [CONTEXT.md](../../CONTEXT.md)
