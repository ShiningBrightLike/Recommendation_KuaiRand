# 实验台账

> 本文件是精排实验的**唯一台账**（roadmap RANK-P2-3）：每次实验批次一行，记录 run 目录、配置、指标与结论。
> 规则见 ROADMAP §5：结论只依据**特征决策集**（验证集）选型；最终确认集（测试集）每配置只评估一次；对比结论要求同协议且 ≥3 个 seed（`baselines.py --seeds`）。

## 1. 已完成实验

### 1.1 当前有效结果

| # | 时间 | run 目录 | 配置 | 指标（测试集，mean±std over seeds） | 结论 |
| --- | --- | --- | --- | --- | --- |
| E6 | 2026-09-14/15 | `KuaiRand-Pure/saved/runs/baselines-v2-3seed` | 6 配置 × 3 seed（2025/2026/2027）：`logistic`、`shared_bottom+mlp`、`single_task+mlp`、`mmoe+mlp`（默认）、`mmoe+dcn`、`mmoe+senet`；全量 train/val/test = 950,310 / 190,802 / 295,497；epochs 30、`val_auc_mean` 早停 | 四任务均值：0.6936±0.0033 / 0.7089±0.0074 / 0.7142±0.0095 / 0.7132±0.0064 / 0.7076±0.0142 / 0.7170±0.0077；门控均值：0.7432 / 0.7670 / 0.7666 / 0.7649 / 0.7664 / 0.7695 | `senet` 两口径最好但未超出种子噪声（+0.0038 / +0.0046 vs 默认）；`dcn` 编码器参数最多（108,900 ≈ `mlp` 的 5 倍）却最差且方差最大；各结构差异在噪声内，v1 的「单任务领先」被弱化。归档：`docs/assets/baselines_v2.{md,json,png}` |
| E7 | 2026-09-15 | `KuaiRand-Pure/saved/runs/fi-v2-base_20260915_090053` | 新默认 `mmoe+mlp`（seed 2025，全量，best epoch 11）+ 置换重要度（决策集 = 验证集 190,802 行，94 特征 × 3 次，213 秒，`cutoff=0.001`） | 测试集 点击 0.7212 / 点赞 0.8054 / 关注 0.7054 / 评论 0.6360；判定分布 通过 50 / 待确认 4 / 不通过 40；`shadow_0` = 0.000082±0.000050 | 换模型导致 **13 个特征改判**（集中在 0.0005–0.002 弱信号区），旧默认模型的判定全部作废；影子特征仍远低于阈值，阈值口径可沿用；本报告仅为 ADR-0002 **批量阶段**。归档：`docs/assets/feature_importance_v2*` |
| E5 | 2026-09-13 | `audit_all_20260913_015323`、`audit_nostats_20260913_015506`、`audit2_all_20260913_021208`、`audit2_nostats_20260913_021525` | 统计特征泄漏审计：全量特征 vs 去掉 51 列全期视频统计特征，其余同协议 | 四任务平均测试 AUC 0.7143 → 0.6929（−0.0214）；门控均值 −0.0072 | 这批特征贡献可观但含 point-in-time 风险；ADR-0004 决定默认仍保留全量特征，严格重算（RANK-P0-4）未完成。**绝对数值基于旧默认模型，需在新默认下复核** |

### 1.2 已作废（旧默认模型 / 单 seed）

> 作废原因统一为：ADR-0005 落地后默认模型变为「特征编码器 `mlp` → 多任务结构」，且这些结论多为单 seed。它们的产物保留在 `docs/assets/` 作历史档案，**不得作为当前采纳依据**。

| # | 时间 | run 目录 | 配置 | 指标 | 作废说明 |
| --- | --- | --- | --- | --- | --- |
| E1 | 2026-09-07 | `baseline-v1_20260907_001737`（原目录已不在本地，产物归档于 `docs/assets/`） | MMoE（旧默认，无编码器）、seed 2025、全量数据、30 epochs、best epoch 9 | 总 loss 0.6914；点击 0.7223 / 点赞 0.8090 / 关注 0.7110 / 评论 0.6495 | 旧默认模型、单 seed；归档 `docs/assets/baseline_v1_metrics.json`、`baseline_v1_curves.png` |
| E2 | 2026-09-10 | `fi-shadow_20260910_010531` | 同 E1 + `shadow_0`；置换重要度 94 特征 × 3 次（验证集，约 4 分钟） | 判定分布 通过 50 / 不通过 44；`shadow_0` = 0.000095±0.000062 | 旧默认模型；判定已被 E7 取代。归档 `docs/assets/feature_importance.*` |
| E3 | 2026-09-13 | `baselines_20260913_020123`、`baselines_20260913_021844` | logistic / shared_bottom / single_task / mmoe，seed 2025 单 seed | 四任务均值 0.6898 / 0.7050 / 0.7186 / 0.7143；门控均值 0.7428 / 0.7664 / 0.7685 / 0.7664 | 旧默认模型 + 单 seed；结论已被 E6 取代。归档 `docs/assets/baselines_v1.*` |
| E4 | 2026-09-14 | `t1-*`、`t3-*`、`t4-*`、`t5-*`、`t6-*` 等冒烟 run | 迁移与新增编码器的冒烟验证（`--smoke` 或 `--max-rows` + 1 epoch） | 只验证"能跑通、能落盘、能重载"，指标无意义 | 工程证据，不作为任何性能结论 |

## 2. 产物约定

- 每个 run 目录（`KuaiRand-Pure/saved/runs/<run>/`）含 `metrics.json`（配置、两轴 `model` 块、逐 epoch history、测试集指标）、`model.keras`、`curves.png`、`training.log`；`saved/` 不入库。
- `docs/assets/` 只放**发布用**的汇总产物：对照表（`baselines_v*`）、特征重要度（`feature_importance*`）、泄漏审计与基线归档。
- 新增一行台账的最小步骤：跑完实验 → 在 `docs/assets/` 放汇总产物 → 在本文件补一行（run 目录、配置、指标、结论）→ 若结论推翻了旧行，把旧行移入 §1.2 并写明作废原因。

## 3. 相关文档

- [ROADMAP.md](../ROADMAP.md)（里程碑、验收标准、§9 进展记录）
- [project_overview.md](project_overview.md)（架构与结果详述）
- [CONTEXT.md](../CONTEXT.md)（术语表：置换重要度、特征决策集、最终确认集、特征编码器、多任务结构……）
- [ADR-0001 特征决策集纪律](adr/0001-feature-decision-set-discipline.md) · [ADR-0002 置换重要度与两阶段门控](adr/0002-permutation-importance-and-two-stage-gate.md) · [ADR-0005 两轴分层](adr/0005-pluggable-encoder-and-multi-task-structure.md)
