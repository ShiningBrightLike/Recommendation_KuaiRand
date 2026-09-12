# 精排模块路线图（Ranking-stage Roadmap）

> 范围：本项目定位为**个人离线研究仓库**，数据为离线下载的 KuaiRand 数据集，不追公司级全链路，不实现召回/粗排/重排与线上服务。目标是把**精排（Fine Ranking）**这一环做成一个自洽、完整、可复现的离线研究模块。

相关文档：[README](README.md) · [project_overview](docs/project_overview.md) · [CONTEXT](CONTEXT.md) · [ADR-0001 特征决策集纪律](docs/adr/0001-feature-decision-set-discipline.md) · [ADR-0002 置换重要度与两阶段门控](docs/adr/0002-permutation-importance-and-two-stage-gate.md)

---

## 1. 完整性定义（Definition of Done）

精排这一环算“完整”，标准不是模型数量，而是：

> **在同一条离线数据上，任意两个特征集合或模型结构都能被公平训练、按排序指标比较、给出置信度，并记录成可复现的结论。**

由此拆出五条必须闭环的链路：样本与评估单元、模型能力、训练协议、评估体系、特征迭代。任何一条缺失，结论都不可信或不可比。

## 2. 现状盘点

| 维度 | 现状 | 缺口 |
| --- | --- | --- |
| 精排模型 | MMoE 四任务基线（`MMoE_model.py`） | 缺对照模型（Shared-Bottom / PLE-CGC / DCN-v2 / FM）、缺序列建模（DIN） |
| 训练协议 | train/val/test 时间切分、固定 seed、早停只看 val | 缺多种子重复与方差报告、缺超参搜索、缺损失/采样策略实验 |
| 特征迭代 | 置换重要度 + 影子特征噪声对照（`feature_importance.py`） | ADR-0002 的“确认阶段（同种子重训对比）”未实现；特征无版本指纹 |
| 评估指标 | 每任务 AUC | 缺排序指标（GAUC / NDCG@K / Recall@K / MAP）、缺概率校准（ECE）、缺分片评估 |
| 评估单元 | 仅特征 + 标签 parquet | **缺 group/id 列**（`user_id` / `video_id` / `date`），无法按用户组成候选列表 |
| 数据正确性 | 离线静态 CSV | 视频统计特征是**全期聚合**，相对训练期可能含未来信息（潜在泄漏） |
| 结果呈现 | README 已展示基线曲线与特征重要度 | 缺模型对比表、消融表、实验台账、批量打分脚本 |
| 工程质量 | 特征模块单测、锁定依赖、ADR/术语表 | 缺 CI（smoke + 数据契约 + 单测）、缺 lint/格式检查、缺 schema 校验 |

## 3. 待决策项（含推荐默认值）

这些是路线图里唯一需要人来拍板的分支；每项都给了推荐值，未拍板前按推荐值推进。

| 编号 | 决策 | 选项 | 推荐 |
| --- | --- | --- | --- |
| D1 | 候选列表的分组键 | 用户级（测试期全部曝光） vs 用户-天级 | **用户级**：数据量足够、实现简单；用户-天作为后续细化 |
| D2 | 主排序指标 | GAUC / NDCG@K / Recall@K | **GAUC（点击、点赞）为主**，NDCG@10 与 Recall@10 作辅助；关注/评论仅参考 |
| D3 | 候选集范围 | 仅曝光样本 vs 曝光 + 采样负例 | **v1 仅曝光样本**（衡量“曝光列表内的重排序”），采样负例留作后续 |
| D4 | 概率校准目标 | 不校准 / Platt / Isotonic | **Platt（按任务）**：样本量友好；校准前后同时报告 ECE |

## 4. 路线图

### P0 — 让精排“可评估”

前置：无。产出后，后续所有模型实验才有可信的比较口径。

| ID | 任务 | 交付物 | 验收标准 | 预估 |
| --- | --- | --- | --- | --- |
| RANK-P0-1 | 评估用 id 侧表 | `data_process.py` 额外输出 `processed_ids[_val/_test].parquet`（`user_id, video_id, date`），不进模型输入 | 行数与 X/y 一致；可 join 回特征 | 0.5 天 |
| RANK-P0-2 | 排序指标套件 | `evaluation.py`：GAUC、NDCG@K、Recall@K、MAP，按 D1/D2 口径 | 单测用合成候选列表验证指标数值；基线报告含新指标 | 2 天 |
| RANK-P0-3 | 概率校准与 ECE | 校准层（Platt）+ 可靠性曲线，输出到 run 目录 | 校准前后 ECE 均记录；曲线入库 `docs/assets/` | 1 天 |
| RANK-P0-4 | 统计特征泄漏审计 | “全期聚合 vs 仅训练期重算”两组对照实验 + ADR | 两组指标差异与结论写入 ADR，明确后续默认用哪版 | 1.5 天 |
| RANK-P0-5 | 分片与置信区间 | 按用户活跃度/视频热度/冷启动分片评估 + bootstrap CI | 报告含分片表与 95% CI | 1 天 |

**P0 出口标准**：`main.py` 训练结束后自动产出“AUC + GAUC + NDCG@10 + ECE”评估报告，并带分片与置信区间；泄漏审计结论已写入 ADR。

> 进度说明（2026-09-13）：本轮只完成了 FIX-1..FIX-5（见 §9），其中泄漏审计为“去特征”上界对照；GAUC/NDCG@10/ECE、评估 id 侧表、分片置信区间仍为 open，P0 出口标准尚未达成。

### P1 — 让精排“可比较”

前置：P0 完成（否则模型对比没有可靠口径）。

| ID | 任务 | 交付物 | 验收标准 | 预估 |
| --- | --- | --- | --- | --- |
| RANK-P1-1 | 模型对照矩阵 | Shared-Bottom / MMoE / PLE-CGC 三个结构，同一配置可切换 | 三者在同数据、同 seed 下跑通并出对比表 | 3 天 |
| RANK-P1-2 | 特征交叉模块 | DCN-v2 或 FM 层，可插拔 | 开关式配置；对比表含“有/无交叉” | 2 天 |
| RANK-P1-3 | 多种子与方差报告 | `--seed` 支持多值/重复运行；汇总 mean±std | 每个配置至少 3 个 seed，报告含方差 | 1 天 |
| RANK-P1-4 | 损失与采样实验 | focal loss、任务权重、负采样比例实验 | 对关注/评论专项报告；结论写入实验台账 | 2 天 |
| RANK-P1-5 | 序列特征 + DIN | 由日志构造用户历史序列，DIN 注意力塔 | 与 MMoE 基线同口径对比；序列特征纳入重要度报告 | 4 天 |

**P1 出口标准**：`docs/experiments.md` 中的“模型对比表”和“消融表”覆盖全部 P1 实验，每行都有 run 目录、配置、指标与结论。

### P2 — 让精排“可交付”

前置：P0 + P1 完成。

| ID | 任务 | 交付物 | 验收标准 | 预估 |
| --- | --- | --- | --- | --- |
| RANK-P2-1 | 特征门控 v2（ADR-0002 确认阶段） | 加载层 `--keep-features/--drop-features` + 同种子重训对比脚本 | 能一键产出“有/无候选”对比并给出采纳建议 | 2 天 |
| RANK-P2-2 | 批量打分脚本 | `score.py`：加载模型 + parquet → 输出每样本打分/每用户排序 | 支持 `--split` 与 `--top-k`，输出 csv/parquet | 1 天 |
| RANK-P2-3 | 实验台账 | `docs/experiments.md`：run、配置、指标、结论四列表 | 每个已完成实验一行，链接到 run 目录/metrics.json | 1 天 |
| RANK-P2-4 | CI 与数据契约 | GitHub Actions：单测 + `--smoke` 训练 + 数据 schema 校验 | PR 上自动运行；失败阻断合并 | 1.5 天 |
| RANK-P2-5 | 结果发布流程 | 从 run 目录筛选结果 → `docs/assets/` → README 展示 | 流程写入 README；至少发布一次模型对比表 | 0.5 天 |

**P2 出口标准**：新特征/新模型从实验到结论再到展示，全程有脚本可跑、有台账可查、有 CI 兜底。

## 5. 实验协议（沿用并强化现有纪律）

1. **评估集纪律**（ADR-0001）：特征筛选与模型对比只看验证集；测试集仅在组合定稿后评估一次。
2. **多种子**：除现有 `seed=2025` 外，补充 2026/2027；所有对比表给 mean±std，单次结果不入结论。
3. **配置记录**：每个 run 的 `metrics.json` 必须含特征 schema 指纹、超参、种子、数据行数。
4. **特征门控**（ADR-0002）：批量过滤用置换重要度，采纳前必须做确认阶段重训对比。
5. **结论沉淀**：任何“某特征/某结构更好”的结论，必须能在 `docs/experiments.md` 找到对应 run。

## 6. 风险与约束

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 候选集只有曝光样本 | NDCG/Recall 衡量的是“曝光内重排序”，非真实全量召回后排序 | 文档明确口径；后续可加采样负例做敏感性分析 |
| 关注/评论正样本极稀疏（0.1%–0.25%） | 排序指标波动大，易被噪声主导 | 默认不参与门控；报告中单独标注，必要时只做定性观察 |
| 视频统计特征全期聚合 | 潜在未来信息泄漏，虚高指标 | RANK-P0-4：首轮“去特征”对照已完成；默认版本待训练期重算对照后确定（见 ADR-0004） |
| 单机 CPU 训练 | 序列模型与多 seed 实验耗时上升 | 优先小规模实验 + `--max-rows`；必要时再考虑 GPU |
| 无上游候选生成 | 无法评估端到端推荐效果 | 明确非目标；精排结论只在“给定候选集”内成立 |

## 7. 非目标（明确不做）

- 召回 / 粗排 / 重排模型与多阶段链路编排；
- 在线服务、实时特征、A/B 实验平台、监控告警；
- 自动获取新数据、流式训练与线上闭环。

## 8. 立即可执行的 Issue 清单

> 状态说明：本清单仅作文档记录，尚未创建到 GitHub。需要时可按现有 issue tracker 配置批量创建。

| Issue 标题 | 对应任务 | 建议标签 | 优先级 |
| --- | --- | --- | --- |
| 输出评估用 user/video id 侧表 | RANK-P0-1 | `ready-for-agent` | P0 |
| 精排排序指标套件（GAUC/NDCG@K/Recall@K/MAP） | RANK-P0-2 | `ready-for-agent` | P0 |
| 概率校准与 ECE 报告 | RANK-P0-3 | `ready-for-agent` | P0 |
| 视频统计特征泄漏审计（全期 vs 训练期） | RANK-P0-4 | `ready-for-human` | P0 |
| 分片评估与 bootstrap 置信区间 | RANK-P0-5 | `ready-for-agent` | P0 |
| 模型对照矩阵（Shared-Bottom/MMoE/PLE-CGC） | RANK-P1-1 | `ready-for-agent` | P1 |
| DCN-v2/FM 特征交叉模块 | RANK-P1-2 | `ready-for-agent` | P1 |
| 多种子方差报告 | RANK-P1-3 | `ready-for-agent` | P1 |
| 损失与采样策略实验 | RANK-P1-4 | `ready-for-human` | P1 |
| 序列特征 + DIN | RANK-P1-5 | `ready-for-human` | P1 |
| 特征门控 v2（确认阶段重训对比） | RANK-P2-1 | `ready-for-agent` | P2 |
| 批量打分脚本 score.py | RANK-P2-2 | `ready-for-agent` | P2 |
| 实验台账 docs/experiments.md | RANK-P2-3 | `ready-for-agent` | P2 |
| CI：单测 + smoke + 数据契约 | RANK-P2-4 | `ready-for-agent` | P2 |
| 结果发布流程文档化 | RANK-P2-5 | `ready-for-agent` | P2 |

## 9. 进展记录

### 2026-09-13

- FIX-1：修复 `--candidate-cols` 字符集合解析 bug，补单测。
- FIX-2：统一“总体重要度”口径为门控任务（代码/README/CONTEXT 一致）。
- FIX-3：早停默认改为 `val_auc_mean`（ADR-0003），`--monitor` 可切换。
- FIX-4：新增 `--drop-features` / `--drop-stat-features` 与 `leakage_audit.py`；已完成首轮统计特征泄漏审计，结论见 `docs/leakage_audit.md`（严格 point-in-time 重算仍待做，RANK-P0-4 保持 open）。
- FIX-5：新增 Logistic / Shared-Bottom / 单任务对照模型与 `baselines.py`，首轮结果见 `docs/assets/baselines_v1.md`；PLE-CGC 等结构升级仍属 RANK-P1-1 后续。
- 复审修正：`val_auc_mean` 早停改为**门控任务（点击/点赞）平均**，与 ADR-0003/CONTEXT 口径一致；新增 ADR-0004 记录泄漏审计的默认取舍。
- 复审后按新口径重跑：单任务 0.7186 > MMoE 0.7143 > Shared-Bottom 0.7050 > Logistic 0.6898（四任务均值）；泄漏审计差值调整为 −0.0214（四任务均值）/ −0.0072（门控均值）。
