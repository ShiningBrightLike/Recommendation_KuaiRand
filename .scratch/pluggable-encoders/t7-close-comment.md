T7 已完成并合入 `main`。

- 提交：`031ef61` feat: rerun permutation importance on the new default model
- 基础 run：`fi-v2-base_20260915_090053`（新默认模型 `mmoe+mlp`，seed 2025，全量数据，best epoch 11；测试集 点击 0.7212 / 点赞 0.8054 / 关注 0.7054 / 评论 0.6360）
- 重跑口径：**特征决策集**（验证集 4/16–4/21，190,802 行），94 特征 × 3 次置换，耗时 213 秒，**未触碰最终确认集**；`cutoff=0.001`、门控任务 点击/点赞，与 ADR-0002 一致。
- 判定变化（旧默认模型 → 新默认模型）：
  - 分布：通过 50 / 待确认 4 / 不通过 40（旧：通过 50 / 不通过 44）
  - 13 个特征改判：`onehot_feat6`、`onehot_feat11`、`onehot_feat12` 通过→待确认；`is_live_streamer`、`register_days_range`、`share_user_num` 通过→不通过；`complete_play_user_num`、`direct_comment_cnt`、`download_cnt`、`follow_cnt`、`follow_user_num1`、`reduce_similar_cnt` 不通过→通过；`comment_like_user_num` 不通过→待确认。变化集中在 0.0005–0.002 的弱信号边界区。
  - 头部稳定：`tab` 0.0632 仍居首，第 10 位由 `onehot_feat1` 换成 `like_cnt`。
- 噪声对照：影子特征 `shadow_0` 总体重要度 0.000082 ± 0.000050，远低于 cutoff，噪声下限识别正常，两版阈值口径可直接沿用。
- 结论：基于旧默认模型的门控判定**全部作废**，以 v2 为准；本报告仅为 ADR-0002 的批量阶段，采纳前仍需确认阶段（同种子重训对比），该阶段仍属未实现项（RANK-P2-1）。
- 产物：`docs/assets/feature_importance_v2_report.md` / `feature_importance_v2.{csv,json}` / `feature_importance_v2_top.png`；README「特征优选」新增 v2 小节并存档 v1；`docs/project_overview.md` 新增 7.6 节；ROADMAP §2 与 §9（2026-09-15）同步。
- 全量测试 44 条通过。
