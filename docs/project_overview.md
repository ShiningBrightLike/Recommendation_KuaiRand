# KuaiRand 多目标 CVR 预测（MMoE）项目梳理

> 本文档基于仓库当前代码与产物整理，描述项目目标、技术方案、数据链路、实现细节与已有结果。

---

## 1. 项目概述

本项目基于 **KuaiRand**（CIKM 2022）公开数据集，构建短视频场景下的**多目标 CVR/反馈预测模型**，同时预测用户对视频的**点击（click）、点赞（like）、关注（follow）、评论（comment）**四个行为。

当前实现状态：

- 数据预处理管线已完成：原始 CSV → 样本拼接 → 特征编码/归一化 → parquet + 编码器产物；
- 已实现 **MMoE（Multi-gate Mixture-of-Experts）** 多任务模型并完成两轮训练/评估；
- 训练产物（模型权重、训练日志、损失/AUC 曲线）保存在 `KuaiRand-Pure/saved/`。

---

## 2. 技术方案

### 2.1 任务定义

| 任务 | 标签列 | 说明 |
| --- | --- | --- |
| Task1 | `is_click` | 是否点击 |
| Task2 | `is_like` | 是否点赞 |
| Task3 | `is_follow` | 是否关注 |
| Task4 | `is_comment` | 是否评论 |

每个任务均为二分类，输出层使用 sigmoid。

### 2.2 模型结构（MMoE）

```
输入
├── 35 个类别特征 (int32)
└── 58 个数值特征 (float32)

类别特征 → 共享 Embedding (vocab=2500, dim=8) → Flatten → Concat
                                                          │
数值特征 ──────────────────────────────────────────────► Concat（93 维）
                                                          │
                                                   MMoE 层
                        ┌── 8 个专家（Dense(64, ReLU)）
                        └── 每个任务一个 softmax gate
                                                          │
                        每个任务：加权专家输出 → Dense(32, ReLU) → Dense(1, sigmoid)
                                                          │
输出：is_click / is_like / is_follow / is_comment
```

核心实现位于 `MMoE_model.py`：

- `MMoE` 自定义 Layer：`num_experts=8`，每个任务独立 gate（softmax 加权组合专家输出）；
- `build_mmoe_model()`：所有类别特征共享同一个 Embedding 层，数值特征直接拼接；
- 每任务 tower：`Dense(32, ReLU) → Dense(1, sigmoid)`。

### 2.3 训练配置（`main.py`）

| 配置项 | 取值 |
| --- | --- |
| 优化器 | Adam |
| 损失 | 每任务 binary_crossentropy |
| 损失权重 | `is_click: 1.0`，`is_like: 1.0`，`is_follow: 0.5`，`is_comment: 0.1` |
| 评估指标 | 每任务 AUC |
| Batch size | 1024 |
| Epochs | 30（早停生效时提前结束） |
| 早停 | `monitor=val_loss`，`patience=5`，`restore_best_weights=True` |
| 验证集 | 按时间切分的测试集（4/22–5/08） |

损失权重体现了对标签稀疏度的先验调整：点击/点赞样本充足给满权重，关注、评论更稀疏给低权重。

---

## 3. 数据链路

### 3.1 整体流程

```
KuaiRand-Pure/data/（原始 CSV）
  log_standard_4_08_to_4_21_pure.csv   ← 训练行为日志
  log_standard_4_22_to_5_08_pure.csv   ← 测试行为日志
  user_features_pure.csv               ← 用户特征
  video_features_basic_pure.csv        ← 视频基础特征
  video_features_statistic_pure.csv    ← 视频统计特征
                    │
                    ▼  data_process.py
  ① 按 user_id / video_id 左连接拼接样本
  ② date → 星期几；缺失值统一填 -1
  ③ 35 个类别特征：LabelEncoder + 全局偏移量 → feature_id（预留 UNK）
  ④ 58 个数值特征：StandardScaler（训练集 fit，测试集 transform）
  ⑤ 提取 4 个标签列
                    │
                    ▼  KuaiRand-Pure/data_processed/
  processed_X.parquet / processed_y.parquet          （训练集）
  processed_X_test.parquet / processed_y_test.parquet（测试集）
  label_encoders.pkl / scaler.pkl / feature_offsets.pkl
                    │
                    ▼  main.py
  MMoE 四任务训练 → 早停 → 保存模型与训练曲线
                    ▼
  KuaiRand-Pure/saved/（.h5 模型 / PNG 曲线 / 训练日志）
```

### 3.2 数据规模

| 文件 | 行数（含表头） | 用途 |
| --- | --- | --- |
| `log_standard_4_08_to_4_21_pure.csv` | ≈ 114.1 万 | 训练行为日志 |
| `log_standard_4_22_to_5_08_pure.csv` | ≈ 29.5 万 | 测试行为日志 |
| `log_random_4_22_to_5_08_pure.csv` | ≈ 118.6 万 | 随机曝光日志（**尚未接入** pipeline，可用于去偏/对照实验） |
| `user_features_pure.csv` | ≈ 2.7 万 | 用户特征 |
| `video_features_basic_pure.csv` | ≈ 0.76 万 | 视频基础特征 |
| `video_features_statistic_pure.csv` | ≈ 0.76 万 | 视频统计特征 |

### 3.3 特征说明

- **类别特征（35 个）**：`date`、`hourmin`、`tab`、用户活跃度/直播/作者标记、`onehot_feat0~17`、关注/粉丝/好友/注册天数分桶、`video_type`、`upload_dt`、`upload_type`、`visible_status`、`music_type`、`tag`；
- **数值特征（58 个）**：用户粉丝/关注/注册天数，视频时长/分辨率，以及视频侧统计量（曝光、播放、点赞、评论、分享、下载、举报、收藏等计数）；
- **标签（4 个）**：`is_click`、`is_like`、`is_follow`、`is_comment`。

特征共 93 维（35 类别 + 58 数值）。

---

## 4. 目录结构

```
Recommendation_KuaiRand/
├── docs/
│   └── project_overview.md            # 本文档
├── KuaiRand-Pure/
│   ├── data/                          # 原始 CSV 数据（入库）
│   ├── data_processed/                # 预处理产物（parquet + pkl）
│   ├── saved/                         # 训练产物（.h5 / PNG / 日志）
│   └── LICENSE
├── data_process.py                    # 数据预处理脚本
├── MMoE_model.py                      # MMoE 网络模型定义
├── main.py                            # 模型训练与评估
├── demo.py                            # 快排练习脚本（与建模链路无关）
├── .gitignore                         # 忽略大文件（data_processed 目录）
└── README.md
```

---

## 5. 关键实现细节与注意事项

### 5.1 特征编码

- 每个类别列独立 `LabelEncoder`，编码后加上全局 `offset`，得到全局唯一的 `feature_id`，方便后续直接作为 Embedding 索引；
- 每列额外预留一个 `UNK` 类，测试集/新数据中未见过的取值会映射为 UNK；
- 数值特征使用训练集拟合的 `StandardScaler`，测试集仅做 transform，避免数据泄漏。

### 5.2 大文件与版本控制

- `processed_X.parquet`（约 101 MB）超过 GitHub 单文件 100 MB 上限，已加入 `.gitignore`，**不入库**；完整复现需先运行 `data_process.py` 本地生成；
- 其余预处理产物（测试集 parquet、编码器 pkl）体积较小，已入库。

### 5.3 已知问题 / 建议

- `data_process.py` 顶部注释提到 `nrows=10000`，已过时——当前实际全量读取；
- 共享 Embedding 的 `cat_vocab_size=2500` 是硬编码。当前数据最大 feature_id 为 2413，恰好能覆盖；若扩充数据/特征导致 id 超限，需要根据 `feature_offsets.pkl` 动态计算 vocab 大小；
- `demo.py` 是快速排序练习脚本，与推荐建模无关，建议移出或归档；
- `__pycache__/MMoE_model.cpython-311.pyc` 被误提交到仓库，建议删除并加入 `.gitignore`；
- 模型以 `.h5`（legacy 格式）保存，Keras 新版本建议使用 `.keras`；
- `log_random_*` 随机曝光数据尚未接入，可作为后续去偏实验的对照数据。

---

## 6. 运行方式

依赖：`pandas`、`numpy`、`scikit-learn`、`joblib`、`pyarrow`、`tensorflow`、`matplotlib`。

```bash
# 1. 数据预处理（需原始 CSV 位于 KuaiRand-Pure/data/）
python data_process.py

# 2. 模型训练与评估（需先完成第 1 步）
python main.py

# 3. 单独验证模型结构/前向（读取前 10 条样本预测）
python MMoE_model.py
```

训练产物自动写入 `KuaiRand-Pure/saved/`：模型 `mmoe_model_<时间戳>.h5`、训练日志 `training_log_<时间戳>.log`、损失/AUC 曲线 `MMoE_model_loss_auc_train_val_<时间戳>.png`。

---

## 7. 训练结果

已完成两轮训练（2025-07-03），均在约第 15/16 epoch 因早停结束，测试集指标如下：

| 指标 | Run 1（215806） | Run 2（233116） |
| --- | --- | --- |
| 总 val_loss | 0.6820 | 0.6821 |
| Task1 点击 AUC | 0.7283 | 0.7290 |
| Task2 点赞 AUC | 0.8202 | 0.8232 |
| Task3 关注 AUC | 0.6651 | 0.6540 |
| Task4 评论 AUC | 0.6584 | 0.6749 |

观察：

- 点赞任务 AUC 最高（≈0.82），与行为分布及特征相关性一致；
- 关注、评论任务 AUC 较低且两轮之间波动（0.65–0.68），与标签稀疏及损失权重较低有关；
- 日志 `training_log_20250703_232935.log` 仅包含 "Loading data..."，该次运行未完成训练。

---

## 8. 后续计划与建议

1. **多任务结构升级**：尝试 PLE/CGC（渐进式分层抽取）替代基础 MMoE，或按任务相关性分组专家；
2. **特征工程**：加入序列特征（用户观看历史）、时间衰减、视频画像聚合特征；
3. **样本不均衡**：针对关注/评论使用 focal loss、负采样或任务独立阈值；
4. **Embedding 优化**：按特征分域设置不同 vocab/dim，或动态计算 vocab 以适配数据扩展；
5. **工程化**：接入 `log_random` 去偏数据、补充实验管理（wandb/mlflow）、统一配置与评估脚本；
6. **仓库卫生**：移除误提交的 `__pycache__`，考虑用 Git LFS 管理大文件。

---

## 9. 数据集引用

```bibtex
@inproceedings{gao2022kuairand,
  title = {KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos},
  author = {Gao, Chongming and Li, Shijun and Zhang, Yuan and Chen, Jiawei and Li, Biao and Lei, Wenqiang and Jiang, Peng and He, Xiangnan},
  url = {https://doi.org/10.1145/3511808.3557624},
  doi = {10.1145/3511808.3557624},
  booktitle = {Proceedings of the 31st ACM International Conference on Information and Knowledge Management},
  series = {CIKM '22},
  year = {2022},
  pages = {3953--3957}
}
```
