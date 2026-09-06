# KuaiRand 多目标 CVR 预测（MMoE）项目梳理

> 本文档基于仓库当前代码与产物整理，描述项目目标、技术方案、数据链路、实现细节与已有结果。

---

## 1. 项目概述

本项目基于 **KuaiRand**（CIKM 2022）公开数据集，构建短视频场景下的**多目标 CVR/反馈预测模型**，同时预测用户对视频的**点击（click）、点赞（like）、关注（follow）、评论（comment）**四个行为。

当前实现状态：

- 数据预处理管线已完成：原始 CSV → 样本拼接 → 特征编码/归一化 → parquet + 编码器产物；
- 已实现 **MMoE（Multi-gate Mixture-of-Experts）** 多任务模型并完成两轮训练/评估；
- 已切换到 **train/val/test 时间切分协议**：验证集（4/16–4/21）从训练日志内切出，测试集（4/22–5/08）仅在训练结束后评估一次；
- 新增单一配置源 `config.py` 与动态 `cat_vocab_size`（写于 `pipeline_meta.json`），特征清单不再三处重复；
- 每次运行的产物（`model.keras` / `metrics.json` / `curves.png` / `training.log`）统一保存到 `KuaiRand-Pure/saved/runs/<run>/`；`KuaiRand-Pure/saved/` 根目录下为旧协议历史产物。

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
| 随机种子 | 2025（可通过 `--seed` 覆盖） |
| 验证集 | 训练日志尾部按时间切分（4/16–4/21，约 19.1 万行，行数见 `pipeline_meta.json`） |
| 测试集 | 仅最终评估一次（4/22–5/08，29.5 万行） |
| 运行产物 | `saved/runs/<tag>_<时间戳>/`：`model.keras` + `metrics.json` + `curves.png` + `training.log` |

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
  ② 训练日志按 date ≥ 20220416 切出验证集（train / val）
  ③ date → 星期几；缺失值统一填 -1
  ④ 35 个类别特征：仅在训练集 fit LabelEncoder + 全局偏移量 → feature_id（预留 UNK），val/test 复用并映射未见值到 UNK
  ⑤ 58 个数值特征：StandardScaler（训练集 fit，val/test 仅 transform）
  ⑥ 提取 4 个标签列；记录总 vocab 与各行数到 pipeline_meta.json
                    │
                    ▼  KuaiRand-Pure/data_processed/
  processed_X.parquet / processed_y.parquet          （训练集）
  processed_X_val.parquet / processed_y_val.parquet  （验证集）
  processed_X_test.parquet / processed_y_test.parquet（测试集）
  label_encoders.pkl / scaler.pkl / feature_offsets.pkl / pipeline_meta.json
                    │
                    ▼  main.py
  MMoE 四任务训练（seed 固定）→ 早停只盯 val → 测试集最终评估一次
                    ▼
  KuaiRand-Pure/saved/runs/<run>/（model.keras / metrics.json / curves.png / training.log）
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
│   ├── data_processed/                # 预处理产物（本地生成，不入库）
│   ├── saved/                         # 训练产物（本地生成，不入库；每次运行一个 runs/<run>/ 子目录）
│   └── LICENSE
├── config.py                          # 特征清单、路径与超参数唯一配置源
├── data_process.py                    # 数据预处理脚本
├── MMoE_model.py                      # MMoE 网络模型定义
├── main.py                            # 模型训练与评估
├── requirements.txt                   # 锁版本依赖（env_tf）
├── .gitignore                         # 忽略 data_processed 与 saved/runs
└── README.md
```

---

## 5. 关键实现细节与注意事项

### 5.1 特征编码

- 每个类别列独立 `LabelEncoder`，编码后加上全局 `offset`，得到全局唯一的 `feature_id`，方便后续直接作为 Embedding 索引；
- 每列额外预留一个 `UNK` 类，测试集/新数据中未见过的取值会映射为 UNK；
- 数值特征使用训练集拟合的 `StandardScaler`，测试集仅做 transform，避免数据泄漏。

### 5.2 大文件与版本控制

- 生成类产物（`data_processed/` 全部内容、`saved/` 下的模型/曲线/日志）已**停止入库**并加入 `.gitignore`，仓库只保留源码、文档与原始数据 CSV；
- 完整复现请运行 `python data_process.py` 本地重新生成全部 parquet/pkl/meta（当前实现约 30 秒）；
- 说明：停止跟踪只影响后续提交；`git` 历史中已提交过的旧版本仍占用仓库体积，如需彻底瘦身需重写历史（风险高，暂不建议）。

### 5.3 已知问题 / 建议

- ~~`data_process.py` 顶部过时的 `nrows=10000` 注释~~ 已解决（脚本已重写）；
- ~~`cat_vocab_size=2500` 硬编码~~ 已解决：vocab 由编码器动态算出并写入 `pipeline_meta.json`（当前 2385），`main.py` 直接读取；
- `demo.py` 当前不存在于磁盘/未入库，无清理对象（IDE 中若仍有该标签属过期状态）；
- ~~误提交的 `__pycache__`~~ 已解决：`.gitignore` 已忽略，当前跟踪列表中无 pyc；
- ~~`.h5` legacy 保存格式~~ 已解决：新运行保存为 Keras 3 原生 `model.keras`；
- `log_random_*` 随机曝光数据尚未接入，可作为后续去偏实验的对照数据。

---

## 6. 运行方式

依赖：`tensorflow`、`pandas`、`numpy`、`scikit-learn`、`joblib`、`pyarrow`、`matplotlib`，版本锁定见 `requirements.txt`。标准环境为 conda 环境 `env_tf`（Python 3.11，CPU）。

```bash
# 0. 使用标准环境
conda activate env_tf

# 1. 数据预处理（生成 train/val/test parquet + pipeline_meta.json）
python data_process.py

# 2. 快速自检（每份数据取前 2048 行、1 个 epoch）
python main.py --smoke

# 3. 正式训练与评估（早停看 val，测试集最后评估一次）
python main.py

# 4. 模型结构轻量自检（随机输入，不需要数据文件）
python MMoE_model.py
```

正式训练产物自动写入 `KuaiRand-Pure/saved/runs/<tag>_<时间戳>/`：`model.keras`、`metrics.json`（含种子、超参、正样本占比、逐 epoch history 与测试集指标）、`curves.png`、`training.log`。常用覆盖参数：`--seed`、`--epochs`、`--batch-size`、`--patience`、`--tag`。

---

## 7. 训练结果

### 7.1 新协议基线（可复现，2026-09-07）

运行：`baseline-v1_20260907_001737`（seed=2025，best epoch=9，早停于 epoch 13；train 950,310 / val 190,802 / test 295,497）。

| 指标（测试集 4/22–5/08） | Baseline v1 |
| --- | --- |
| 总 loss | 0.6914 |
| 点击 AUC | 0.7223 |
| 点赞 AUC | 0.8090 |
| 关注 AUC | 0.7110 |
| 评论 AUC | 0.6495 |

完整逐 epoch 历史与配置见 `KuaiRand-Pure/saved/runs/baseline-v1_20260907_001737/metrics.json`。

### 7.2 旧协议历史结果（仅参考）

> ⚠️ 旧协议直接使用 4/22–5/08 测试集做早停，成绩偏乐观；下表仅作历史记录。

旧协议下两轮训练（2025-07-03）均在约第 15/16 epoch 因早停结束：

| 指标 | Run 1（215806） | Run 2（233116） |
| --- | --- | --- |
| 总 val_loss | 0.6820 | 0.6821 |
| Task1 点击 AUC | 0.7283 | 0.7290 |
| Task2 点赞 AUC | 0.8202 | 0.8232 |
| Task3 关注 AUC | 0.6651 | 0.6540 |
| Task4 评论 AUC | 0.6584 | 0.6749 |

观察：新协议下关注任务 AUC（0.7110）明显优于旧协议（0.6540–0.6651），点赞/点击略降——早停不再“看到”测试集，测试指标更可信；评论任务仍是最弱项，可作为里程碑 2 的优化重点。

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
