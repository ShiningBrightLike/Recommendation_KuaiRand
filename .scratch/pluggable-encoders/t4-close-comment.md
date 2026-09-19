T4 已完成并合入 `main`。

- 提交：`7da8088` feat: add the DCN-v2 feature encoder
- 形态：`--encoder dcn` = 低秩矩阵交叉 2 层（rank 64）+ 并行深层分支（64），输出为 cross 与 deep 的拼接，保留自然宽度（339 + 64 = 403）
- 参数量：标准 schema（无影子特征）下编码器 108,900 参数；低秩交叉每层为 U、V（各 dim×rank）加一个 bias，因此开销随 rank 而非输入宽度增长
- 证据：
  - `python main.py --smoke --encoder dcn` 端到端跑通并落盘
  - 该 run 的 `model.keras` 在新进程用 `models.custom_objects()` 重载成功（356,530 参数、4 输出）
  - `metrics.json` 记录 `num_cross_layers=2 / rank=64 / deep_units=64`
  - `python baselines.py --models "mmoe+dcn"` 产出对照表行 `mmoe+dcn`
  - 测试覆盖：输出形状与概率范围、参数量（18 维输入下 5,860）、编码器超参覆盖、保存后重载一致
- 全量测试 37 条通过
