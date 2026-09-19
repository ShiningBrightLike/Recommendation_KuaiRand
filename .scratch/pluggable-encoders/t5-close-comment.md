T5 已完成并合入 `main`。

- 提交：`26ed0d9` feat: add the SENet feature encoder
- 形态：`--encoder senet` = 按特征域 squeeze-excite —— 每个域 squeeze 成一个均值，经 reduction 瓶颈与 sigmoid 得到该域的门，再按域整体重加权。输出保留输入宽度，只改变各域的贡献权重（这正是它与 Dense 类编码器的本质区别）。
- 域划分随 schema 变化（实测）：
  - 无影子特征：93 域（35 个类别域各 8 维 + 58 个数值域各 1 维）→ 编码器 8,695 参数、输出 338
  - 带 `shadow_0` 的 run（`t5-senet_smoke_20260914_233841`）：94 域 → 8,977 参数、输出 339
- 证据：
  - `python main.py --smoke --encoder senet` 端到端跑通并落盘；该 run 的 `model.keras` 在新进程用 `models.custom_objects()` 重载成功（221,469 参数、4 输出）
  - `metrics.json` 记录 `reduction=2`
  - `python baselines.py --models "mmoe+senet"` 产出对照表行 `mmoe+senet`
  - 测试覆盖输出形状与概率范围、域数量随 schema、参数量（4 域 22 参数 / 5 域 27 参数）；「每个已注册编码器保存后重载一致」的循环测试现在覆盖 `mlp`/`dcn`/`senet`
- 全量测试 39 条通过
