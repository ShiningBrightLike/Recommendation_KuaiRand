# 学习教程

从这里开始：[推荐精排学习地图](00-learning-map.md)。

教程采用双轨路线：

- 90 分钟快速导览：建立推荐全链路地图，理解曝光样本、时间切分、多任务模型和曝光内排序，并跑通 smoke；
- 深入学习与实验线：完整阅读六章、运行 examples/ 中的 toy 示例，再对照实验台账、ADR 和发布资产。

教程只把本项目已经实现的离线精排作为实践主线；召回、粗排、全量候选生成、线上服务和序列模型会在边界与扩展章节中说明。

六章主线完成后，进入 [深入参考层](reference/README.md)，阅读置换重要度、泄漏审计、bootstrap、ADR、扩展地图、FAQ、限制和阅读清单。

## 验证清单

在 Windows 的 env_tf 环境中，可以用 Python 自带的 unittest 运行仓库测试（该环境当前没有安装 pytest）：

~~~powershell
conda run -n env_tf python -m unittest discover -s tests -p "test_*.py"
~~~

教程示例不依赖原始数据：

~~~powershell
conda run -n env_tf python docs/learning/examples/sample_semantics.py
conda run -n env_tf python docs/learning/examples/ranking_metrics_toy.py
conda run -n env_tf python docs/learning/examples/mmoe_gate_toy.py
conda run -n env_tf python docs/learning/examples/time_split_leakage.py
~~~

