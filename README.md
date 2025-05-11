```markdown
# Recommendation_KuaiRand

## 基于 KuaiRand 的推荐系统

本项目基于 KuaiRand 数据集，构建用于点击、点赞、关注、评论等多任务反馈预测的推荐系统模型。当前已完成数据加载与样本拼接的基础处理流程。

---

## 📁 项目结构

```

Recommendation\_KuaiRand/
├── KuaiRand-Pure/
│   ├── data/
│   │   ├── log\_standard\_4\_08\_to\_4\_21\_pure.csv
│   │   ├── user\_features\_pure.csv
│   │   ├── video\_features\_basic\_pure.csv
│   │   ├── video\_features\_statistic\_pure.csv
│   │   └── merged\_train\_sample.csv       # ← 脚本执行后生成
├── sample\_merge.py                        # 样本拼接脚本
└── README.md

````

---

## ✅ 已完成工作

- [x] 加载行为日志、用户特征、视频特征数据
- [x] 将用户和视频特征合并至行为数据，生成训练样本
- [ ] 特征处理（缺失值填充、编码等）
- [ ] 构建 CVR 预测模型（点击/点赞/评论等）
- [ ] 模型评估与优化

---

## 🚀 使用说明

1. 确保将原始数据放置在 `KuaiRand-Pure/data/` 目录下；
2. 运行样本拼接脚本：

```bash
python sample_merge.py
````

3. 合并结果将保存在：

```
KuaiRand-Pure/data/merged_train_sample.csv
```

---

## 📚 数据集引用

本项目使用的 KuaiRand 数据集来自 CIKM 2022：

```bibtex
@inproceedings{gao2022kuairand,
  title = {KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos},
  author = {Gao, Chongming and Li, Shijun and Zhang, Yuan and Chen, Jiawei and Li, Biao and Lei, Wenqiang and Jiang, Peng and He, Xiangnan},
  url = {https://doi.org/10.1145/3511808.3557624},
  doi = {10.1145/3511808.3557624},
  booktitle = {Proceedings of the 31st ACM International Conference on Information and Knowledge Management},
  series = {CIKM '22},
  location = {Atlanta, GA, USA},
  numpages = {5},
  year = {2022},
  pages = {3953–3957}
}
```

---

## 📌 后续计划

* 模块化数据处理与建模流程
* 支持多反馈目标的多任务学习
* 引入深度模型（如 Transformer）进行序列建模
* 支持线上推理与实验评估

---

欢迎交流与贡献 👋

```

