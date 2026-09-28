# 03 特征与输入张量

## 从字段到向量

本项目把字段分成类别特征和数值特征：

~~~text
35 个类别字段 → 各自 LabelEncoder + Embedding(dim=8)
58 个数值字段 → train-only StandardScaler
                 ↓
         35 × 8 + 58 = 338 维输入
~~~

这里的 93 是字段数量，不是 embedding 后的向量宽度。词表大小由 pipeline_meta.json 动态记录；如果加入 shadow_0，数值字段和输入宽度都会增加 1。

## 类别特征

类别字段不能直接把整数编号当成连续数值。项目为每个字段建立训练集词表，未知值映射到 UNK，再通过 embedding 学习稠密表示。不同字段使用 offset 后的统一 embedding 空间，实际词表规模由预处理产物决定。

## 数值特征

数值列用训练集拟合的 StandardScaler 做变换。验证集和测试集只能使用训练集统计量，避免把未来分布信息带回训练流程。

## 特征编码器的职责

编码器位于“拼接后的输入”与“多任务结构”之间：

~~~text
embedding + numeric concat
            ↓
   MLP / DCN / SENet
            ↓
       shared representation
~~~

它负责表示变换，不负责决定多个任务如何共享专家。后者属于多任务结构轴。

## 哪些字段不进模型

user_id、video_id、row_id 和原始日期字符串不会直接进入模型。日期会转换成 weekday 作为模型特征，原始日期仍保存在 sidecar，供切分和审计使用。

## 代码导航

- [config.py](../../config.py)：类别列、数值列和标签配置；
- [data_process.py](../../data_process.py)：编码、标准化、schema 和 pipeline_meta.json；
- [models/inputs.py](../../models/inputs.py)：embedding 分支和数值分支；
- [models/encoders/](../../models/encoders/)：MLP、DCN、SENet。

## 常见误读

- **误读**：date 进入模型时仍是原始 YYYYMMDD。  
  **纠正**：模型看到的是 weekday；原始日期只保留在 sidecar。
- **误读**：特征重要度越高，就越有因果作用。  
  **纠正**：置换重要度衡量固定模型下的预测依赖，不是因果效应。
- **误读**：当前缓存中的词表大小就是模型永远的常数。  
  **纠正**：词表由当前数据处理产物动态决定。

## 练习

在 pipeline_meta.json 中确认当前产物的字段数和词表规模，再比较加入 shadow_0 前后的输入宽度。为什么 shadow feature 必须在数据处理和训练时真实存在，不能在事后临时添加？

