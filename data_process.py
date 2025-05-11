import pandas as pd


# The "nrows=10000" argument reads the first 10000 lines of each file

df_train = pd.read_csv("KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv")
df_test = pd.read_csv("KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv")

user_features = pd.read_csv("KuaiRand-Pure/data/user_features_pure.csv")

video_features_basic = pd.read_csv("KuaiRand-Pure/data/video_features_basic_pure.csv")
video_features_statistics = pd.read_csv("KuaiRand-Pure/data/video_features_statistic_pure.csv")

# 打印检查列名
print("Train columns:", df_train.columns)
print("User features columns:", user_features.columns)
print("Video basic features columns:", video_features_basic.columns)
print("Video statistic features columns:", video_features_statistics.columns)

# 合并用户特征
df_merged = df_train.merge(user_features, on='user_id', how='left')

# 合并视频特征（基本 + 统计）
df_merged = df_merged.merge(video_features_basic, on='video_id', how='left')
df_merged = df_merged.merge(video_features_statistics, on='video_id', how='left')

# 查看合并结果
print("Merged Data Shape:", df_merged.shape)
print("Sample rows:\n", df_merged.head())

# 保存合并后的数据（可选）
df_merged.to_csv("KuaiRand-Pure/data/merged_train_sample.csv", index=False)