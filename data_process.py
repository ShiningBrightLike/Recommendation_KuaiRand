import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import numpy as np
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
df_merged['date'] = pd.to_datetime(df_merged['date'], format='%Y%m%d').dt.dayofweek

# 查看合并结果
print("Merged Data Shape:", df_merged.shape)
print("Sample rows:\n", df_merged.head())

# 1. 缺失值处理
df_processed = df_merged.copy()
df_processed.fillna(-1, inplace=True)  # 用 -1 填充所有缺失值

# 2. 编码类别特征
categorical_cols = [
    'date', 'hourmin', 'tab', 
    'user_active_degree', 'is_lowactive_period', 'is_live_streamer','is_video_author',
    'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3', 'onehot_feat4',
    'onehot_feat5', 'onehot_feat6', 'onehot_feat7', 'onehot_feat8', 'onehot_feat9',
    'onehot_feat10', 'onehot_feat11', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14',
    'onehot_feat15', 'onehot_feat16', 'onehot_feat17',
    'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range', 'register_days_range', 
    'video_type', 'upload_dt', 'upload_type', 'visible_status','music_type', 'tag'] # author_id music_id 暂时不处理 

# 初始化
label_encoders = {}
feature_offsets = {}
offset = 0

# 对每列进行LabelEncoder，并加上全局偏移量形成feature_id
for col in categorical_cols:
    # 获取所有唯一值并添加'UNK'标记
    unique_values = df_processed[col].astype(str).unique()
    unique_values = np.append(unique_values, 'UNK')

    le = LabelEncoder()
    le.fit(unique_values)

    # 转换数据，将'UNK'标记映射为最高编码值
    df_processed[col] = le.transform(df_processed[col].astype(str)) + offset
    
    # 保存编码器
    label_encoders[col] = le
    # 保存偏移量
    feature_offsets[col] = offset
    # 更新偏移量（+1是为了预留'UNK'的位置）
    offset += len(le.classes_)

"""
# 对新数据进行编码
for col in categorical_cols:
    # 先将新数据中不存在的标签替换为'UNK'
    new_data[col] = new_data[col].astype(str).apply(lambda x: x if x in label_encoders[col].classes_ else 'UNK')
    # 然后进行转换
    new_data[col] = label_encoders[col].transform(new_data[col]) + feature_offsets[col]
"""
# 3. 数值标准化
numeric_cols = [
    'follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days',
    'video_duration', 'server_width', 'server_height',
    'counts', 'show_cnt', 'show_user_num', 'play_cnt',
    'play_user_num', 'play_duration', 'complete_play_cnt',
    'complete_play_user_num', 'valid_play_cnt', 'valid_play_user_num',
    'long_time_play_cnt', 'long_time_play_user_num', 'short_time_play_cnt',
    'short_time_play_user_num', 'play_progress', 'comment_stay_duration',
    'like_cnt', 'like_user_num', 'click_like_cnt', 'double_click_cnt',
    'cancel_like_cnt', 'cancel_like_user_num', 'comment_cnt',
    'comment_user_num', 'direct_comment_cnt', 'reply_comment_cnt',
    'delete_comment_cnt', 'delete_comment_user_num', 'comment_like_cnt',
    'comment_like_user_num', 'follow_cnt', 'follow_user_num1',
    'cancel_follow_cnt', 'cancel_follow_user_num', 'share_cnt',
    'share_user_num', 'download_cnt', 'download_user_num', 'report_cnt',
    'report_user_num', 'reduce_similar_cnt', 'reduce_similar_user_num',
    'collect_cnt', 'collect_user_num', 'cancel_collect_cnt',
    'cancel_collect_user_num', 'direct_comment_user_num',
    'reply_comment_user_num', 'share_all_cnt', 'share_all_user_num',
    'outsite_share_all_cnt']

scaler = StandardScaler()
df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

# 4. 标签列（多任务）
label_cols = ['is_click', 'is_like', 'is_follow', 'is_comment']
y = df_processed[label_cols]

# 5. 特征列
X = df_processed[categorical_cols + numeric_cols]

# 打印处理结果
print("Processed feature shape:", X.shape)
print("Processed label shape:", y.shape)

# 保存特征和标签
X.to_parquet("KuaiRand-Pure/data_processed/processed_X.parquet")
y.to_parquet("KuaiRand-Pure/data_processed/processed_y.parquet")

# 保存编码器和标准化器
joblib.dump(label_encoders, "KuaiRand-Pure/data_processed/label_encoders.pkl")
joblib.dump(scaler, "KuaiRand-Pure/data_processed/scaler.pkl")
joblib.dump(feature_offsets, "KuaiRand-Pure/data_processed/feature_offsets.pkl")  # 可选：用于后续Embedding映射
