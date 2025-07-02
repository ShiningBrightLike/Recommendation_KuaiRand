import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MMoE_model import MMoE, build_mmoe_model
from tensorflow.keras.metrics import AUC

categorical_cols = [
    'date', 'hourmin', 'tab', 
    'user_active_degree', 'is_lowactive_period', 'is_live_streamer','is_video_author',
    'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3', 'onehot_feat4',
    'onehot_feat5', 'onehot_feat6', 'onehot_feat7', 'onehot_feat8', 'onehot_feat9',
    'onehot_feat10', 'onehot_feat11', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14',
    'onehot_feat15', 'onehot_feat16', 'onehot_feat17',
    'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range', 'register_days_range', 
    'video_type', 'upload_dt', 'upload_type', 'visible_status','music_type', 'tag'
]

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
    'outsite_share_all_cnt'
]

# 加载数据
df_X = pd.read_parquet("KuaiRand-Pure/data_processed/processed_X.parquet")
df_y = pd.read_parquet("KuaiRand-Pure/data_processed/processed_y.parquet")

# 准备输入
X_categorical = [df_X[col].astype('int32').values for col in categorical_cols]
X_numeric = df_X[numeric_cols].astype('float32').values
y_task1 = df_y.iloc[:, 0].values.reshape(-1, 1).astype(np.float32)
y_task2 = df_y.iloc[:, 1].values.reshape(-1, 1).astype(np.float32)
y_task3 = df_y.iloc[:, 2].values.reshape(-1, 1).astype(np.float32)
y_task4 = df_y.iloc[:, 3].values.reshape(-1, 1).astype(np.float32)

# 模型
model = build_mmoe_model(categorical_cols, numeric_cols, num_tasks=4)

model.compile(
    optimizer='adam',
    loss={'output_1': 'binary_crossentropy', 'output_2': 'binary_crossentropy', 'output_3': 'binary_crossentropy', 'output_4': 'binary_crossentropy'},
    loss_weights={'output_1': 1.0, 'output_2': 1.0, 'output_3': 1.0, 'output_4': 1.0},
    metrics={'output_1': AUC(name='auc'), 'output_2': AUC(name='auc'), 'output_3': AUC(name='auc'), 'output_4': AUC(name='auc')}
)

model.summary()

history = model.fit(X_categorical + [X_numeric], [y_task1, y_task2, y_task3, y_task4], batch_size=1024, epochs=50)


# 绘制 loss 曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['output_1_loss'], label='Task 1 Loss')
plt.plot(history.history['output_2_loss'], label='Task 2 Loss')
plt.plot(history.history['output_3_loss'], label='Task 3 Loss')
plt.plot(history.history['output_4_loss'], label='Task 4 Loss')
plt.plot(history.history['loss'], label='Total Loss', linestyle='--')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制 AUC 曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['output_1_auc'], label='Task 1 AUC')
plt.plot(history.history['output_2_auc'], label='Task 2 AUC')
plt.plot(history.history['output_3_auc'], label='Task 3 AUC')
plt.plot(history.history['output_4_auc'], label='Task 4 AUC')
plt.title('AUC Curve')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.savefig('MMoE_model_loss_auc.png', dpi=600)
plt.show()