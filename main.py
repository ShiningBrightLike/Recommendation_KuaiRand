import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MMoE_model import MMoE, build_mmoe_model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping
import logging
import sys
import time
from tensorflow.keras.callbacks import Callback

# logging配置
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'KuaiRand-Pure/saved/training_log_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 自定义回调函数用于实时记录训练日志
class TrainingLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info(f"Epoch {epoch + 1}/{self.params['epochs']}")
        logger.info(f"Train - Loss: {logs.get('loss', 0):.4f}, "
                  f"Task1 AUC: {logs.get('output_1_auc', 0):.4f}, "
                  f"Task2 AUC: {logs.get('output_2_auc', 0):.4f}, "
                  f"Task3 AUC: {logs.get('output_3_auc', 0):.4f}, "
                  f"Task4 AUC: {logs.get('output_4_auc', 0):.4f}")
        logger.info(f"Val - Loss: {logs.get('val_loss', 0):.4f}, "
                  f"Task1 AUC: {logs.get('val_output_1_auc', 0):.4f}, "
                  f"Task2 AUC: {logs.get('val_output_2_auc', 0):.4f}, "
                  f"Task3 AUC: {logs.get('val_output_3_auc', 0):.4f}, "
                  f"Task4 AUC: {logs.get('val_output_4_auc', 0):.4f}")

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
logger.info("Loading data...")
df_X = pd.read_parquet("KuaiRand-Pure/data_processed/processed_X.parquet")
df_y = pd.read_parquet("KuaiRand-Pure/data_processed/processed_y.parquet")
df_X_test = pd.read_parquet("KuaiRand-Pure/data_processed/processed_X_test.parquet")
df_y_test = pd.read_parquet("KuaiRand-Pure/data_processed/processed_y_test.parquet")
logger.info("Data loaded successfully.")

# 准备输入
X_categorical = [df_X[col].astype('int32').values for col in categorical_cols]
X_numeric = df_X[numeric_cols].astype('float32').values
y_task1 = df_y.iloc[:, 0].values.reshape(-1, 1).astype(np.float32)
y_task2 = df_y.iloc[:, 1].values.reshape(-1, 1).astype(np.float32)
y_task3 = df_y.iloc[:, 2].values.reshape(-1, 1).astype(np.float32)
y_task4 = df_y.iloc[:, 3].values.reshape(-1, 1).astype(np.float32)

# 正负样本占比
logger.info("Positive sample ratios:")
logger.info(f"task1 positive sample ratio:{np.sum(y_task1) / len(y_task1)}")
logger.info(f"task2 positive sample ratio:{np.sum(y_task2) / len(y_task2)}")
logger.info(f"task3 positive sample ratio:{np.sum(y_task3) / len(y_task3)}")
logger.info(f"task4 positive sample ratio:{np.sum(y_task4) / len(y_task4)}")

# 准备测试集输入
X_test_categorical = [df_X_test[col].astype('int32').values for col in categorical_cols]
X_test_numeric = df_X_test[numeric_cols].astype('float32').values
y_test_task1 = df_y_test.iloc[:, 0].values.reshape(-1, 1).astype(np.float32)
y_test_task2 = df_y_test.iloc[:, 1].values.reshape(-1, 1).astype(np.float32)
y_test_task3 = df_y_test.iloc[:, 2].values.reshape(-1, 1).astype(np.float32)
y_test_task4 = df_y_test.iloc[:, 3].values.reshape(-1, 1).astype(np.float32)

# 模型
logger.info("Building MMoE model...")
model = build_mmoe_model(categorical_cols, numeric_cols, num_tasks=4)

model.compile(
  optimizer='adam',
  loss={'output_1': 'binary_crossentropy', 'output_2': 'binary_crossentropy', 'output_3': 'binary_crossentropy', 'output_4': 'binary_crossentropy'},
  loss_weights={'output_1': 1.0, 'output_2': 1.0, 'output_3': 0.5, 'output_4': 0.1},
  metrics={'output_1': AUC(name='auc'), 'output_2': AUC(name='auc'), 'output_3': AUC(name='auc'), 'output_4': AUC(name='auc')}
)

early_stopping = EarlyStopping(
  monitor='val_loss',    # 监控验证集总损失
  patience=5,        # 连续5个epoch性能未提升则停止
  restore_best_weights=True # 恢复最佳模型权重
)

training_logger = TrainingLogger()

logger.info("Starting model training...")
try:
    history = model.fit(
      X_categorical + [X_numeric], 
      [y_task1, y_task2, y_task3, y_task4], 
      validation_data=(
        X_test_categorical + [X_test_numeric],
        [y_test_task1, y_test_task2, y_test_task3, y_test_task4]
      ),
      batch_size=1024, 
      epochs=30,
      callbacks=[early_stopping, training_logger] # 加入早停回调和日志回调
    )
except Exception as e:
    logger.error(f"Training failed with error: {str(e)}")
    raise

# 保存模型
logger.info("Saving model...")
model.save(f"KuaiRand-Pure/saved/mmoe_model_{time.strftime('%Y%m%d%H%M%S')}.h5")
logger.info("Model saved successfully.")

# 单独评估测试集
logger.info("Evaluating on test set...")
test_results = model.evaluate(
  X_test_categorical + [X_test_numeric],
  [y_test_task1, y_test_task2, y_test_task3, y_test_task4],
  batch_size=1024
)

# 打印测试集评估结果
logger.info("\nTest set evaluation:")
for i, metric in enumerate(model.metrics_names):
    logger.info(f"{metric}: {test_results[i]:.4f}")
logger.info("\nAUC for each task:")
logger.info(f"output_1 AUC: {test_results[5]:.4f}") # output_1_auc
logger.info(f"output_2 AUC: {test_results[6]:.4f}") # output_2_auc
logger.info(f"output_3 AUC: {test_results[7]:.4f}") # output_3_auc
logger.info(f"output_4 AUC: {test_results[8]:.4f}") # output_4_auc

plt.figure(figsize=(16, 6))

# ========== 1. Loss 曲线 ==========
plt.subplot(1, 2, 1)

# 训练损失（train）
plt.plot(history.history['output_1_loss'], label='Task 1 Train Loss', color='blue', alpha=0.3)
plt.plot(history.history['output_2_loss'], label='Task 2 Train Loss', color='orange', alpha=0.3)
plt.plot(history.history['output_3_loss'], label='Task 3 Train Loss', color='green', alpha=0.3)
plt.plot(history.history['output_4_loss'], label='Task 4 Train Loss', color='red', alpha=0.3)
plt.plot(history.history['loss'], label='Total Train Loss', linestyle='--', color='black')

# 验证损失（val）
plt.plot(history.history['val_output_1_loss'], label='Task 1 Val Loss', color='blue')
plt.plot(history.history['val_output_2_loss'], label='Task 2 Val Loss', color='orange')
plt.plot(history.history['val_output_3_loss'], label='Task 3 Val Loss', color='green')
plt.plot(history.history['val_output_4_loss'], label='Task 4 Val Loss', color='red')
plt.plot(history.history['val_loss'], label='Total Val Loss', linestyle='--', color='black')

plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# ========== 2. AUC 曲线 ==========
plt.subplot(1, 2, 2)

# 训练 AUC（train）
plt.plot(history.history['output_1_auc'], label='Task 1 Train AUC', color='blue', alpha=0.3)
plt.plot(history.history['output_2_auc'], label='Task 2 Train AUC', color='orange', alpha=0.3)
plt.plot(history.history['output_3_auc'], label='Task 3 Train AUC', color='green', alpha=0.3)
plt.plot(history.history['output_4_auc'], label='Task 4 Train AUC', color='red', alpha=0.3)

# 验证 AUC（val）
plt.plot(history.history['val_output_1_auc'], label='Task 1 Val AUC', color='blue')
plt.plot(history.history['val_output_2_auc'], label='Task 2 Val AUC', color='orange')
plt.plot(history.history['val_output_3_auc'], label='Task 3 Val AUC', color='green')
plt.plot(history.history['val_output_4_auc'], label='Task 4 Val AUC', color='red')

plt.title('Training & Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.savefig(f'KuaiRand-Pure/saved/MMoE_model_loss_auc_train_val_{time.strftime("%Y%m%d%H%M%S")}.png', dpi=600)
logger.info("Training curves saved to KuaiRand-Pure/saved/MMoE_model_loss_auc_train_val.png")
plt.show()
