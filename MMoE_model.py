import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC
import pandas as pd
import numpy as np

# --------------------------------------------
# 自定义 MMoE 层
# --------------------------------------------
class MMoE(Layer):
    def __init__(self, units, num_experts, num_tasks, **kwargs):
        super(MMoE, self).__init__(**kwargs)
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = [Dense(units, activation='relu') for _ in range(num_experts)]
        self.gates = [Dense(num_experts, activation='softmax') for _ in range(num_tasks)]

    def call(self, inputs):
        expert_outputs = tf.stack([expert(inputs) for expert in self.experts], axis=1)
        outputs = []
        for gate in self.gates:
            gate_weights = tf.expand_dims(gate(inputs), axis=-1)
            weighted_output = tf.reduce_sum(expert_outputs * gate_weights, axis=1)
            outputs.append(weighted_output)
        return outputs

# --------------------------------------------
# 构建 MMoE 模型（带 Embedding）
# --------------------------------------------
def build_mmoe_model(categorical_cols, numeric_cols, cat_vocab_size=2500, embed_dim=8, num_experts=8, num_tasks=2, units=64):
    categorical_inputs = [Input(shape=(1,), name=col, dtype='int32') for col in categorical_cols]
    numeric_input = Input(shape=(len(numeric_cols),), name='numeric_input', dtype='float32')

    # 所有类别特征共享一个 Embedding 层
    shared_embedding = Embedding(input_dim=cat_vocab_size, output_dim=embed_dim, name="shared_embedding")

    embedded = [Flatten()(shared_embedding(inp)) for inp in categorical_inputs]
    cat_concat = Concatenate()(embedded)

    # 拼接数值特征
    all_features = Concatenate()([cat_concat, numeric_input])

    # MMoE + tower
    mmoe_outputs = MMoE(units=units, num_experts=num_experts, num_tasks=num_tasks)(all_features)

    task_outputs = []
    for i, out in enumerate(mmoe_outputs):
        tower = Dense(32, activation='relu')(out)
        final_out = Dense(1, activation='sigmoid', name=f'output_{i+1}')(tower)
        task_outputs.append(final_out)

    model = Model(inputs=categorical_inputs + [numeric_input], outputs=task_outputs)
    return model

# --------------------------
# 模拟输入数据
# --------------------------
if __name__ == '__main__':
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

    model.summary()

    # 拼接输入
    X = X_categorical + [X_numeric]
    output_1, output_2, output_3, output_4 = model.predict([x[0:10] for x in X])
    print(output_1.shape)
    print(output_2.shape)
    print(output_3.shape)
    print(output_4.shape)