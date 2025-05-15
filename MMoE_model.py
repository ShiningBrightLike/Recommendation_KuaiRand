import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# 自定义 MMoE 层
# --------------------------
class MMoE(Layer):
    def __init__(self, units, num_experts, num_tasks, **kwargs):
        super(MMoE, self).__init__(**kwargs)
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = [Dense(units, activation='relu') for _ in range(num_experts)]
        self.gates = [Dense(num_experts, activation='softmax') for _ in range(num_tasks)]

    def call(self, inputs):
        expert_outputs = tf.stack([expert(inputs) for expert in self.experts], axis=1)  # (batch, num_experts, units)
        outputs = []
        for gate in self.gates:
            gate_weights = gate(inputs)  # (batch, num_experts)
            gate_weights = tf.expand_dims(gate_weights, axis=-1)  # (batch, num_experts, 1)
            weighted_output = tf.reduce_sum(expert_outputs * gate_weights, axis=1)  # (batch, units)
            outputs.append(weighted_output)
        return outputs

# --------------------------
# 构建 MMoE 模型
# --------------------------
def build_mmoe_model(input_dim, num_experts=8, num_tasks=2, units=64):
    inputs = Input(shape=(input_dim,), name='input')
    mmoe_outputs = MMoE(units=units, num_experts=num_experts, num_tasks=num_tasks)(inputs)

    task_outputs = []
    for i, out in enumerate(mmoe_outputs):
        tower = Dense(32, activation='relu')(out)
        final_out = Dense(1, activation='sigmoid', name=f'output_{i+1}')(tower)
        task_outputs.append(final_out)

    model = Model(inputs=inputs, outputs=task_outputs)
    return model

# --------------------------
# 模拟输入数据
# --------------------------
if __name__ == '__main__':
    input_dim = 50
    num_samples = 1024

    # 读取真实数据
    X = pd.read_parquet("processed_X.parquet").values.astype(np.float32)
    y = pd.read_parquet("processed_y.parquet")

    # 提取两个任务的标签（一共有四个）
    y_task1 = y.iloc[:, 0].values.reshape(-1, 1).astype(np.float32)
    y_task2 = y.iloc[:, 1].values.reshape(-1, 1).astype(np.float32)

    # 获取特征维度（自动推断）
    input_dim = X.shape[1]
    num_samples = X.shape[0]

    model = build_mmoe_model(input_dim=input_dim, num_experts=8, num_tasks=2, units=64)

    model.compile(
        optimizer='adam',
        loss={
            'output_1': 'binary_crossentropy',
            'output_2': 'binary_crossentropy',
        },
        metrics={
            'output_1': 'AUC',
            'output_2': 'AUC',
        }
    )

    model.summary()

    # 模型训练
    history = model.fit(
        X,
        {'output_1': y_task1, 'output_2': y_task2},
        epochs=30,
        batch_size=256
    )

    # 绘制 loss 曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['output_1_loss'], label='Task 1 Loss')
    plt.plot(history.history['output_2_loss'], label='Task 2 Loss')
    plt.plot(history.history['loss'], label='Total Loss', linestyle='--')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制 AUC 曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['output_1_auc'], label='Task 1 AUC')
    plt.plot(history.history['output_2_auc'], label='Task 2 AUC')
    plt.title('AUC Curve')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_auc.png')
    plt.show()
