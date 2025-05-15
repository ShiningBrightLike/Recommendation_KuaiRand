import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
import numpy as np

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

    # 虚拟输入和标签
    X = np.random.rand(num_samples, input_dim).astype(np.float32)
    y_task1 = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)
    y_task2 = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)

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
    model.fit(
        X,
        {'output_1': y_task1, 'output_2': y_task2},
        epochs=5,
        batch_size=32
    )
