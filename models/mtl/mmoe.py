"""MMoE (Multi-gate Mixture-of-Experts) multi-task structure."""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable()
class MMoE(Layer):
    """MMoE block: shared experts + one softmax gate per task."""

    def __init__(self, units, num_experts, num_tasks, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = [Dense(units, activation="relu") for _ in range(num_experts)]
        self.gates = [Dense(num_experts, activation="softmax") for _ in range(num_tasks)]

    def call(self, inputs):
        expert_outputs = tf.stack([expert(inputs) for expert in self.experts], axis=1)
        outputs = []
        for gate in self.gates:
            gate_weights = tf.expand_dims(gate(inputs), axis=-1)
            weighted_output = tf.reduce_sum(expert_outputs * gate_weights, axis=1)
            outputs.append(weighted_output)
        return outputs

    def build(self, input_shape):
        for expert in self.experts:
            expert.build(input_shape)
        for gate in self.gates:
            gate.build(input_shape)
        super().build(input_shape)


def build_mmoe_structure(features, num_tasks=2, num_experts=8, units=64, tower_units=32):
    """MMoE over encoded features: shared experts, one gate and tower per task.

    Args:
        features: encoded feature representation, `[batch, encoder_output_dim]`.
        num_tasks: number of tasks/outputs.
        num_experts: number of shared experts.
        units: expert hidden size.
        tower_units: task-tower hidden size.
    """
    mmoe_outputs = MMoE(units=units, num_experts=num_experts, num_tasks=num_tasks)(
        features
    )

    task_outputs = []
    for i, out in enumerate(mmoe_outputs):
        tower = Dense(tower_units, activation="relu")(out)
        task_outputs.append(Dense(1, activation="sigmoid", name=f"output_{i + 1}")(tower))
    return task_outputs
