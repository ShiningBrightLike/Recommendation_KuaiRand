"""MMoE (Multi-gate Mixture-of-Experts) multi-task model."""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.saving import register_keras_serializable

from models.inputs import build_shared_inputs


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


def build_mmoe_model(
    categorical_cols,
    numeric_cols,
    cat_vocab_size=2500,
    embed_dim=8,
    num_experts=8,
    num_tasks=2,
    units=64,
    tower_units=32,
):
    """Build the MMoE model with a shared embedding table for categorical inputs.

    Args:
        categorical_cols: names of the categorical features (one Input each).
        numeric_cols: names of the numeric features (one concatenated Input).
        cat_vocab_size: total vocabulary size across all categorical columns.
            Pass the value from `pipeline_meta.json` rather than hard-coding it.
        num_tasks: number of tasks/outputs.
        units: expert hidden size.
        tower_units: task-tower hidden size.
    """
    inputs, all_features = build_shared_inputs(
        categorical_cols, numeric_cols, cat_vocab_size, embed_dim
    )

    mmoe_outputs = MMoE(
        units=units, num_experts=num_experts, num_tasks=num_tasks
    )(all_features)

    task_outputs = []
    for i, out in enumerate(mmoe_outputs):
        tower = Dense(tower_units, activation="relu")(out)
        final_out = Dense(1, activation="sigmoid", name=f"output_{i + 1}")(tower)
        task_outputs.append(final_out)

    return Model(inputs=inputs, outputs=task_outputs)

