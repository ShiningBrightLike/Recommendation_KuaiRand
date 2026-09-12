"""MMoE (Multi-gate Mixture-of-Experts) multi-task model.

Input convention:
    - one int32 Input per categorical feature (Embedding indices)
    - one float32 Input with all numeric features
Output convention:
    - outputs named output_1 ... output_N, in the same order as the task list
      passed by the caller (the pipeline uses config.LABEL_COLS).

`cat_vocab_size` must be strictly larger than the largest embedding index used.
The preprocessing pipeline computes it dynamically and records it in
`data_processed/pipeline_meta.json`; pass that value in when building the model.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Embedding,
    Flatten,
    Input,
    Layer,
)
from tensorflow.keras.models import Model
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


def _shared_inputs(categorical_cols, numeric_cols, cat_vocab_size, embed_dim):
    """Build the shared input branch used by all ranking models/baselines."""
    categorical_inputs = [
        Input(shape=(1,), name=col, dtype="int32") for col in categorical_cols
    ]
    numeric_input = Input(shape=(len(numeric_cols),), name="numeric_input", dtype="float32")
    shared_embedding = Embedding(
        input_dim=cat_vocab_size, output_dim=embed_dim, name="shared_embedding"
    )
    embedded = [Flatten()(shared_embedding(inp)) for inp in categorical_inputs]
    cat_concat = Concatenate()(embedded)
    all_features = Concatenate()([cat_concat, numeric_input])
    return categorical_inputs + [numeric_input], all_features


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
    inputs, all_features = _shared_inputs(
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


def build_shared_bottom_model(
    categorical_cols,
    numeric_cols,
    cat_vocab_size=2500,
    embed_dim=8,
    num_tasks=2,
    bottom_units=64,
    tower_units=32,
):
    """Shared-Bottom multi-task baseline: one trunk, one tower per task."""
    inputs, all_features = _shared_inputs(
        categorical_cols, numeric_cols, cat_vocab_size, embed_dim
    )
    shared = Dense(bottom_units, activation="relu", name="shared_bottom")(all_features)
    outputs = []
    for i in range(num_tasks):
        tower = Dense(tower_units, activation="relu", name=f"tower_{i + 1}")(shared)
        outputs.append(Dense(1, activation="sigmoid", name=f"output_{i + 1}")(tower))
    return Model(inputs=inputs, outputs=outputs)


def build_single_task_model(
    categorical_cols,
    numeric_cols,
    cat_vocab_size=2500,
    embed_dim=8,
    units=64,
    tower_units=32,
):
    """Single-task MLP baseline using the same embedding branch (output_1)."""
    inputs, all_features = _shared_inputs(
        categorical_cols, numeric_cols, cat_vocab_size, embed_dim
    )
    hidden = Dense(units, activation="relu", name="hidden")(all_features)
    tower = Dense(tower_units, activation="relu", name="tower_1")(hidden)
    out = Dense(1, activation="sigmoid", name="output_1")(tower)
    return Model(inputs=inputs, outputs=out)


def build_logistic_model(
    categorical_cols,
    numeric_cols,
    cat_vocab_size=2500,
    num_tasks=2,
):
    """Logistic-regression baseline: one weight per one-hot/numeric feature.

    Categorical ids are embedded with dim=1, so the concatenated features are
    exactly a one-hot vector, and each task is a single Dense(1) layer.
    """
    inputs, all_features = _shared_inputs(
        categorical_cols, numeric_cols, cat_vocab_size, embed_dim=1
    )
    outputs = [
        Dense(1, activation="sigmoid", name=f"output_{i + 1}")(all_features)
        for i in range(num_tasks)
    ]
    return Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    # Lightweight structural smoke test with random inputs (no data required).
    # A full forward pass against real data is exercised by `main.py`.
    import numpy as np

    from config import CATEGORICAL_COLS, LABEL_COLS, NUMERIC_COLS

    vocab_size = 256
    model = build_mmoe_model(
        categorical_cols=CATEGORICAL_COLS,
        numeric_cols=NUMERIC_COLS,
        cat_vocab_size=vocab_size,
        num_tasks=len(LABEL_COLS),
    )
    model.summary()

    rng = np.random.default_rng(0)
    n_samples = 4
    x_cat = [
        rng.integers(0, vocab_size, size=n_samples).astype("int32")
        for _ in CATEGORICAL_COLS
    ]
    x_num = rng.normal(size=(n_samples, len(NUMERIC_COLS))).astype("float32")
    preds = model.predict(x_cat + [x_num])
    for task, pred in zip(LABEL_COLS, preds):
        print(f"{task}: {pred.shape}")
