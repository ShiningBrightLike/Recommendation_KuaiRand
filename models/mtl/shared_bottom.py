"""Shared-Bottom multi-task baseline: one trunk, one tower per task."""

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from models.inputs import build_shared_inputs


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
    inputs, all_features = build_shared_inputs(
        categorical_cols, numeric_cols, cat_vocab_size, embed_dim
    )
    shared = Dense(bottom_units, activation="relu", name="shared_bottom")(all_features)
    outputs = []
    for i in range(num_tasks):
        tower = Dense(tower_units, activation="relu", name=f"tower_{i + 1}")(shared)
        outputs.append(Dense(1, activation="sigmoid", name=f"output_{i + 1}")(tower))
    return Model(inputs=inputs, outputs=outputs)

