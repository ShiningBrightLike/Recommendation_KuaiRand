"""Single-task MLP baseline: one independent model per task."""

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from models.inputs import build_shared_inputs


def build_single_task_model(
    categorical_cols,
    numeric_cols,
    cat_vocab_size=2500,
    embed_dim=8,
    units=64,
    tower_units=32,
):
    """Single-task MLP baseline using the same embedding branch (output_1)."""
    inputs, all_features = build_shared_inputs(
        categorical_cols, numeric_cols, cat_vocab_size, embed_dim
    )
    hidden = Dense(units, activation="relu", name="hidden")(all_features)
    tower = Dense(tower_units, activation="relu", name="tower_1")(hidden)
    out = Dense(1, activation="sigmoid", name="output_1")(tower)
    return Model(inputs=inputs, outputs=out)

