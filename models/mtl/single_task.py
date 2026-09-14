"""Single-task MLP baseline: one independent model per task."""

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from models.encoders import DEFAULT_ENCODER, create_encoder
from models.inputs import build_shared_inputs


def build_single_task_model(
    categorical_cols,
    numeric_cols,
    cat_vocab_size=2500,
    embed_dim=8,
    units=64,
    tower_units=32,
    encoder=DEFAULT_ENCODER,
    encoder_overrides=None,
):
    """One task's model with its own feature encoder, hidden layer and tower.

    The baseline stays "fully independent": each task gets its own encoder
    instance, so nothing is shared across tasks.
    """
    inputs, all_features, layout = build_shared_inputs(
        categorical_cols, numeric_cols, cat_vocab_size, embed_dim
    )
    encoded = create_encoder(encoder, layout.field_dims, **(encoder_overrides or {}))(
        all_features
    )
    hidden = Dense(units, activation="relu", name="hidden")(encoded)
    tower = Dense(tower_units, activation="relu", name="tower_1")(hidden)
    out = Dense(1, activation="sigmoid", name="output_1")(tower)
    return Model(inputs=inputs, outputs=out)
