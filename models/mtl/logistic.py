"""Logistic-regression baseline: one weight per one-hot/numeric feature."""

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from models.inputs import build_shared_inputs


def build_logistic_model(
    categorical_cols,
    numeric_cols,
    cat_vocab_size=2500,
    num_tasks=2,
    encoder=None,
):
    """Logistic-regression baseline: one weight per one-hot/numeric feature.

    Categorical ids are embedded with dim=1, so the concatenated features are
    exactly a one-hot vector, and each task is a single Dense(1) layer. Having
    no hidden representation is what this baseline measures, so it rejects a
    feature encoder instead of silently ignoring one.
    """
    if encoder is not None:
        raise ValueError(
            "the logistic baseline has no feature encoder; it is one weight per "
            "feature by definition"
        )
    inputs, all_features, _ = build_shared_inputs(
        categorical_cols, numeric_cols, cat_vocab_size, embed_dim=1
    )
    outputs = [
        Dense(1, activation="sigmoid", name=f"output_{i + 1}")(all_features)
        for i in range(num_tasks)
    ]
    return Model(inputs=inputs, outputs=outputs)
