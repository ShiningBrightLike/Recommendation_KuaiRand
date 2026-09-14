"""Shared input branch and feature layout used by every ranking model."""

from typing import NamedTuple

from tensorflow.keras.layers import Concatenate, Embedding, Flatten, Input


class FeatureLayout(NamedTuple):
    """How the concatenated feature vector is partitioned into fields.

    `field_dims[i]` is the width of field i: one field per categorical feature
    (width = embedding dim) followed by one field per numeric feature (width 1).
    Feature encoders use it to work per field instead of per dimension.
    """

    field_dims: tuple
    categorical_cols: tuple
    numeric_cols: tuple
    embed_dim: int


def build_shared_inputs(categorical_cols, numeric_cols, cat_vocab_size, embed_dim):
    """Build the shared input branch used by all ranking models/baselines.

    Returns `(inputs, all_features, layout)`: `inputs` is the model input list
    (one int32 Input per categorical column, followed by one float32 Input
    holding all numeric features), `all_features` is the concatenated dense
    vector that the feature encoder consumes, and `layout` describes how that
    vector is partitioned into fields.
    """
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

    layout = FeatureLayout(
        field_dims=tuple([embed_dim] * len(categorical_cols) + [1] * len(numeric_cols)),
        categorical_cols=tuple(categorical_cols),
        numeric_cols=tuple(numeric_cols),
        embed_dim=embed_dim,
    )
    return categorical_inputs + [numeric_input], all_features, layout

