"""Shared input branch used by every ranking model and baseline."""

from tensorflow.keras.layers import Concatenate, Embedding, Flatten, Input


def build_shared_inputs(categorical_cols, numeric_cols, cat_vocab_size, embed_dim):
    """Build the shared input branch used by all ranking models/baselines.

    Returns `(inputs, all_features)`: `inputs` is the model input list (one
    int32 Input per categorical column, followed by one float32 Input holding
    all numeric features) and `all_features` is the concatenated dense vector
    that the downstream model consumes.
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
    return categorical_inputs + [numeric_input], all_features

