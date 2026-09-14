"""DCN-v2 feature encoder: low-rank crosses in parallel with a deep branch."""

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Layer
from tensorflow.keras.saving import register_keras_serializable

from models.encoders.base import FeatureEncoder

DEFAULTS = {"num_cross_layers": 2, "rank": 64, "deep_units": 64}


@register_keras_serializable()
class LowRankCrossLayer(Layer):
    """One DCN-v2 cross layer: `x' = x0 * (x V U^T + b) + x`.

    `U` and `V` are `(dim, rank)`, so the full `(dim, dim)` cross matrix is
    never materialised — that is what keeps the parameters and the compute
    proportionate to the rank instead of the input width.
    """

    def __init__(self, rank, **kwargs):
        super().__init__(**kwargs)
        self.rank = int(rank)

    def build(self, input_shape):
        dim = int(input_shape[0][-1])
        self.v = self.add_weight(
            shape=(dim, self.rank), initializer="glorot_uniform", name="v"
        )
        self.u = self.add_weight(
            shape=(dim, self.rank), initializer="glorot_uniform", name="u"
        )
        self.bias = self.add_weight(shape=(dim,), initializer="zeros", name="bias")
        super().build(input_shape)

    def call(self, inputs):
        x0, x = inputs
        projected = tf.matmul(tf.matmul(x, self.v), self.u, transpose_b=True) + self.bias
        return x0 * projected + x

    def get_config(self):
        config = super().get_config()
        config.update({"rank": self.rank})
        return config


@register_keras_serializable()
class DCNEncoder(FeatureEncoder):
    """DCN-v2: low-rank cross layers and a deep branch, concatenated.

    The output keeps its natural width: the cross network passes the input
    width through unchanged, so the encoder emits
    `input_width + deep_units`.
    """

    def __init__(
        self,
        field_dims,
        num_cross_layers=DEFAULTS["num_cross_layers"],
        rank=DEFAULTS["rank"],
        deep_units=DEFAULTS["deep_units"],
        output_dim=None,
        **kwargs,
    ):
        if int(num_cross_layers) < 1:
            raise ValueError("the dcn encoder needs at least one cross layer")
        expected_dim = sum(field_dims) + int(deep_units)
        if output_dim is not None and int(output_dim) != expected_dim:
            raise ValueError(
                f"the dcn encoder emits cross width + deep width = {expected_dim}; "
                f"got output_dim={output_dim}"
            )
        super().__init__(output_dim=expected_dim, field_dims=field_dims, **kwargs)
        self.num_cross_layers = int(num_cross_layers)
        self.rank = int(rank)
        self.deep_units = int(deep_units)
        self.cross_layers = [
            LowRankCrossLayer(self.rank) for _ in range(self.num_cross_layers)
        ]
        self.deep_branch = Dense(self.deep_units, activation="relu")
        self.concat = Concatenate()

    def call(self, features):
        crossed = features
        for layer in self.cross_layers:
            crossed = layer([features, crossed])
        deep = self.deep_branch(features)
        return self.concat([crossed, deep])

    def build(self, input_shape):
        for layer in self.cross_layers:
            layer.build([input_shape, input_shape])
        self.deep_branch.build(input_shape)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_cross_layers": self.num_cross_layers,
                "rank": self.rank,
                "deep_units": self.deep_units,
            }
        )
        return config

