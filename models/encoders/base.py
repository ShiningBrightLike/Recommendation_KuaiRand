"""Base class for feature encoders."""

from tensorflow.keras.layers import Layer


class FeatureEncoder(Layer):
    """Maps the concatenated feature vector to a dense representation.

    Contract: `call` takes the encoded feature vector
    (`[batch, sum(field_dims)]`) and returns `[batch, output_dim]`. Encoders
    keep their natural output width — the multi-task structure consumes
    whatever the encoder produces.
    """

    def __init__(self, output_dim, field_dims, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = int(output_dim)
        self.field_dims = tuple(field_dims)

    @property
    def input_dim(self):
        """Width of the concatenated feature vector this encoder consumes."""
        return sum(self.field_dims)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "field_dims": list(self.field_dims),
            }
        )
        return config

