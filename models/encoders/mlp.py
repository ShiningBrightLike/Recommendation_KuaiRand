"""MLP feature encoder: dense hidden layers over the feature vector."""

from tensorflow.keras.layers import Dense
from tensorflow.keras.saving import register_keras_serializable

from models.encoders.base import FeatureEncoder

DEFAULTS = {"hidden": (64,), "activation": "relu"}


@register_keras_serializable()
class MLPEncoder(FeatureEncoder):
    """The default encoder: a single dense hidden layer unless told otherwise."""

    def __init__(
        self,
        field_dims,
        hidden=DEFAULTS["hidden"],
        activation=DEFAULTS["activation"],
        output_dim=None,
        **kwargs,
    ):
        hidden = tuple(hidden)
        if not hidden:
            raise ValueError("the mlp encoder needs at least one hidden layer")
        super().__init__(
            output_dim=hidden[-1] if output_dim is None else output_dim,
            field_dims=field_dims,
            **kwargs,
        )
        self.hidden = hidden
        self.activation = activation
        self.hidden_layers = [Dense(width, activation=activation) for width in hidden]

    def call(self, features):
        encoded = features
        for layer in self.hidden_layers:
            encoded = layer(encoded)
        return encoded

    def build(self, input_shape):
        shape = input_shape
        for layer in self.hidden_layers:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden": list(self.hidden), "activation": self.activation})
        return config
