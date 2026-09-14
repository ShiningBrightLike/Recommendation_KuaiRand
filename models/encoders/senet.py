"""SENet feature encoder: squeeze-excite reweighting per feature field."""

import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

from models.encoders.base import FeatureEncoder

DEFAULTS = {"reduction": 2}


@register_keras_serializable()
class SENetEncoder(FeatureEncoder):
    """Reweights whole feature fields by a gate learned from all fields.

    Squeeze turns each field into one number (its mean), excite runs a small
    bottleneck over those numbers and squashes them through a sigmoid, and the
    result scales each field as a unit. The encoder keeps the input width — it
    only changes how much each field contributes — which is what makes it
    different from a plain dense layer.
    """

    def __init__(
        self,
        field_dims,
        reduction=DEFAULTS["reduction"],
        output_dim=None,
        **kwargs,
    ):
        if int(reduction) < 1:
            raise ValueError("the senet encoder needs reduction >= 1")
        expected_dim = sum(field_dims)
        if output_dim is not None and int(output_dim) != expected_dim:
            raise ValueError(
                f"the senet encoder keeps the input width {expected_dim}; "
                f"got output_dim={output_dim}"
            )
        super().__init__(output_dim=expected_dim, field_dims=field_dims, **kwargs)
        self.reduction = int(reduction)

    @property
    def num_fields(self):
        """How many fields the gate works over."""
        return len(self.field_dims)

    @property
    def bottleneck_units(self):
        """Width of the excitation bottleneck."""
        return max(1, self.num_fields // self.reduction)

    def build(self, input_shape):
        width = int(input_shape[-1])
        if width != sum(self.field_dims):
            raise ValueError(
                f"the senet encoder expects {sum(self.field_dims)} input dims "
                f"({self.num_fields} fields) but got {width}"
            )
        fields = self.num_fields
        hidden = self.bottleneck_units
        self.w1 = self.add_weight(
            shape=(fields, hidden), initializer="glorot_uniform", name="w1"
        )
        self.b1 = self.add_weight(shape=(hidden,), initializer="zeros", name="b1")
        self.w2 = self.add_weight(
            shape=(hidden, fields), initializer="glorot_uniform", name="w2"
        )
        self.b2 = self.add_weight(shape=(fields,), initializer="zeros", name="b2")
        super().build(input_shape)

    def call(self, features):
        field_slices = tf.split(features, self.field_dims, axis=-1)
        squeezed = tf.stack(
            [tf.reduce_mean(field, axis=-1) for field in field_slices], axis=-1
        )
        excited = tf.sigmoid(
            tf.matmul(tf.nn.relu(tf.matmul(squeezed, self.w1) + self.b1), self.w2)
            + self.b2
        )
        reweighted = [
            field * excited[:, index : index + 1]
            for index, field in enumerate(field_slices)
        ]
        return tf.concat(reweighted, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction})
        return config

