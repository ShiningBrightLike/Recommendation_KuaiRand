"""Feature encoders — the swappable feature-representation axis."""

from models.encoders.base import FeatureEncoder
from models.encoders.mlp import MLPEncoder

ENCODERS = {
    "mlp": MLPEncoder,
}

#: The encoder every model uses unless the caller picks another one.
DEFAULT_ENCODER = "mlp"

# Keys of `get_config()` that describe the contract rather than a hyper-parameter.
_CONFIG_PLUMBING = {"name", "trainable", "dtype", "field_dims", "output_dim"}


def available_encoders():
    """Registered encoder names, sorted."""
    return sorted(ENCODERS)


def create_encoder(name, field_dims, **overrides):
    """Build the feature encoder registered under `name`."""
    try:
        encoder_class = ENCODERS[name]
    except KeyError:
        raise ValueError(
            f"unknown encoder {name!r}; available: {available_encoders()}"
        ) from None
    return encoder_class(field_dims=field_dims, **overrides)


def encoder_hyperparams(encoder):
    """Effective hyper-parameters of a built encoder, for run metadata."""
    return {
        key: value
        for key, value in encoder.get_config().items()
        if key not in _CONFIG_PLUMBING
    }


__all__ = [
    "DEFAULT_ENCODER",
    "ENCODERS",
    "FeatureEncoder",
    "MLPEncoder",
    "available_encoders",
    "create_encoder",
    "encoder_hyperparams",
]
