"""Assemble a ranking model from the two axes: feature encoder + structure."""

from tensorflow.keras.models import Model

from models.encoders import DEFAULT_ENCODER, create_encoder, encoder_hyperparams
from models.inputs import build_shared_inputs
from models.registry import (
    BASELINES,
    STRUCTURES,
    available_models,
    structure_builder,
    structure_hyperparams,
)


def build_model(
    encoder=DEFAULT_ENCODER,
    structure="mmoe",
    *,
    categorical_cols,
    numeric_cols,
    cat_vocab_size,
    embed_dim=8,
    num_tasks=2,
    encoder_overrides=None,
    **structure_kwargs,
):
    """Build a ranking model from the feature-encoder and structure axes.

    Returns `(model, axes)`: the model, plus the run-metadata block describing
    both axes — names, effective hyper-parameters and the encoder's parameter
    count. `structure_kwargs` are passed to the structure (for example
    `num_experts`, `units`, `tower_units` for MMoE).
    """
    inputs, all_features, layout = build_shared_inputs(
        categorical_cols, numeric_cols, cat_vocab_size, embed_dim
    )
    encoder_layer = create_encoder(
        encoder, layout.field_dims, **(encoder_overrides or {})
    )
    encoded = encoder_layer(all_features)

    structure_args = {"num_tasks": num_tasks, **structure_kwargs}
    outputs = structure_builder(structure)(encoded, **structure_args)
    model = Model(inputs=inputs, outputs=outputs)

    axes = {
        "encoder": {
            "name": encoder,
            "output_dim": encoder_layer.output_dim,
            "params": int(encoder_layer.count_params()),
            "hyperparams": encoder_hyperparams(encoder_layer),
        },
        "structure": {
            "name": structure,
            "hyperparams": structure_hyperparams(structure, **structure_args),
        },
        "total_params": int(model.count_params()),
    }
    return model, axes


def create_model(name, encoder=None, **kwargs):
    """Build a model by catalog name — a structure or a baseline.

    `encoder=None` keeps the model's own default. Baselines that have no
    feature encoder reject one rather than ignoring it.
    """
    if name in BASELINES:
        if encoder is not None:
            kwargs["encoder"] = encoder
        return BASELINES[name](**kwargs)
    if name in STRUCTURES:
        model, _ = build_model(
            encoder=encoder or DEFAULT_ENCODER, structure=name, **kwargs
        )
        return model
    raise ValueError(f"unknown model {name!r}; available: {available_models()}")
