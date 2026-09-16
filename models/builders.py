"""Assemble ranking models and their traceable two-axis metadata."""

from copy import deepcopy

from tensorflow.keras.models import Model

from models.encoders import (
    DEFAULT_ENCODER,
    FeatureEncoder,
    create_encoder,
    encoder_hyperparams,
)
from models.inputs import build_shared_inputs
from models.registry import (
    BASELINES,
    STRUCTURES,
    available_models,
    baseline_hyperparams,
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
    model, _ = build_registered_model(name, encoder=encoder, **kwargs)
    return model


def build_registered_model(name, encoder=None, **kwargs):
    """Build any catalog model and return complete, uniform run metadata.

    Multi-output structures use :func:`build_model`. Baselines keep their
    defining shape, but expose the same metadata interface so callers never
    need special cases merely to make a run traceable.
    """
    if name in STRUCTURES:
        return build_model(
            encoder=encoder or DEFAULT_ENCODER, structure=name, **kwargs
        )
    if name not in BASELINES:
        raise ValueError(f"unknown model {name!r}; available: {available_models()}")

    builder_kwargs = dict(kwargs)
    if encoder is not None:
        builder_kwargs["encoder"] = encoder
    model = BASELINES[name](**builder_kwargs)

    encoder_layer = next(
        (layer for layer in model.layers if isinstance(layer, FeatureEncoder)), None
    )
    encoder_meta = {
        "name": encoder,
        "output_dim": None,
        "params": 0,
        "hyperparams": {},
    }
    if encoder_layer is not None:
        encoder_meta = {
            "name": encoder or DEFAULT_ENCODER,
            "output_dim": encoder_layer.output_dim,
            "params": int(encoder_layer.count_params()),
            "hyperparams": encoder_hyperparams(encoder_layer),
        }

    return model, {
        "encoder": encoder_meta,
        "structure": {
            "name": name,
            "hyperparams": baseline_hyperparams(name, **builder_kwargs),
        },
        "total_params": int(model.count_params()),
    }


def aggregate_independent_model_axes(per_model_axes):
    """Describe a group of independently trained copies as one comparison row."""
    if not per_model_axes:
        raise ValueError("at least one independent model is required")
    first = per_model_axes[0]
    for axes in per_model_axes[1:]:
        if axes["encoder"] != first["encoder"] or axes["structure"] != first["structure"]:
            raise ValueError("independent models must use the same two-axis configuration")

    grouped = deepcopy(first)
    grouped["structure"]["hyperparams"]["num_tasks"] = len(per_model_axes)
    grouped["per_task_params"] = [axes["total_params"] for axes in per_model_axes]
    grouped["total_params"] = sum(grouped["per_task_params"])
    return grouped
