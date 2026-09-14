"""Name-based lookup for the ranking model builders.

Builder signatures differ on purpose — single-task builds one task's model,
logistic has no hidden layer — so `create_model` forwards the keyword
arguments that the selected model needs.
"""

from models.mtl.logistic import build_logistic_model
from models.mtl.mmoe import MMoE, build_mmoe_model
from models.mtl.shared_bottom import build_shared_bottom_model
from models.mtl.single_task import build_single_task_model

BUILDERS = {
    "mmoe": build_mmoe_model,
    "shared_bottom": build_shared_bottom_model,
    "single_task": build_single_task_model,
    "logistic": build_logistic_model,
}

# Custom layers that `keras.models.load_model` must be told about when a saved
# run is reloaded outside this package.
CUSTOM_LAYERS = {
    "MMoE": MMoE,
}


def available_models():
    """Registered model names, sorted."""
    return sorted(BUILDERS)


def custom_objects():
    """Objects to pass to `load_model` when reloading a saved run."""
    return dict(CUSTOM_LAYERS)


def create_model(name, **kwargs):
    """Build the model registered under `name`."""
    try:
        builder = BUILDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; available: {available_models()}"
        ) from None
    return builder(**kwargs)
