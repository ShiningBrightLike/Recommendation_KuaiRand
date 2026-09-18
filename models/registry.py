"""Name maps for the ranking models: structures, baselines and custom layers.

The two modelling axes are documented in ADR-0005. Feature encoders have their
own map in `models.encoders`; this module covers everything downstream of them.
"""

import inspect

from models.encoders import DCNEncoder, MLPEncoder, SENetEncoder
from models.encoders.dcn import LowRankCrossLayer
from models.mtl.logistic import build_logistic_model
from models.mtl.mmoe import MMoE, build_mmoe_structure
from models.mtl.ple_cgc import PLECGC, build_ple_cgc_structure
from models.mtl.shared_bottom import build_shared_bottom_structure
from models.mtl.single_task import build_single_task_model

#: The structure every model uses unless the caller picks another one.
DEFAULT_STRUCTURE = "mmoe"

# Structures consume a feature encoder and produce one output per task.
STRUCTURES = {
    "mmoe": build_mmoe_structure,
    "ple_cgc": build_ple_cgc_structure,
    "shared_bottom": build_shared_bottom_structure,
}

# Baselines keep their own shape: single-task builds one task's model, and
# logistic has no hidden representation at all.
BASELINES = {
    "single_task": build_single_task_model,
    "logistic": build_logistic_model,
}

# Custom layers `keras.models.load_model` must be told about when a saved run
# is reloaded outside this package.
CUSTOM_LAYERS = {
    "DCNEncoder": DCNEncoder,
    "LowRankCrossLayer": LowRankCrossLayer,
    "MMoE": MMoE,
    "PLECGC": PLECGC,
    "MLPEncoder": MLPEncoder,
    "SENetEncoder": SENetEncoder,
}


def available_models():
    """Every model name the comparison scripts may ask for, sorted."""
    return sorted(set(STRUCTURES) | set(BASELINES))


def available_structures():
    """Structure names that accept a feature encoder, sorted."""
    return sorted(STRUCTURES)


def structure_builder(name):
    """The structure builder registered under `name`."""
    try:
        return STRUCTURES[name]
    except KeyError:
        raise ValueError(
            f"unknown structure {name!r}; available: {available_structures()}"
        ) from None


def structure_hyperparams(name, **kwargs):
    """Effective keyword hyper-parameters the named structure will use."""
    builder = structure_builder(name)
    bound = inspect.signature(builder).bind_partial(**kwargs)
    bound.apply_defaults()
    return {
        key: _jsonable(value)
        for key, value in bound.arguments.items()
        if key != "features"
    }


def baseline_hyperparams(name, **kwargs):
    """Effective non-input hyper-parameters for a baseline model."""
    try:
        builder = BASELINES[name]
    except KeyError:
        raise ValueError(
            f"unknown baseline {name!r}; available: {sorted(BASELINES)}"
        ) from None
    bound = inspect.signature(builder).bind_partial(**kwargs)
    bound.apply_defaults()
    input_args = {
        "categorical_cols",
        "numeric_cols",
        "cat_vocab_size",
        "encoder",
        "encoder_overrides",
    }
    return {
        key: _jsonable(value)
        for key, value in bound.arguments.items()
        if key not in input_args
    }


def custom_objects():
    """Objects to pass to `load_model` when reloading a saved run."""
    return dict(CUSTOM_LAYERS)


def _jsonable(value):
    """Tuples become lists so the run metadata stays JSON-serialisable."""
    return list(value) if isinstance(value, tuple) else value
