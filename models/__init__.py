"""Ranking models for the multi-task KuaiRand feedback prediction project.

A model is assembled from two independent axes (ADR-0005):
    - a **feature encoder** that turns the concatenated feature vector into a
      dense representation (`models.encoders`), and
    - a **multi-task structure** that decides how tasks share and differ
      (`models.mtl`).

Input convention (shared by every model here):
    - one int32 Input per categorical feature (Embedding indices)
    - one float32 Input with all numeric features
Output convention:
    - outputs named output_1 ... output_N, in the same order as the task list
      passed by the caller (the pipeline uses config.LABEL_COLS)

`cat_vocab_size` must be strictly larger than the largest embedding index used.
The preprocessing pipeline computes it dynamically and records it in
`data_processed/pipeline_meta.json`; pass that value in when building a model.
"""

from models.builders import (
    aggregate_independent_model_axes,
    build_model,
    build_registered_model,
    create_model,
)
from models.encoders import (
    DEFAULT_ENCODER,
    FeatureEncoder,
    MLPEncoder,
    available_encoders,
    create_encoder,
)
from models.inputs import FeatureLayout, build_shared_inputs
from models.mtl.logistic import build_logistic_model
from models.mtl.mmoe import MMoE
from models.mtl.ple_cgc import PLECGC
from models.mtl.single_task import build_single_task_model
from models.registry import (
    DEFAULT_STRUCTURE,
    available_models,
    available_structures,
    baseline_hyperparams,
    custom_objects,
    structure_hyperparams,
)

__all__ = [
    "DEFAULT_ENCODER",
    "DEFAULT_STRUCTURE",
    "FeatureEncoder",
    "FeatureLayout",
    "MLPEncoder",
    "MMoE",
    "PLECGC",
    "available_encoders",
    "available_models",
    "available_structures",
    "aggregate_independent_model_axes",
    "build_logistic_model",
    "build_model",
    "build_registered_model",
    "build_shared_inputs",
    "build_single_task_model",
    "create_encoder",
    "create_model",
    "custom_objects",
    "baseline_hyperparams",
    "structure_hyperparams",
]
