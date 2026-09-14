"""Ranking models for the multi-task KuaiRand feedback prediction project.

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

from models.mtl.logistic import build_logistic_model
from models.mtl.mmoe import MMoE, build_mmoe_model
from models.mtl.shared_bottom import build_shared_bottom_model
from models.mtl.single_task import build_single_task_model
from models.registry import available_models, create_model

__all__ = [
    "MMoE",
    "available_models",
    "build_logistic_model",
    "build_mmoe_model",
    "build_shared_bottom_model",
    "build_single_task_model",
    "create_model",
]
