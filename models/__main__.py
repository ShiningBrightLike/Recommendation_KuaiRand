"""Structural self-check: build every model with random inputs (no data needed).

Run from the repo root:
    python -m models
"""

import numpy as np

import config as C
from models import (
    build_logistic_model,
    build_mmoe_model,
    build_shared_bottom_model,
    build_single_task_model,
)

VOCAB_SIZE = 256
SAMPLES = 4


def make_inputs(n_samples=SAMPLES):
    """Random inputs matching the shared input convention."""
    rng = np.random.default_rng(0)
    categorical = [
        rng.integers(0, VOCAB_SIZE, size=n_samples).astype("int32")
        for _ in C.CATEGORICAL_COLS
    ]
    numeric = rng.normal(size=(n_samples, len(C.NUMERIC_COLS))).astype("float32")
    return categorical + [numeric]


def build_all_models():
    """Every builder the package exports, keyed by model name."""
    num_tasks = len(C.LABEL_COLS)
    return {
        "mmoe": build_mmoe_model(
            C.CATEGORICAL_COLS, C.NUMERIC_COLS, VOCAB_SIZE, num_tasks=num_tasks
        ),
        "shared_bottom": build_shared_bottom_model(
            C.CATEGORICAL_COLS, C.NUMERIC_COLS, VOCAB_SIZE, num_tasks=num_tasks
        ),
        "single_task": build_single_task_model(
            C.CATEGORICAL_COLS, C.NUMERIC_COLS, VOCAB_SIZE
        ),
        "logistic": build_logistic_model(
            C.CATEGORICAL_COLS, C.NUMERIC_COLS, VOCAB_SIZE, num_tasks=num_tasks
        ),
    }


def main():
    inputs = make_inputs()
    for name, model in build_all_models().items():
        predictions = model.predict(inputs, verbose=0)
        if not isinstance(predictions, list):
            predictions = [predictions]
        shapes = ", ".join(str(tuple(pred.shape)) for pred in predictions)
        print(f"{name}: params={model.count_params():,} outputs={len(predictions)} {shapes}")


if __name__ == "__main__":
    main()
