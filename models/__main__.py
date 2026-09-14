"""Structural self-check: build every model with random inputs (no data needed).

Run from the repo root:
    python -m models
"""

import numpy as np

import config as C
from models import available_models, create_model

VOCAB_SIZE = 256
SAMPLES = 4

# Keyword arguments each registered model needs beyond the shared schema.
# Every registry entry must appear here; `build_all_models` fails loudly
# otherwise, so a newly registered model cannot be silently skipped.
PER_MODEL_KWARGS = {
    "mmoe": {"num_tasks": len(C.LABEL_COLS)},
    "shared_bottom": {"num_tasks": len(C.LABEL_COLS)},
    "single_task": {},
    "logistic": {"num_tasks": len(C.LABEL_COLS)},
}


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
    """Every registered model, built with its documented defaults."""
    missing = [name for name in available_models() if name not in PER_MODEL_KWARGS]
    if missing:
        raise RuntimeError(f"no smoke-test kwargs registered for: {missing}")
    return {
        name: create_model(
            name,
            categorical_cols=C.CATEGORICAL_COLS,
            numeric_cols=C.NUMERIC_COLS,
            cat_vocab_size=VOCAB_SIZE,
            **PER_MODEL_KWARGS[name],
        )
        for name in available_models()
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
