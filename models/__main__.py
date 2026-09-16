"""Structural self-check: build every model with random inputs (no data needed).

Run from the repo root:
    python -m models
"""

import numpy as np

import config as C
from models import (
    available_encoders,
    available_structures,
    build_model,
    build_registered_model,
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
    """Every registered model and encoder x structure combination."""
    models = {
        f"{structure}+{encoder}": build_model(
            encoder=encoder,
            structure=structure,
            categorical_cols=C.CATEGORICAL_COLS,
            numeric_cols=C.NUMERIC_COLS,
            cat_vocab_size=VOCAB_SIZE,
            num_tasks=len(C.LABEL_COLS),
        )
        for structure in available_structures()
        for encoder in available_encoders()
    }
    models.update(
        {
            f"single_task+{encoder}": build_registered_model(
                "single_task",
                encoder=encoder,
                categorical_cols=C.CATEGORICAL_COLS,
                numeric_cols=C.NUMERIC_COLS,
                cat_vocab_size=VOCAB_SIZE,
            )
            for encoder in available_encoders()
        }
    )
    models["logistic"] = build_registered_model(
        "logistic",
        categorical_cols=C.CATEGORICAL_COLS,
        numeric_cols=C.NUMERIC_COLS,
        cat_vocab_size=VOCAB_SIZE,
        num_tasks=len(C.LABEL_COLS),
    )
    return models


def main():
    inputs = make_inputs()
    for name, (model, axes) in build_all_models().items():
        predictions = model.predict(inputs, verbose=0)
        if not isinstance(predictions, list):
            predictions = [predictions]
        shapes = ", ".join(str(tuple(pred.shape)) for pred in predictions)
        print(
            f"{name}: encoder_params={axes['encoder']['params']:,} "
            f"total_params={axes['total_params']:,} "
            f"outputs={len(predictions)} {shapes}"
        )


if __name__ == "__main__":
    main()
