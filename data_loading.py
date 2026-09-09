"""Shared helpers for loading processed splits and the feature schema.

The feature schema (categorical/numeric column lists) is recorded by
`data_process.py` in `pipeline_meta.json`. Consumers use that schema instead
of hard-coding `config.py` lists, so runs that add shadow (random noise)
features keep every downstream script in sync automatically.
"""

import json

import pandas as pd

import config as C


def read_pipeline_meta():
    """Return pipeline metadata dict, or None when data_process hasn't run."""
    meta_file = C.PROCESSED_DIR / "pipeline_meta.json"
    if not meta_file.exists():
        return None
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_feature_schema():
    """Return (categorical_cols, numeric_cols, shadow_cols) for the processed data.

    Falls back to the canonical config lists when pipeline metadata is absent.
    """
    meta = read_pipeline_meta()
    if meta is None:
        return list(C.CATEGORICAL_COLS), list(C.NUMERIC_COLS), []
    return (
        list(meta.get("categorical_cols", C.CATEGORICAL_COLS)),
        list(meta.get("numeric_cols", C.NUMERIC_COLS)),
        list(meta.get("shadow_cols", [])),
    )


def load_cat_vocab_size():
    """Vocab size comes from pipeline metadata; fall back to the encoders."""
    meta = read_pipeline_meta()
    if meta is not None and "cat_vocab_size" in meta:
        return int(meta["cat_vocab_size"])

    import joblib

    encoders = joblib.load(C.PROCESSED_DIR / "label_encoders.pkl")
    offsets = joblib.load(C.PROCESSED_DIR / "feature_offsets.pkl")
    return max(offsets[col] + len(encoders[col].classes_) for col in C.CATEGORICAL_COLS)


def load_split(name, max_rows=None, categorical_cols=None, numeric_cols=None):
    """Load a processed split as model inputs / targets.

    Returns (inputs, targets, n_rows, positive_ratios), where `inputs` is the
    model input list: one int32 array per categorical column followed by one
    float32 numeric matrix.
    """
    if name not in C.SPLIT_FILES:
        raise ValueError(f"unknown split: {name}")
    if categorical_cols is None or numeric_cols is None:
        categorical_cols, numeric_cols, _ = load_feature_schema()

    x_file, y_file = (C.PROCESSED_DIR / f for f in C.SPLIT_FILES[name])
    if not x_file.exists() or not y_file.exists():
        raise FileNotFoundError(
            f"missing processed files for split '{name}'. "
            f"Run `python data_process.py` first."
        )

    df_x = pd.read_parquet(x_file)
    df_y = pd.read_parquet(y_file)
    if max_rows is not None:
        df_x = df_x.head(max_rows)
        df_y = df_y.head(max_rows)

    missing_cat = [c for c in categorical_cols if c not in df_x.columns]
    missing_num = [c for c in numeric_cols if c not in df_x.columns]
    if missing_cat or missing_num:
        raise ValueError(
            "processed X is out of sync with pipeline_meta.json; regenerate with "
            f"`python data_process.py`. missing categorical={missing_cat}, "
            f"missing numeric={missing_num}"
        )
    if list(df_y.columns) != C.LABEL_COLS:
        raise ValueError(
            f"processed y columns {list(df_y.columns)} != config.LABEL_COLS "
            f"{C.LABEL_COLS}; regenerate with `python data_process.py`"
        )

    x_categorical = [df_x[col].astype("int32").values for col in categorical_cols]
    x_numeric = df_x[numeric_cols].astype("float32").values
    targets = [
        df_y[col].values.reshape(-1, 1).astype("float32") for col in C.LABEL_COLS
    ]
    positive_ratios = {col: float(df_y[col].mean()) for col in C.LABEL_COLS}
    return x_categorical + [x_numeric], targets, len(df_x), positive_ratios
