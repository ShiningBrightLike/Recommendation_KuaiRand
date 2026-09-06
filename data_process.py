"""Preprocess the KuaiRand raw CSVs into train / validation / test parquet files.

Layout decisions:
    - The validation split is carved out of the training log by time
      (date >= config.VAL_START_DATE), so early stopping never touches the
      held-out test period (4/22-5/08).
    - Categorical encoders and the numeric scaler are fit on the train split
      only, then applied to val/test (no leakage).
    - The total categorical vocabulary size is derived from the fitted
      encoders and written to `pipeline_meta.json` for model building.

Usage:
    python data_process.py
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config as C

UNK = "UNK"


def _read_and_merge(log_file):
    """Read a behavior log and left-join user/video features onto it."""
    log = pd.read_csv(log_file)
    user_features = pd.read_csv(C.USER_FEATURES_FILE)
    video_basic = pd.read_csv(C.VIDEO_FEATURES_BASIC_FILE)
    video_statistics = pd.read_csv(C.VIDEO_FEATURES_STATISTIC_FILE)

    merged = log.merge(user_features, on="user_id", how="left")
    merged = merged.merge(video_basic, on="video_id", how="left")
    merged = merged.merge(video_statistics, on="video_id", how="left")
    return merged


def _convert_date_to_weekday(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").dt.dayofweek
    return df


def _fill_missing(df):
    df = df.copy()
    df.fillna(-1, inplace=True)
    return df


def _fit_categorical(train_df):
    """Fit one LabelEncoder per categorical column on train data.

    Each column is assigned a contiguous block of ids (feature_offsets[col] ..),
    and one extra UNK class is reserved for values unseen during fitting.
    Mutates `train_df` in place, encoding its categorical columns.
    """
    label_encoders = {}
    feature_offsets = {}
    offset = 0
    for col in C.CATEGORICAL_COLS:
        unique_values = np.append(train_df[col].astype(str).unique(), UNK)
        le = LabelEncoder().fit(unique_values)
        train_df[col] = le.transform(train_df[col].astype(str)) + offset
        label_encoders[col] = le
        feature_offsets[col] = offset
        offset += len(le.classes_)
    return label_encoders, feature_offsets, offset


def _transform_categorical(df, label_encoders, feature_offsets):
    """Apply train-fitted encoders to another split, mapping unseen values to UNK."""
    for col in C.CATEGORICAL_COLS:
        values = df[col].astype(str)
        values = values.where(values.isin(label_encoders[col].classes_), UNK)
        df[col] = label_encoders[col].transform(values) + feature_offsets[col]


def _positive_ratios(df):
    return {col: float((df[col] == 1).mean()) for col in C.LABEL_COLS}


def _save_split(name, df):
    """Persist feature/label parquet files for a split and return (rows, ratios)."""
    x_file, y_file = (C.PROCESSED_DIR / f for f in C.SPLIT_FILES[name])
    x = df[C.CATEGORICAL_COLS + C.NUMERIC_COLS]
    y = df[C.LABEL_COLS]
    x.to_parquet(x_file, index=False)
    y.to_parquet(y_file, index=False)
    return len(df), _positive_ratios(df)


def main():
    started = time.time()
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and merging raw data...")
    train_all = _read_and_merge(C.TRAIN_LOG_FILE)
    test_df = _read_and_merge(C.TEST_LOG_FILE)

    # Time-based split inside the training log: everything from VAL_START_DATE
    # (inclusive) onwards becomes validation.
    date_str = train_all["date"].astype(str)
    val_mask = date_str >= C.VAL_START_DATE
    train_df = train_all.loc[~val_mask].copy()
    val_df = train_all.loc[val_mask].copy()
    del train_all

    train_df = _fill_missing(_convert_date_to_weekday(train_df))
    val_df = _fill_missing(_convert_date_to_weekday(val_df))
    test_df = _fill_missing(_convert_date_to_weekday(test_df))

    print(
        f"Row counts -> train: {len(train_df):,}, "
        f"val: {len(val_df):,}, test: {len(test_df):,}"
    )

    # Fit encoders/scaler on train only, then apply to all splits.
    label_encoders, feature_offsets, cat_vocab_size = _fit_categorical(train_df)
    _transform_categorical(val_df, label_encoders, feature_offsets)
    _transform_categorical(test_df, label_encoders, feature_offsets)

    scaler = StandardScaler().fit(train_df[C.NUMERIC_COLS])
    for df in (train_df, val_df, test_df):
        df[C.NUMERIC_COLS] = scaler.transform(df[C.NUMERIC_COLS])

    print("Saving processed splits...")
    rows, ratios = {}, {}
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        n_rows, n_ratios = _save_split(name, df)
        rows[name] = n_rows
        ratios[name] = n_ratios
        print(f"  {name:5s}: {n_rows:,} rows | positive ratios: {n_ratios}")

    # Persist reusable artifacts and pipeline metadata.
    joblib.dump(label_encoders, C.PROCESSED_DIR / "label_encoders.pkl")
    joblib.dump(scaler, C.PROCESSED_DIR / "scaler.pkl")
    joblib.dump(feature_offsets, C.PROCESSED_DIR / "feature_offsets.pkl")

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "val_start_date": C.VAL_START_DATE,
        "cat_vocab_size": cat_vocab_size,
        "categorical_cols": C.CATEGORICAL_COLS,
        "numeric_cols": C.NUMERIC_COLS,
        "label_cols": C.LABEL_COLS,
        "rows": rows,
        "positive_ratios": ratios,
    }
    with open(C.PROCESSED_DIR / "pipeline_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"Done in {time.time() - started:.1f}s. "
        f"cat_vocab_size = {cat_vocab_size}"
    )


if __name__ == "__main__":
    main()
