"""Train and evaluate the MMoE multi-task model.

Protocol:
    - Train on the train split; early stopping watches the validation split.
    - The held-out test split is evaluated exactly once, after training.
    - Every run gets its own directory under `KuaiRand-Pure/saved/runs/`
      containing model.keras, metrics.json, training.log and curves.png.

Usage:
    python main.py                          # full training run
    python main.py --smoke                  # tiny run to check the pipeline
    python main.py --seed 42 --epochs 20 --tag reproduce-baseline
"""

import argparse
import json
import logging
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")  # headless by default; `--show` still displays afterwards
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.metrics import AUC

import config as C
from MMoE_model import build_mmoe_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=C.EARLY_STOP_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=C.LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    parser.add_argument("--tag", default="mmoe", help="prefix for the run directory")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="limit each split to N rows (quick debugging)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="shortcut: max_rows=2048, 1 epoch, no plot window",
    )
    parser.add_argument("--show", action="store_true", help="display the plots at the end")
    return parser.parse_args()


def set_seed(seed):
    """Make sampling reproducible where TensorFlow allows it."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def setup_logging(run_dir):
    logger = logging.getLogger("kuairand")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_split(name, max_rows=None):
    """Load a processed split as model inputs / targets."""
    if name not in C.SPLIT_FILES:
        raise ValueError(f"unknown split: {name}")
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

    missing_cat = [c for c in C.CATEGORICAL_COLS if c not in df_x.columns]
    missing_num = [c for c in C.NUMERIC_COLS if c not in df_x.columns]
    if missing_cat or missing_num:
        raise ValueError(
            "processed X is out of sync with config.py; regenerate with "
            f"`python data_process.py`. missing categorical={missing_cat}, "
            f"missing numeric={missing_num}"
        )
    if list(df_y.columns) != C.LABEL_COLS:
        raise ValueError(
            f"processed y columns {list(df_y.columns)} != config.LABEL_COLS "
            f"{C.LABEL_COLS}; regenerate with `python data_process.py`"
        )

    x_categorical = [df_x[col].astype("int32").values for col in C.CATEGORICAL_COLS]
    x_numeric = df_x[C.NUMERIC_COLS].astype("float32").values
    targets = [
        df_y[col].values.reshape(-1, 1).astype("float32") for col in C.LABEL_COLS
    ]
    positive_ratios = {col: float(df_y[col].mean()) for col in C.LABEL_COLS}
    return x_categorical + [x_numeric], targets, len(df_x), positive_ratios


def load_cat_vocab_size():
    """Vocab size comes from pipeline metadata; fall back to the encoders."""
    meta_file = C.PROCESSED_DIR / "pipeline_meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        return int(meta["cat_vocab_size"])

    import joblib

    encoders = joblib.load(C.PROCESSED_DIR / "label_encoders.pkl")
    offsets = joblib.load(C.PROCESSED_DIR / "feature_offsets.pkl")
    return max(offsets[col] + len(encoders[col].classes_) for col in C.CATEGORICAL_COLS)


class TrainingLogger(Callback):
    """Log per-epoch train/val loss and AUC per task."""

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for i, task in enumerate(C.LABEL_COLS):
            out = f"output_{i + 1}"
            self.logger.info(
                f"Epoch {epoch + 1}: {task} "
                f"train_loss={logs.get(f'{out}_loss', float('nan')):.4f} "
                f"val_loss={logs.get(f'val_{out}_loss', float('nan')):.4f} "
                f"train_auc={logs.get(f'{out}_auc', float('nan')):.4f} "
                f"val_auc={logs.get(f'val_{out}_auc', float('nan')):.4f}"
            )


def plot_history(history, save_path, show=False):
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(16, 6))

    for i in range(len(C.LABEL_COLS)):
        out = f"output_{i + 1}"
        ax_loss.plot(history[f"{out}_loss"], label=f"{C.LABEL_COLS[i]} train", alpha=0.35)
        ax_loss.plot(history[f"val_{out}_loss"], label=f"{C.LABEL_COLS[i]} val")
        ax_auc.plot(history[f"{out}_auc"], label=f"{C.LABEL_COLS[i]} train", alpha=0.35)
        ax_auc.plot(history[f"val_{out}_auc"], label=f"{C.LABEL_COLS[i]} val")
    ax_loss.plot(history["loss"], "k--", label="total train")
    ax_loss.plot(history["val_loss"], "k-", label="total val")

    ax_loss.set_title("Training & Validation Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize="small")
    ax_auc.set_title("Training & Validation AUC")
    ax_auc.set_xlabel("Epoch")
    ax_auc.set_ylabel("AUC")
    ax_auc.legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    if args.smoke:
        args.max_rows = args.max_rows or 2048
        args.epochs = 1
        args.tag = f"{args.tag}_smoke"

    set_seed(args.seed)
    run_name = f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = C.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir)

    logger.info("Loading processed data...")
    x_train, y_train, n_train, ratios_train = load_split("train", args.max_rows)
    x_val, y_val, n_val, ratios_val = load_split("val", args.max_rows)
    x_test, y_test, n_test, ratios_test = load_split("test", args.max_rows)

    for split, ratios in (
        ("train", ratios_train),
        ("val", ratios_val),
        ("test", ratios_test),
    ):
        logger.info(f"positive ratios ({split}): {ratios}")

    cat_vocab_size = load_cat_vocab_size()
    logger.info(f"Building MMoE model (cat_vocab_size={cat_vocab_size})...")
    model = build_mmoe_model(
        categorical_cols=C.CATEGORICAL_COLS,
        numeric_cols=C.NUMERIC_COLS,
        cat_vocab_size=cat_vocab_size,
        embed_dim=C.EMBED_DIM,
        num_experts=C.NUM_EXPERTS,
        num_tasks=len(C.LABEL_COLS),
        units=C.EXPERT_UNITS,
        tower_units=C.TOWER_UNITS,
    )

    output_names = [f"output_{i + 1}" for i in range(len(C.LABEL_COLS))]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss={name: "binary_crossentropy" for name in output_names},
        loss_weights={
            name: C.LOSS_WEIGHTS[task] for name, task in zip(output_names, C.LABEL_COLS)
        },
        metrics={name: [AUC(name="auc")] for name in output_names},
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
        verbose=1,
    )
    logger.info(
        f"Training: train={n_train:,}, val={n_val:,}, test={n_test:,}, "
        f"epochs={args.epochs}, batch_size={args.batch_size}, seed={args.seed}"
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=0,
        callbacks=[early_stopping, TrainingLogger(logger)],
    )

    best_epoch = getattr(early_stopping, "best_epoch", None)
    best_epoch = None if best_epoch is None else int(best_epoch) + 1
    logger.info(
        f"Training finished. best_epoch={best_epoch}, "
        f"stopped_epoch={getattr(early_stopping, 'stopped_epoch', None)}"
    )

    model_path = run_dir / "model.keras"
    model.save(model_path)
    logger.info(f"Saved model to {model_path}")

    logger.info("Evaluating on the held-out test split...")
    test_metrics = model.evaluate(
        x_test, y_test, batch_size=args.batch_size, verbose=0, return_dict=True
    )
    test_metrics = {k: float(v) for k, v in test_metrics.items()}
    for i, task in enumerate(C.LABEL_COLS):
        logger.info(f"test {task}: auc={test_metrics.get(f'output_{i + 1}_auc'):.4f}")

    history_dict = {k: list(v) for k, v in history.history.items()}
    summary = {
        "run_name": run_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "cat_vocab_size": cat_vocab_size,
        "row_counts": {"train": n_train, "val": n_val, "test": n_test},
        "positive_ratios": {
            "train": ratios_train,
            "val": ratios_val,
            "test": ratios_test,
        },
        "best_epoch": best_epoch,
        "stopped_epoch": getattr(early_stopping, "stopped_epoch", None),
        "test_metrics": test_metrics,
        "history": history_dict,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Saved metrics to {metrics_path}")

    curves_path = run_dir / "curves.png"
    plot_history(history_dict, curves_path, show=args.show)
    logger.info(f"Saved training curves to {curves_path}")
    logger.info(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
