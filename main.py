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
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.metrics import AUC

import config as C
from data_loading import (
    filter_features,
    load_cat_vocab_size,
    load_feature_schema,
    load_split,
    parse_name_list,
    video_statistic_cols,
)
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
    parser.add_argument(
        "--monitor",
        default="val_auc_mean",
        choices=["val_auc_mean", "val_loss"]
        + [f"val_output_{i}_auc" for i in range(1, len(C.LABEL_COLS) + 1)],
        help="metric watched by early stopping (default: mean val AUC over tasks)",
    )
    parser.add_argument(
        "--drop-features",
        default=None,
        help="comma-separated feature names to exclude from model inputs",
    )
    parser.add_argument(
        "--drop-stat-features",
        action="store_true",
        help="exclude all post-exposure video statistic features (leakage audit)",
    )
    return parser.parse_args()


def set_seed(seed):
    """Make sampling reproducible where TensorFlow allows it."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def mean_val_auc(logs, num_tasks):
    """Mean of the available per-task validation AUCs (NaN values skipped)."""
    values = []
    for i in range(1, num_tasks + 1):
        value = logs.get(f"val_output_{i}_auc")
        if value is None:
            continue
        value = float(value)
        if value == value:  # not NaN
            values.append(value)
    return sum(values) / len(values) if values else None


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
        mean_auc = logs.get("val_auc_mean")
        if mean_auc is not None:
            self.logger.info(f"Epoch {epoch + 1}: val_auc_mean={mean_auc:.4f}")


class MeanValAUC(Callback):
    """Expose the per-task mean validation AUC as `val_auc_mean` in logs.

    Registered before EarlyStopping so model selection can use this metric
    instead of the click-dominated weighted `val_loss`.
    """

    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        value = mean_val_auc(logs, self.num_tasks)
        if value is not None:
            logs["val_auc_mean"] = value


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
    cat_cols, num_cols, shadow_cols = load_feature_schema()
    drop_names = set(parse_name_list(args.drop_features) or [])
    if args.drop_stat_features:
        drop_names |= set(video_statistic_cols())
    cat_cols, num_cols, dropped = filter_features(cat_cols, num_cols, drop_names)
    if dropped:
        logger.info(f"Dropped {len(dropped)} features from model inputs: {dropped}")
    if shadow_cols:
        logger.info(f"Schema includes shadow features: {shadow_cols}")
    x_train, y_train, n_train, ratios_train = load_split(
        "train", args.max_rows, cat_cols, num_cols
    )
    x_val, y_val, n_val, ratios_val = load_split(
        "val", args.max_rows, cat_cols, num_cols
    )
    x_test, y_test, n_test, ratios_test = load_split(
        "test", args.max_rows, cat_cols, num_cols
    )

    for split, ratios in (
        ("train", ratios_train),
        ("val", ratios_val),
        ("test", ratios_test),
    ):
        logger.info(f"positive ratios ({split}): {ratios}")

    cat_vocab_size = load_cat_vocab_size()
    logger.info(f"Building MMoE model (cat_vocab_size={cat_vocab_size})...")
    model = build_mmoe_model(
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
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
        monitor=args.monitor,
        mode="min" if args.monitor == "val_loss" else "max",
        patience=args.patience,
        restore_best_weights=True,
        verbose=1,
    )
    mean_val_auc_callback = MeanValAUC(num_tasks=len(C.LABEL_COLS))
    logger.info(
        f"Training: train={n_train:,}, val={n_val:,}, test={n_test:,}, "
        f"epochs={args.epochs}, batch_size={args.batch_size}, seed={args.seed}, "
        f"monitor={args.monitor}"
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=0,
        callbacks=[mean_val_auc_callback, early_stopping, TrainingLogger(logger)],
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
        "shadow_cols": shadow_cols,
        "dropped_features": dropped,
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
