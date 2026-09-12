"""Train ranking baselines and compare per-task test AUC.

Baselines (same data/protocol/seed as main.py):
    - logistic:      one weight per one-hot/numeric feature, no hidden layer
    - shared_bottom: one shared trunk + per-task towers
    - single_task:   one independent MLP per task (trained separately)
    - mmoe:          the current MMoE model, for a matched-protocol reference

Usage:
    python baselines.py
    python baselines.py --models logistic,shared_bottom
    python baselines.py --mmoe-run KuaiRand-Pure/saved/runs/fi-shadow_xxx
    python baselines.py --drop-stat-features --tag baselines_nostats
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
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
from main import MeanValAUC, set_seed
from MMoE_model import (
    build_logistic_model,
    build_mmoe_model,
    build_shared_bottom_model,
    build_single_task_model,
)

ALL_MODELS = ("logistic", "shared_bottom", "single_task", "mmoe")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ranking baselines.")
    parser.add_argument(
        "--models",
        default="logistic,shared_bottom,single_task",
        help=f"comma-separated subset of {ALL_MODELS}",
    )
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=C.EARLY_STOP_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=C.LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--monitor",
        default="val_auc_mean",
        choices=["val_auc_mean", "val_loss"]
        + [f"val_output_{i}_auc" for i in range(1, len(C.LABEL_COLS) + 1)],
        help="early-stopping metric for multi-output baselines",
    )
    parser.add_argument("--tag", default="baselines")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mmoe-run", type=Path, default=None, help="existing run dir to add as a row")
    parser.add_argument("--drop-features", default=None)
    parser.add_argument("--drop-stat-features", action="store_true")
    return parser.parse_args()


def _load_data(args, cat_cols, num_cols):
    return (
        load_split("train", args.max_rows, cat_cols, num_cols),
        load_split("val", args.max_rows, cat_cols, num_cols),
        load_split("test", args.max_rows, cat_cols, num_cols),
    )


def _compile_and_fit(model, train, val, args, monitor, task_names, single_output=False):
    output_names = [f"output_{i + 1}" for i in range(len(task_names))]
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    if single_output:
        # Keras flattens metric names for single-output models: val_auc, not
        # val_output_1_auc.
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[AUC(name="auc")],
        )
        monitor = "val_auc"
    else:
        model.compile(
            optimizer=optimizer,
            loss={name: "binary_crossentropy" for name in output_names},
            loss_weights={
                name: C.LOSS_WEIGHTS[task] for name, task in zip(output_names, task_names)
            },
            metrics={name: [AUC(name="auc")] for name in output_names},
        )
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode="min" if monitor == "val_loss" else "max",
        patience=args.patience,
        restore_best_weights=True,
        verbose=0,
    )
    gate_indices = [
        task_names.index(task) + 1 for task in C.GATE_TASKS if task in task_names
    ]
    model.fit(
        train[0],
        train[1],
        validation_data=(val[0], val[1]),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=0,
        callbacks=[MeanValAUC(gate_indices), early_stopping],
    )
    best_epoch = getattr(early_stopping, "best_epoch", None)
    return None if best_epoch is None else int(best_epoch) + 1


def _test_auc(model, test, task_names, single_output=False):
    metrics = model.evaluate(
        test[0], test[1], batch_size=4096, verbose=0, return_dict=True
    )
    if single_output:
        return {task_names[0]: float(metrics["auc"])}
    return {
        task: float(metrics[f"output_{i + 1}_auc"])
        for i, task in enumerate(task_names)
    }


def train_multi_output(model, train, val, test, args, task_names):
    best_epoch = _compile_and_fit(
        model, train, val, args, args.monitor, task_names
    )
    return {
        "tasks": _test_auc(model, test, task_names),
        "best_epoch": best_epoch,
        "params": int(model.count_params()),
    }


def train_single_task(cat_cols, num_cols, vocab, data, args, task_names):
    train, val, test = data
    per_task, best_epochs, params = {}, {}, 0
    for i, task in enumerate(task_names):
        model = build_single_task_model(
            cat_cols, num_cols, vocab, embed_dim=C.EMBED_DIM,
            units=C.EXPERT_UNITS, tower_units=C.TOWER_UNITS,
        )
        train_i = (train[0], [train[1][i]])
        val_i = (val[0], [val[1][i]])
        test_i = (test[0], [test[1][i]])
        best_epoch = _compile_and_fit(
            model, train_i, val_i, args, "val_auc", [task], single_output=True
        )
        auc = _test_auc(model, test_i, [task], single_output=True)
        per_task.update(auc)
        best_epochs[task] = best_epoch
        params += int(model.count_params())
    return {"tasks": per_task, "best_epochs": best_epochs, "params": params}


def build_baseline(name, cat_cols, num_cols, vocab, num_tasks):
    if name == "logistic":
        return build_logistic_model(cat_cols, num_cols, vocab, num_tasks=num_tasks)
    if name == "shared_bottom":
        return build_shared_bottom_model(
            cat_cols, num_cols, vocab, embed_dim=C.EMBED_DIM,
            num_tasks=num_tasks, bottom_units=C.EXPERT_UNITS,
            tower_units=C.TOWER_UNITS,
        )
    if name == "mmoe":
        return build_mmoe_model(
            cat_cols, num_cols, vocab, embed_dim=C.EMBED_DIM,
            num_experts=C.NUM_EXPERTS, num_tasks=num_tasks,
            units=C.EXPERT_UNITS, tower_units=C.TOWER_UNITS,
        )
    raise ValueError(f"unknown baseline: {name}")


def _existing_mmoe_row(run_dir, task_names):
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    tasks = {
        task: float(metrics["test_metrics"][f"output_{i + 1}_auc"])
        for i, task in enumerate(task_names)
    }
    return {"tasks": tasks, "best_epoch": metrics.get("best_epoch"), "params": None}


def _summary(tasks):
    values = [v for v in tasks.values() if v == v]
    gate = [tasks[t] for t in C.GATE_TASKS if t in tasks]
    return {
        "mean_all_tasks": float(np.mean(values)) if values else None,
        "mean_gate_tasks": float(np.mean(gate)) if gate else None,
    }


def write_reports(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "baselines.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = out_dir / "baselines.md"
    lines = [
        "# 精排对照模型结果",
        "",
        f"- seed: {results['config']['seed']} | epochs: {results['config']['epochs']} "
        f"| early-stop monitor: {results['config']['monitor']}",
        f"- rows (train/val/test): {results['config']['rows']}",
        f"- dropped features: {results['config']['dropped_features'] or 'none'}",
        "",
        "| 模型 | 点击 | 点赞 | 关注 | 评论 | 平均(4任务) | 门控均值(点击/点赞) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in results["models"].items():
        tasks = r["tasks"]
        s = _summary(tasks)
        cells = " | ".join(f"{tasks.get(t, float('nan')):.4f}" for t in C.LABEL_COLS)
        lines.append(
            f"| {name} | {cells} | {s['mean_all_tasks']:.4f} | {s['mean_gate_tasks']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main():
    args = parse_args()
    selected = parse_name_list(args.models) or []
    unknown = sorted(set(selected) - set(ALL_MODELS))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")

    set_seed(args.seed)
    cat_cols, num_cols, shadow_cols = load_feature_schema()
    drop = set(parse_name_list(args.drop_features) or [])
    if args.drop_stat_features:
        drop |= set(video_statistic_cols())
    cat_cols, num_cols, dropped = filter_features(cat_cols, num_cols, drop)
    vocab = load_cat_vocab_size()
    task_names = list(C.LABEL_COLS)

    print(f"Schema after filtering: {len(cat_cols)} cat + {len(num_cols)} num; dropped={dropped}")
    train, val, test = _load_data(args, cat_cols, num_cols)
    rows = (train[2], val[2], test[2])
    print(f"Rows train/val/test: {rows}")

    out_dir = args.out_dir or C.RUNS_DIR / f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    results = {
        "config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "monitor": args.monitor,
            "rows": rows,
            "shadow_cols": shadow_cols,
            "dropped_features": dropped,
            "features": {"categorical": len(cat_cols), "numeric": len(num_cols)},
        },
        "models": {},
    }

    for name in selected:
        started = time.time()
        print(f"Training {name}...")
        if name == "single_task":
            result = train_single_task(cat_cols, num_cols, vocab, (train, val, test), args, task_names)
        else:
            model = build_baseline(name, cat_cols, num_cols, vocab, len(task_names))
            result = train_multi_output(model, train, val, test, args, task_names)
        result["train_seconds"] = round(time.time() - started, 1)
        results["models"][name] = result
        print(f"  {name}: {result['tasks']} ({result['train_seconds']}s)")

    if args.mmoe_run:
        run_dir = Path(args.mmoe_run)
        if not (run_dir / "metrics.json").exists():
            raise FileNotFoundError(f"no metrics.json under {run_dir}")
        results["models"]["mmoe_existing_run"] = _existing_mmoe_row(run_dir, task_names)

    json_path, md_path = write_reports(results, out_dir)
    print(f"Results: {json_path}")
    print(f"Report:  {md_path}")


if __name__ == "__main__":
    main()
