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

import matplotlib

matplotlib.use("Agg")  # headless: reports are written to files
import matplotlib.pyplot as plt
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
from models import (
    DEFAULT_ENCODER,
    available_encoders,
    available_models,
    available_structures,
    build_logistic_model,
    build_model,
    build_single_task_model,
)

ALL_MODELS = tuple(available_models())


def parse_args():
    parser = argparse.ArgumentParser(description="Train ranking baselines.")
    parser.add_argument(
        "--models",
        default="logistic,shared_bottom,single_task",
        help=(
            f"comma-separated subset of {ALL_MODELS}, each optionally with a "
            "feature encoder, e.g. `mmoe+dcn` (the logistic baseline takes no encoder)"
        ),
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
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "comma-separated seeds to repeat every model with, e.g. "
            "`2025,2026,2027`; the report then shows mean±std per task "
            "(default: just --seed)"
        ),
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


def train_single_task(cat_cols, num_cols, vocab, encoder, data, args, task_names):
    train, val, test = data
    per_task, best_epochs, params = {}, {}, 0
    for i, task in enumerate(task_names):
        model = build_single_task_model(
            cat_cols, num_cols, vocab, embed_dim=C.EMBED_DIM,
            units=C.EXPERT_UNITS, tower_units=C.TOWER_UNITS,
            encoder=encoder,
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


def parse_model_spec(value):
    """Parse `structure` or `structure+encoder` into a (structure, encoder) pair.

    A bare structure name uses the default encoder. The logistic baseline has
    no feature encoder by definition, so giving it one is an error rather than
    a silently ignored option.
    """
    structure, _, encoder = value.partition("+")
    structure = structure.strip()
    encoder = encoder.strip() or None
    if structure not in ALL_MODELS:
        raise ValueError(f"unknown model {structure!r}; available: {ALL_MODELS}")
    if encoder is None:
        return structure, None if structure == "logistic" else DEFAULT_ENCODER
    if structure == "logistic":
        raise ValueError(
            "the logistic baseline has no feature encoder; use `logistic` alone"
        )
    if encoder not in available_encoders():
        raise ValueError(
            f"unknown encoder {encoder!r}; available: {available_encoders()}"
        )
    return structure, encoder


def format_model_spec(spec):
    """A (structure, encoder) pair back as the label used in reports."""
    structure, encoder = spec
    return structure if encoder is None else f"{structure}+{encoder}"


def build_multi_task_model(structure, encoder, cat_cols, num_cols, vocab, num_tasks):
    """Build one encoder-fed structure, plus its two-axis run metadata."""
    return build_model(
        encoder=encoder,
        structure=structure,
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
        cat_vocab_size=vocab,
        embed_dim=C.EMBED_DIM,
        num_tasks=num_tasks,
        **C.STRUCTURE_PARAMS[structure],
    )


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


def parse_seeds(value):
    """Parse a comma-separated seed list; None or empty means "not given"."""
    return [
        int(part) for part in (part.strip() for part in (value or "").split(",")) if part
    ] or None


def aggregate_seeds(seed_results, task_names):
    """Mean and sample spread (ddof=1) across independent seed runs.

    Per-task numbers are averaged over seeds; the four-task and gate averages
    are computed per seed first and then averaged, so their spread reflects
    seed-to-seed variation rather than the spread between tasks.
    """
    tasks_mean, tasks_std = {}, {}
    for task in task_names:
        values = np.array([r["tasks"][task] for r in seed_results], dtype=float)
        tasks_mean[task] = float(np.mean(values))
        tasks_std[task] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    def per_seed_average(tasks):
        return [float(np.mean([r["tasks"][task] for task in tasks])) for r in seed_results]

    def mean_std(values):
        return float(np.mean(values)), (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )

    all_mean, all_std = mean_std(per_seed_average(task_names))
    gate_tasks = [task for task in C.GATE_TASKS if task in task_names]
    gate_mean, gate_std = mean_std(per_seed_average(gate_tasks))
    return {
        "tasks_mean": tasks_mean,
        "tasks_std": tasks_std,
        "summary": {
            "mean_all_tasks": all_mean,
            "std_all_tasks": all_std,
            "mean_gate_tasks": gate_mean,
            "std_gate_tasks": gate_std,
        },
    }


def _format_metric(mean, std):
    """`0.7225±0.0012`, or just the mean when there is a single seed."""
    if mean is None or mean != mean:
        return "nan"
    if not std:
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"


def plot_comparison(results, path):
    """Per-task test AUC per model, with error bars for the seed spread."""
    task_names = list(C.LABEL_COLS)
    model_names = list(results["models"])
    positions = np.arange(len(task_names))
    width = 0.8 / max(1, len(model_names))

    fig, ax = plt.subplots(figsize=(11, 5))
    for index, name in enumerate(model_names):
        entry = results["models"][name]
        means = [entry["tasks_mean"].get(task, float("nan")) for task in task_names]
        stds = [entry["tasks_std"].get(task, 0.0) for task in task_names]
        ax.bar(
            positions + index * width - 0.4 + width / 2,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=name,
        )

    seeds = results["config"]["seeds"]
    ax.set_xticks(positions)
    ax.set_xticklabels(task_names)
    ax.set_ylabel("Test AUC")
    # English labels, like main.plot_history: the default matplotlib font has no
    # CJK glyphs, so Chinese would render as empty boxes.
    ax.set_title(
        "Ranking models — test AUC per task "
        f"(seeds={', '.join(map(str, seeds))}, error bars = sample std)"
    )
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_reports(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "baselines.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = out_dir / "baselines.md"
    seeds = results["config"]["seeds"]
    lines = [
        "# 精排对照模型结果",
        "",
        f"- seeds: {', '.join(str(seed) for seed in seeds)}"
        f"{'（mean±std，样本标准差 ddof=1）' if len(seeds) > 1 else ''}"
        f" | epochs: {results['config']['epochs']} "
        f"| early-stop monitor: {results['config']['monitor']}",
        f"- rows (train/val/test): {results['config']['rows']}",
        f"- dropped features: {results['config']['dropped_features'] or 'none'}",
        "",
        "| 模型 | 点击 | 点赞 | 关注 | 评论 | 平均(4任务) | 门控均值(点击/点赞) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in results["models"].items():
        tasks_mean, tasks_std = r["tasks_mean"], r["tasks_std"]
        summary = r["summary"]
        cells = " | ".join(
            _format_metric(tasks_mean.get(t), tasks_std.get(t, 0.0)) for t in C.LABEL_COLS
        )
        lines.append(
            f"| {name} | {cells} | "
            f"{_format_metric(summary['mean_all_tasks'], summary['std_all_tasks'])} | "
            f"{_format_metric(summary['mean_gate_tasks'], summary['std_gate_tasks'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    chart_path = plot_comparison(results, out_dir / "baselines.png")
    return json_path, md_path, chart_path


def train_one_seed(spec, data, args, cat_cols, num_cols, vocab, task_names):
    """Train one model spec at the seed the caller has already set."""
    structure, encoder = spec
    train, val, test = data
    if structure == "single_task":
        return train_single_task(
            cat_cols, num_cols, vocab, encoder, data, args, task_names
        )
    if structure == "logistic":
        model = build_logistic_model(cat_cols, num_cols, vocab, num_tasks=len(task_names))
        return train_multi_output(model, train, val, test, args, task_names)
    model, axes = build_multi_task_model(
        structure, encoder, cat_cols, num_cols, vocab, len(task_names)
    )
    result = train_multi_output(model, train, val, test, args, task_names)
    result["model"] = axes
    return result


def main():
    args = parse_args()
    selected = [parse_model_spec(name) for name in (parse_name_list(args.models) or [])]
    seeds = parse_seeds(args.seeds) or [args.seed]

    set_seed(seeds[0])
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
            "seeds": seeds,
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

    for spec in selected:
        name = format_model_spec(spec)
        started = time.time()
        per_seed = {}
        for seed in seeds:
            seed_started = time.time()
            set_seed(seed)
            print(f"Training {name} (seed {seed})...")
            result = train_one_seed(
                spec, (train, val, test), args, cat_cols, num_cols, vocab, task_names
            )
            result["seed"] = seed
            result["train_seconds"] = round(time.time() - seed_started, 1)
            per_seed[str(seed)] = result
            print(f"  seed {seed}: {result['tasks']} ({result['train_seconds']}s)")

        aggregated = aggregate_seeds(list(per_seed.values()), task_names)
        first = next(iter(per_seed.values()))
        results["models"][name] = {
            "tasks_mean": aggregated["tasks_mean"],
            "tasks_std": aggregated["tasks_std"],
            "summary": aggregated["summary"],
            "per_seed": per_seed,
            "params": first.get("params"),
            "model": first.get("model"),
            "train_seconds": round(time.time() - started, 1),
        }
        print(f"  {name}: {aggregated['summary']} "
              f"(total {results['models'][name]['train_seconds']}s)")

    if args.mmoe_run:
        run_dir = Path(args.mmoe_run)
        if not (run_dir / "metrics.json").exists():
            raise FileNotFoundError(f"no metrics.json under {run_dir}")
        existing = _existing_mmoe_row(run_dir, task_names)
        results["models"]["mmoe_existing_run"] = {
            "tasks_mean": existing["tasks"],
            "tasks_std": {task: 0.0 for task in task_names},
            "summary": {**_summary(existing["tasks"]), "std_all_tasks": 0.0, "std_gate_tasks": 0.0},
            "per_seed": {},
            "params": None,
            "model": None,
            "train_seconds": None,
        }

    json_path, md_path, chart_path = write_reports(results, out_dir)
    print(f"Results: {json_path}")
    print(f"Report:  {md_path}")
    print(f"Chart:   {chart_path}")


if __name__ == "__main__":
    main()
