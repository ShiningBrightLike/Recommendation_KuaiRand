"""Permutation feature importance for a trained MMoE model.

For every feature the module repeatedly shuffles that feature's values on the
feature-decision set (validation split by default), re-runs the trained model,
and measures the drop in per-task AUC. Larger drops mean the feature matters
more for those tasks.

Design notes (see docs/adr/0002):
    - Reports keep raw per-task drops, including negative values (AUC can go
      up after shuffling a redundant feature); no silent correction.
    - Ranking and gating use the mean *absolute* drop over the gate tasks
      (default: is_click, is_like; sparse tasks stay in the report).
    - Decision: pass if overall_mean - overall_std >= cutoff; review if only
      the mean clears the cutoff; reject otherwise.
    - Features named shadow_* (injected at training time via
      `data_process.py --shadow-features N`) are random noise by construction
      and act as the null reference for the cutoff.

Usage:
    python feature_importance.py --model KuaiRand-Pure/saved/runs/<run>/model.keras
    python feature_importance.py --model ... --repeats 5 --max-rows 20000
    python feature_importance.py --model ... --candidate-cols shadow_0,new_feat_a
    python feature_importance.py --model ... --smoke
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, default_rng
from sklearn.metrics import roc_auc_score
import tensorflow as tf

import config as C
from data_loading import load_feature_schema, load_split
from MMoE_model import MMoE

SHADOW_PREFIX = "shadow_"

DECISION_PASS = "通过"
DECISION_REVIEW = "待确认"
DECISION_REJECT = "不通过"


class FeatureSpec(NamedTuple):
    name: str
    kind: str  # "categorical" | "numeric"
    input_index: int
    column_index: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Permutation feature importance (AUC drop) for a trained model."
    )
    parser.add_argument("--model", type=Path, required=True, help="path to model.keras")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="evaluation split; keep the default to respect the ADR-0001 discipline",
    )
    parser.add_argument("--repeats", type=int, default=3, help="shuffle repeats per feature")
    parser.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="limit the split to N rows (quick debugging)",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.001,
        help="minimum mean absolute AUC drop for gate consideration",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(C.GATE_TASKS),
        choices=C.LABEL_COLS,
        help="gate tasks used for overall importance/decision",
    )
    parser.add_argument(
        "--candidate-cols",
        type=str,
        default=None,
        help="comma-separated feature subset to analyze (default: all)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.9,
        help="report numeric feature pairs above this |Pearson r| as correlated",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="how many features to draw in the bar chart",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: <model_dir>/feature_importance)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="shortcut: 2048 rows, 1 repeat, batch 1024, top 10",
    )
    return parser.parse_args()


def _auc(y_true, y_score):
    """ROC AUC for one binary task; NaN when either class is absent."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def build_column_descriptors(categorical_cols, numeric_cols):
    """Map every feature to where it lives inside the model input list."""
    specs = []
    for i, name in enumerate(categorical_cols):
        specs.append(FeatureSpec(name=name, kind="categorical", input_index=i, column_index=0))
    numeric_input_index = len(categorical_cols)
    for j, name in enumerate(numeric_cols):
        specs.append(
            FeatureSpec(
                name=name,
                kind="numeric",
                input_index=numeric_input_index,
                column_index=j,
            )
        )
    return specs


def shuffled_inputs(inputs, spec, rng):
    """Return a copy of the input list with one feature permuted."""
    shuffled = list(inputs)
    source = inputs[spec.input_index]
    if spec.kind == "categorical":
        shuffled[spec.input_index] = rng.permutation(source)
    else:
        copy = source.copy()
        copy[:, spec.column_index] = rng.permutation(copy[:, spec.column_index])
        shuffled[spec.input_index] = copy
    return shuffled


def _stats(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(finite))
    std = (
        float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    )
    return mean, std


def decide(overall_mean, overall_std, cutoff):
    if overall_mean - overall_std >= cutoff:
        return DECISION_PASS
    if overall_mean >= cutoff:
        return DECISION_REVIEW
    return DECISION_REJECT


def _correlated_pairs(numeric_matrix, numeric_cols, threshold):
    if numeric_matrix.shape[1] < 2:
        return []
    corr = np.corrcoef(numeric_matrix, rowvar=False)
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            r = float(corr[i, j])
            if np.isfinite(r) and abs(r) >= threshold:
                pairs.append(
                    {
                        "feature_a": numeric_cols[i],
                        "feature_b": numeric_cols[j],
                        "correlation": round(r, 4),
                    }
                )
    return pairs


def analyze(
    model,
    inputs,
    targets,
    task_names,
    specs,
    numeric_cols,
    gate_tasks=None,
    repeats=3,
    seed=C.RANDOM_SEED,
    batch_size=4096,
    cutoff=0.001,
    correlation_threshold=0.9,
    verbose=True,
):
    """Run permutation importance and return a fully serialisable report dict."""
    if gate_tasks is None:
        gate_tasks = list(C.GATE_TASKS)
    for t in gate_tasks:
        if t not in task_names:
            raise ValueError(f"gate task {t!r} not in model task set {task_names}")

    started = time.time()
    baseline_preds = model.predict(inputs, batch_size=batch_size, verbose=0)
    baseline_auc = {
        task: _auc(y, pred) for task, y, pred in zip(task_names, targets, baseline_preds)
    }
    if verbose:
        print(f"Baseline AUC: {baseline_auc}")

    rows = []
    for spec_index, spec in enumerate(specs):
        drops_by_task = {task: [] for task in task_names}
        for repeat in range(repeats):
            rng = default_rng(SeedSequence([seed, spec_index, repeat]))
            preds = model.predict(
                shuffled_inputs(inputs, spec, rng),
                batch_size=batch_size,
                verbose=0,
            )
            for task, y, pred in zip(task_names, targets, preds):
                base = baseline_auc[task]
                perm = _auc(y, pred)
                drops_by_task[task].append(float("nan") if np.isnan(base) or np.isnan(perm) else base - perm)

        per_task = {}
        for task in task_names:
            mean, std = _stats(drops_by_task[task])
            per_task[task] = {
                "mean": mean,
                "std": std,
                "drops": drops_by_task[task],
            }

        # Gate signal: per-task mean absolute drop, averaged over gate tasks.
        # Gate noise: per-task repeat-level std, averaged over gate tasks.
        # (Differences between tasks are signal, not noise.)
        gate_abs_means, gate_abs_stds = [], []
        for task in gate_tasks:
            task_abs = [
                abs(d) for d in drops_by_task[task] if not np.isnan(d)
            ]
            m, s = _stats(task_abs)
            gate_abs_means.append(m)
            gate_abs_stds.append(s)
        overall_mean = (
            float(np.nanmean(gate_abs_means)) if gate_abs_means else float("nan")
        )
        overall_std = (
            float(np.nanmean(gate_abs_stds)) if gate_abs_stds else 0.0
        )
        decision = (
            decide(overall_mean, overall_std, cutoff)
            if gate_abs_means and not np.isnan(overall_mean)
            else DECISION_REJECT
        )
        rows.append(
            {
                "feature": spec.name,
                "kind": spec.kind,
                "is_shadow": spec.name.startswith(SHADOW_PREFIX),
                "overall_mean": overall_mean,
                "overall_std": overall_std,
                "decision": decision,
                "per_task": per_task,
            }
        )
        if verbose:
            elapsed = time.time() - started
            print(
                f"[{spec_index + 1}/{len(specs)}] {spec.name}: "
                f"overall={overall_mean:.5f}±{overall_std:.5f} "
                f"({decision}) | elapsed {elapsed:.1f}s"
            )

    if verbose:
        print(f"Permutation analysis finished in {time.time() - started:.1f}s")

    # Locate the numeric matrix through the first numeric spec (categorical
    # inputs precede the single numeric input).
    numeric_specs = [s for s in specs if s.kind == "numeric"]
    if numeric_specs:
        numeric_matrix = np.asarray(inputs[numeric_specs[0].input_index])
    else:
        numeric_matrix = None

    shadow_rows = [r for r in rows if r["is_shadow"]]
    if shadow_rows:
        shadow_mean, shadow_std = _stats([r["overall_mean"] for r in shadow_rows])
        noise_reference = {
            "present": True,
            "features": [r["feature"] for r in shadow_rows],
            "overall_mean": shadow_mean,
            "overall_std": shadow_std,
        }
    else:
        noise_reference = {
            "present": False,
            "note": (
                "no shadow features in this model; noise protection is the "
                "overall_std term. Rerun with `data_process.py --shadow-features N` "
                "and retrain for an explicit null reference."
            ),
        }

    correlated = (
        _correlated_pairs(
            np.asarray(numeric_matrix, dtype=np.float64),
            numeric_cols,
            correlation_threshold,
        )
        if numeric_matrix is not None
        else []
    )

    return {
        "baseline_auc": baseline_auc,
        "gate_tasks": gate_tasks,
        "cutoff": cutoff,
        "repeats": repeats,
        "seed": seed,
        "decision_summary": dict(Counter(r["decision"] for r in rows)),
        "noise_reference": noise_reference,
        "correlated_pairs": correlated,
        "features": rows,
    }


def _mean_std_str(mean, std):
    if mean != mean:  # NaN
        return "nan"
    return f"{mean:.5f} ± {std:.5f}"


def to_dataframe(result, task_names):
    records = []
    for row in result["features"]:
        record = {
            "feature": row["feature"],
            "kind": row["kind"],
            "is_shadow": row["is_shadow"],
            "overall_mean": row["overall_mean"],
            "overall_std": row["overall_std"],
            "decision": row["decision"],
        }
        for task in task_names:
            record[f"drop_{task}_mean"] = row["per_task"][task]["mean"]
            record[f"drop_{task}_std"] = row["per_task"][task]["std"]
        records.append(record)
    return records


def write_reports(result, task_names, out_dir, top=20, run_meta=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = to_dataframe(result, task_names)
    records_sorted = sorted(
        records, key=lambda r: (np.isnan(r["overall_mean"]), -(r["overall_mean"] or 0.0))
    )

    json_path = out_dir / "importance.json"
    payload = {
        **result,
        "run_meta": run_meta,
        "report_order": [r["feature"] for r in records_sorted],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "importance.csv"
    cols = list(records[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(records_sorted)

    md_path = out_dir / "importance.md"
    lines = [
        "# 置换重要度报告",
        "",
        f"- model: `{run_meta.get('model', '') if run_meta else ''}`",
        f"- split: `{run_meta.get('split', '') if run_meta else ''}` / rows: "
        f"`{run_meta.get('rows', '') if run_meta else ''}`",
        f"- gate tasks: {', '.join(result['gate_tasks'])} | cutoff: {result['cutoff']}",
        f"- baseline AUC: {json.dumps(result['baseline_auc'], ensure_ascii=False)}",
        "",
        "## 结论（按总体重要度排序）",
        "",
        "| 特征 | 类型 | 影子 | 总体(均值±std) | 判定 | "
        + " | ".join(task_names)
        + " |",
        "| --- | --- | --- | --- | --- | " + " | ".join(["---"] * len(task_names)) + " |",
    ]
    for r in records_sorted:
        lines.append(
            f"| {r['feature']} | {r['kind']} | {r['is_shadow']} | "
            f"{_mean_std_str(r['overall_mean'], r['overall_std'])} | {r['decision']} | "
            + " | ".join(_mean_std_str(r[f"drop_{t}_mean"], r[f"drop_{t}_std"]) for t in task_names)
            + " |"
        )
    lines += [
        "",
        "## 噪声参考",
        "",
        json.dumps(result["noise_reference"], ensure_ascii=False, indent=2),
        "",
        "## 相关特征提示（置换重要度会被摊薄，请合并解读）",
        "",
    ]
    if result["correlated_pairs"]:
        lines += [
            "| 特征 A | 特征 B | Pearson r |",
            "| --- | --- | --- |",
        ]
        lines += [
            f"| {p['feature_a']} | {p['feature_b']} | {p['correlation']} |"
            for p in result["correlated_pairs"]
        ]
    else:
        lines.append("_无超过阈值的数值特征对_")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    chart_path = out_dir / "importance_top.png"
    plot_importance(records_sorted, chart_path, result["cutoff"], top=top)
    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path,
        "chart": chart_path,
    }


def plot_importance(records_sorted, path, cutoff, top=20):
    valid = [r for r in records_sorted if not np.isnan(r["overall_mean"])]
    valid = valid[:top] if top else valid
    if not valid:
        return
    valid = valid[::-1]  # largest at the top of the horizontal bar chart
    names = [r["feature"] for r in valid]
    means = [r["overall_mean"] for r in valid]
    stds = [r["overall_std"] for r in valid]
    colors = ["tab:orange" if r["is_shadow"] else "tab:blue" for r in valid]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(valid))))
    ax.barh(range(len(valid)), means, xerr=stds, color=colors, alpha=0.85)
    ax.axvline(cutoff, color="red", linestyle="--", linewidth=1, label=f"cutoff={cutoff}")
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Overall importance (mean absolute AUC drop over gate tasks)")
    ax.set_title("Permutation Feature Importance (AUC drop)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    if args.smoke:
        args.max_rows = args.max_rows or 2048
        args.repeats = 1
        args.batch_size = 1024
        args.top = 10

    cat_cols, num_cols, shadow_cols = load_feature_schema()
    specs = build_column_descriptors(cat_cols, num_cols)

    if args.candidate_cols:
        wanted = set(args.candidate_cols)
        specs = [s for s in specs if s.name in wanted]
        missing = wanted - {s.name for s in specs}
        if missing:
            raise ValueError(f"unknown candidate columns: {sorted(missing)}")
    if not specs:
        raise ValueError("no features to analyze (empty candidate list?)")

    print(f"Loading split '{args.split}' (schema: {len(cat_cols)} cat + {len(num_cols)} num)...")
    inputs, targets, n_rows, _ = load_split(
        args.split, args.max_rows, cat_cols, num_cols
    )
    print(f"Loaded {n_rows:,} rows; analyzing {len(specs)} features × {args.repeats} repeats")

    model = tf.keras.models.load_model(
        str(args.model), custom_objects={"MMoE": MMoE}
    )
    if len(model.inputs) != len(cat_cols) + 1:
        raise ValueError(
            f"model has {len(model.inputs)} inputs but schema expects "
            f"{len(cat_cols) + 1} ({len(cat_cols)} categorical + 1 numeric). "
            "The model and pipeline_meta.json are out of sync; retrain with "
            "`python main.py` after `python data_process.py`."
        )
    task_names = list(C.LABEL_COLS)
    result = analyze(
        model=model,
        inputs=inputs,
        targets=targets,
        task_names=task_names,
        specs=specs,
        numeric_cols=num_cols,
        gate_tasks=args.tasks,
        repeats=args.repeats,
        seed=args.seed,
        batch_size=args.batch_size,
        cutoff=args.cutoff,
        correlation_threshold=args.correlation_threshold,
        verbose=True,
    )

    out_dir = args.out_dir or (args.model.parent / "feature_importance")
    run_meta = {
        "model": str(args.model),
        "split": args.split,
        "rows": n_rows,
        "shadow_cols": shadow_cols,
        "candidate_cols": args.candidate_cols,
    }
    paths = write_reports(result, task_names, out_dir, top=args.top, run_meta=run_meta)
    print(f"Reports written to {out_dir}:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
