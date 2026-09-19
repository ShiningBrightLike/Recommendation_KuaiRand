"""User-level ranking metrics for the exposure-set evaluation protocol.

The public KuaiRand logs provide user/video identifiers for each exposed row,
but they do not provide a full-catalog candidate set or an online ranking.
This module therefore evaluates *re-ranking within each user's exposed rows*.
It never treats these metrics as full recommendation recall.

The public functions are NumPy-friendly and deliberately independent of
TensorFlow so they can be unit-tested on small synthetic ranking lists.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import config as C
from data_loading import load_split, load_split_ids


def _as_1d(name, values):
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _validate_arrays(labels, scores, groups, row_ids=None):
    labels = _as_1d("labels", labels).astype(float)
    scores = _as_1d("scores", scores).astype(float)
    groups = _as_1d("groups", groups)
    if not (len(labels) == len(scores) == len(groups)):
        raise ValueError("labels, scores and groups must have the same length")
    if not np.all(np.isfinite(labels)) or not np.all(np.isfinite(scores)):
        raise ValueError("labels and scores must be finite")
    if row_ids is None:
        row_ids = np.arange(len(labels), dtype="int64")
    else:
        row_ids = _as_1d("row_ids", row_ids)
        if len(row_ids) != len(labels):
            raise ValueError("row_ids must have the same length as labels")
    return labels, scores, groups, row_ids


def _group_indices(groups):
    """Return groups in first-seen order, preserving stable report semantics."""
    codes, uniques = {}, []
    indices = []
    for index, value in enumerate(groups.tolist()):
        # Public logs use scalar ids. Stringifying only the dictionary key
        # avoids surprises from NumPy scalar hash/equality implementations.
        key = (type(value).__name__, value)
        if key not in codes:
            codes[key] = len(uniques)
            uniques.append(value)
            indices.append([])
        indices[codes[key]].append(index)
    return uniques, [np.asarray(indexes, dtype="int64") for indexes in indices]


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    return float(np.dot(values, weights) / total) if total else float("nan")


def bootstrap_ci(values, weights=None, n_bootstrap=1000, seed=2025, confidence=0.95):
    """Bootstrap a mean using groups as the resampling units.

    ``values`` must contain one contribution per user. Optional ``weights``
    makes the statistic a weighted mean (the GAUC convention uses impressions
    per user). The returned interval is a percentile interval.
    """
    values = _as_1d("values", values).astype(float)
    if weights is None:
        weights = np.ones(len(values), dtype=float)
    else:
        weights = _as_1d("weights", weights).astype(float)
        if len(weights) != len(values):
            raise ValueError("weights must have the same length as values")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("values and weights must be finite")
    if np.any(weights < 0) or not weights.sum():
        raise ValueError("weights must be non-negative with a positive sum")

    rng = np.random.default_rng(seed)
    estimates = np.empty(int(n_bootstrap), dtype=float)
    group_count = len(values)
    for replicate in range(len(estimates)):
        sampled = rng.integers(0, group_count, size=group_count)
        estimates[replicate] = _weighted_mean(values[sampled], weights[sampled])
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": _weighted_mean(values, weights),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
        "groups": int(group_count),
        "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence),
        "seed": int(seed),
    }


def gauc_details(labels, scores, groups):
    """Return impression-weighted GAUC plus valid-user diagnostics."""
    labels, scores, groups, _ = _validate_arrays(labels, scores, groups)
    _, grouped = _group_indices(groups)
    values, weights = [], []
    for indexes in grouped:
        group_labels = labels[indexes]
        if np.unique(group_labels).size < 2:
            continue
        values.append(float(roc_auc_score(group_labels, scores[indexes])))
        weights.append(len(indexes))
    result = {
        "value": _weighted_mean(values, weights),
        "valid_users": len(values),
        "total_users": len(grouped),
        "impressions": int(sum(weights)),
        "_values": np.asarray(values, dtype=float),
        "_weights": np.asarray(weights, dtype=float),
    }
    return result


def gauc(labels, scores, groups):
    """Return impression-weighted user-level GAUC."""
    return gauc_details(labels, scores, groups)["value"]


def _ranking_group_values(labels, scores, groups, row_ids, k):
    labels, scores, groups, row_ids = _validate_arrays(
        labels, scores, groups, row_ids
    )
    if k < 1:
        raise ValueError("k must be positive")
    _, grouped = _group_indices(groups)
    ndcgs, recalls, maps = [], [], []
    has_positive = []
    has_negative = []
    for indexes in grouped:
        # lexsort uses the last key as primary: descending score first, then
        # ascending row_id for deterministic ties.
        order = np.lexsort((row_ids[indexes], -scores[indexes]))
        ranked = labels[indexes[order]]
        top = ranked[:k]
        positive_count = int(np.sum(ranked > 0))
        has_positive.append(positive_count > 0)
        has_negative.append(positive_count < len(ranked))

        gains = top > 0
        discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
        dcg = float(np.sum(gains * discounts))
        ideal_count = min(k, positive_count)
        ideal = float(np.sum(discounts[:ideal_count]))
        ndcgs.append(dcg / ideal if ideal else 0.0)

        hits = np.cumsum(gains)
        precision = hits / np.arange(1, len(top) + 1)
        denominator = min(k, positive_count)
        maps.append(float(np.sum(precision * gains) / denominator) if denominator else 0.0)
        recalls.append(float(np.sum(gains) / positive_count) if positive_count else 0.0)

    return {
        "ndcg_at_k": np.asarray(ndcgs, dtype=float),
        "recall_at_k": np.asarray(recalls, dtype=float),
        "map_at_k": np.asarray(maps, dtype=float),
        "has_positive": np.asarray(has_positive, dtype=bool),
        "has_negative": np.asarray(has_negative, dtype=bool),
        "total_users": len(grouped),
    }


def ranking_details(labels, scores, groups, row_ids=None, k=10):
    """Return user-macro NDCG/Recall/MAP details for one binary task."""
    labels, scores, groups, row_ids = _validate_arrays(
        labels, scores, groups, row_ids
    )
    details = _ranking_group_values(labels, scores, groups, row_ids, k)
    has_positive = details.pop("has_positive")
    has_negative = details.pop("has_negative")
    result = {
        "ndcg_at_k": float(np.mean(details["ndcg_at_k"])),
        "map_at_k": float(np.mean(details["map_at_k"])),
        "recall_at_k": float(
            np.mean(details["recall_at_k"][has_positive])
            if np.any(has_positive)
            else float("nan")
        ),
        "total_users": int(details["total_users"]),
        "users_with_positive": int(np.sum(has_positive)),
        "users_with_negative": int(np.sum(has_negative)),
        "_ndcg_values": details["ndcg_at_k"],
        "_map_values": details["map_at_k"],
        "_recall_values": details["recall_at_k"][has_positive],
    }
    return result


def ndcg_at_k(labels, scores, groups, row_ids=None, k=10):
    return ranking_details(labels, scores, groups, row_ids, k)["ndcg_at_k"]


def recall_at_k(labels, scores, groups, row_ids=None, k=10):
    return ranking_details(labels, scores, groups, row_ids, k)["recall_at_k"]


def map_at_k(labels, scores, groups, row_ids=None, k=10):
    return ranking_details(labels, scores, groups, row_ids, k)["map_at_k"]


def ece_details(labels, probabilities, n_bins=10):
    """Return expected calibration error and fixed-width bin diagnostics."""
    labels = _as_1d("labels", labels).astype(float)
    probabilities = _as_1d("probabilities", probabilities).astype(float)
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if not 1 <= n_bins:
        raise ValueError("n_bins must be positive")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    bin_ids = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bin_ids, minlength=n_bins).astype(float)
    confidence_sum = np.bincount(bin_ids, weights=probabilities, minlength=n_bins)
    accuracy_sum = np.bincount(bin_ids, weights=labels, minlength=n_bins)
    nonempty = counts > 0
    confidence = np.divide(
        confidence_sum, counts, out=np.zeros(n_bins), where=nonempty
    )
    accuracy = np.divide(accuracy_sum, counts, out=np.zeros(n_bins), where=nonempty)
    result = {
        "value": float(
            np.sum((counts[nonempty] / len(labels)) * np.abs(accuracy[nonempty] - confidence[nonempty]))
        ),
        "bins": [
            {
                "lower": i / n_bins,
                "upper": (i + 1) / n_bins,
                "count": int(counts[i]),
                "confidence": float(confidence[i]),
                "accuracy": float(accuracy[i]),
            }
            for i in range(n_bins)
            if nonempty[i]
        ],
    }
    return result


def ece(labels, probabilities, n_bins=10):
    return ece_details(labels, probabilities, n_bins)["value"]


def _ece_group_statistics(labels, probabilities, groups, n_bins):
    """Return per-user sufficient statistics for fixed-width ECE bins.

    Keeping counts, confidence sums, and label sums lets the bootstrap resample
    users without materialising a row-level copy for every replicate.
    """
    labels = _as_1d("labels", labels).astype(float)
    probabilities = _as_1d("probabilities", probabilities).astype(float)
    groups = _as_1d("groups", groups)
    if len(labels) != len(probabilities) or len(labels) != len(groups):
        raise ValueError("labels, probabilities and groups must have the same length")
    if not np.all(np.isfinite(labels)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("labels and probabilities must be finite")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be in [0, 1]")

    _, grouped = _group_indices(groups)
    counts = np.zeros((len(grouped), n_bins), dtype=float)
    confidence_sum = np.zeros_like(counts)
    accuracy_sum = np.zeros_like(counts)
    for user_index, indexes in enumerate(grouped):
        bin_ids = np.minimum(
            (probabilities[indexes] * n_bins).astype(int), n_bins - 1
        )
        counts[user_index] = np.bincount(bin_ids, minlength=n_bins)
        confidence_sum[user_index] = np.bincount(
            bin_ids, weights=probabilities[indexes], minlength=n_bins
        )
        accuracy_sum[user_index] = np.bincount(
            bin_ids, weights=labels[indexes], minlength=n_bins
        )
    return counts, confidence_sum, accuracy_sum


def _bootstrap_ece(counts, confidence_sum, accuracy_sum, estimate,
                   n_bootstrap=1000, seed=2025, confidence=0.95):
    """Bootstrap global ECE while resampling users as the statistical unit."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    group_count = len(counts)
    if group_count == 0:
        raise ValueError("at least one user is required")

    rng = np.random.default_rng(seed)
    estimates = np.empty(int(n_bootstrap), dtype=float)
    for replicate in range(len(estimates)):
        sampled = rng.integers(0, group_count, size=group_count)
        sampled_counts = counts[sampled].sum(axis=0)
        sampled_confidence = confidence_sum[sampled].sum(axis=0)
        sampled_accuracy = accuracy_sum[sampled].sum(axis=0)
        total = float(sampled_counts.sum())
        nonempty = sampled_counts > 0
        bin_confidence = np.divide(
            sampled_confidence, sampled_counts,
            out=np.zeros_like(sampled_counts), where=nonempty,
        )
        bin_accuracy = np.divide(
            sampled_accuracy, sampled_counts,
            out=np.zeros_like(sampled_counts), where=nonempty,
        )
        estimates[replicate] = float(
            np.sum((sampled_counts[nonempty] / total)
                   * np.abs(bin_accuracy[nonempty] - bin_confidence[nonempty]))
        )
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(estimate),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
        "groups": int(group_count),
        "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence),
        "seed": int(seed),
    }


def evaluate_task(labels, scores, groups, row_ids=None, k=10, n_bins=10,
                  bootstrap=1000, seed=2025):
    """Evaluate one binary task and attach user-level bootstrap intervals."""
    labels, scores, groups, row_ids = _validate_arrays(
        labels, scores, groups, row_ids
    )
    gauc_result = gauc_details(labels, scores, groups)
    rank_result = ranking_details(labels, scores, groups, row_ids, k)
    calibration = ece_details(labels, scores, n_bins)
    ece_counts, ece_confidence, ece_accuracy = _ece_group_statistics(
        labels, scores, groups, n_bins
    )
    result = {
        "gauc": gauc_result["value"],
        "gauc_users": gauc_result["valid_users"],
        "users": rank_result["total_users"],
        "users_with_positive": rank_result["users_with_positive"],
        "users_with_negative": rank_result["users_with_negative"],
        f"ndcg@{k}": rank_result["ndcg_at_k"],
        f"recall@{k}": rank_result["recall_at_k"],
        f"map@{k}": rank_result["map_at_k"],
        "ece": calibration["value"],
        "ece_users": int(len(ece_counts)),
        "calibration_bins": calibration["bins"],
        "bootstrap": {
            "gauc": bootstrap_ci(
                gauc_result["_values"], gauc_result["_weights"], bootstrap, seed
            ) if gauc_result["valid_users"] else None,
            f"ndcg@{k}": bootstrap_ci(
                rank_result["_ndcg_values"], n_bootstrap=bootstrap, seed=seed
            ),
            f"recall@{k}": bootstrap_ci(
                rank_result["_recall_values"], n_bootstrap=bootstrap, seed=seed
            ) if rank_result["users_with_positive"] else None,
            f"map@{k}": bootstrap_ci(
                rank_result["_map_values"], n_bootstrap=bootstrap, seed=seed
            ),
            "ece": _bootstrap_ece(
                ece_counts, ece_confidence, ece_accuracy,
                calibration["value"], n_bootstrap=bootstrap, seed=seed,
            ),
        },
    }
    return result


def evaluate_user_ranking(labels, predictions, groups, row_ids=None,
                          task_names=None, k=10, n_bins=10,
                          bootstrap=1000, seed=2025):
    """Evaluate all task columns using the same user-level candidate groups."""
    if task_names is None:
        task_names = list(C.LABEL_COLS)
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.ndim == 1:
        labels = labels[:, None]
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same 2-D shape")
    if labels.shape[1] != len(task_names):
        raise ValueError("task_names must match the prediction columns")
    return {
        task: evaluate_task(
            labels[:, index], predictions[:, index], groups, row_ids,
            k=k, n_bins=n_bins, bootstrap=bootstrap, seed=seed + index,
        )
        for index, task in enumerate(task_names)
    }


def _format(value, digits=4):
    return "nan" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _format_interval(metrics, name):
    value = metrics.get(name)
    interval = metrics.get("bootstrap", {}).get(name)
    if interval is None:
        return _format(value)
    return f"{_format(value)} [{_format(interval['lower'])}, {_format(interval['upper'])}]"


def write_report(report, json_path, markdown_path):
    """Write a machine-readable JSON and compact Markdown report."""
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    k = report["protocol"]["k"]
    lines = [
        "# 用户级曝光内排序评估",
        "",
        f"- split: **{report['protocol']['split']}**",
        "- 候选集合：该 split 内每个用户的全部曝光行",
        f"- K：{k}；用户级 bootstrap 重采样次数：{report['protocol']['bootstrap']}",
        "- 这是曝光集合内重排序评估，不代表全量视频召回效果。",
        "",
        f"| 任务 | GAUC [95% CI] | NDCG@{k} [95% CI] | Recall@{k} [95% CI] | MAP@{k} [95% CI] | ECE [95% CI] | 用户数 | GAUC 有效用户 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, metrics in report["tasks"].items():
        lines.append(
            f"| {task} | {_format_interval(metrics, 'gauc')} | "
            f"{_format_interval(metrics, f'ndcg@{k}')} | "
            f"{_format_interval(metrics, f'recall@{k}')} | "
            f"{_format_interval(metrics, f'map@{k}')} | "
            f"{_format_interval(metrics, 'ece')} | "
            f"{metrics['users']} | {metrics['gauc_users']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--final-confirmation", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.split == "test" and not args.final_confirmation:
        raise ValueError("test split requires --final-confirmation")
    if args.k < 1 or args.bootstrap < 1:
        raise ValueError("--k and --bootstrap must be positive")
    import tensorflow as tf
    from models import custom_objects

    inputs, labels, rows, _ = load_split(args.split, args.max_rows)
    ids = load_split_ids(args.split, args.max_rows)
    if len(ids) != rows:
        raise ValueError("id sidecar row count does not match loaded split")
    model = tf.keras.models.load_model(args.model, custom_objects=custom_objects())
    predictions = model.predict(inputs, batch_size=args.batch_size, verbose=0)
    if not isinstance(predictions, list):
        predictions = [predictions]
    label_matrix = np.column_stack([target.reshape(-1) for target in labels])
    prediction_matrix = np.column_stack([np.asarray(pred).reshape(-1) for pred in predictions])
    report = {
        "protocol_status": "final_confirmation" if args.split == "test" else "valid_for_model_selection",
        "protocol": {
            "split": args.split,
            "candidate_set": "该数据划分内每个用户的全部曝光行",
            "candidate_set_zh": "该数据划分内每个用户的全部曝光行",
            "group_key": "user_id",
            "k": args.k,
            "bins": args.bins,
            "bootstrap": args.bootstrap,
            "bootstrap_seed": args.seed,
            "rows": rows,
            "users": int(ids["user_id"].nunique()),
            "final_confirmation": bool(args.final_confirmation),
        },
        "model": str(args.model),
        "model_params": int(model.count_params()),
        "tasks": evaluate_user_ranking(
            label_matrix,
            prediction_matrix,
            ids["user_id"].to_numpy(),
            ids["row_id"].to_numpy(),
            task_names=list(C.LABEL_COLS),
            k=args.k,
            n_bins=args.bins,
            bootstrap=args.bootstrap,
            seed=args.seed,
        ),
    }
    out_dir = args.out_dir or Path(args.model).parent / "ranking_eval" / args.split
    write_report(report, out_dir / "ranking_metrics.json", out_dir / "ranking_metrics.md")
    print(f"JSON: {out_dir / 'ranking_metrics.json'}")
    print(f"Report: {out_dir / 'ranking_metrics.md'}")


if __name__ == "__main__":
    main()
