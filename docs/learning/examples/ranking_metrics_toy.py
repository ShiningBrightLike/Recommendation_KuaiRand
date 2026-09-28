"""Tiny, dependency-free example for within-exposure ranking metrics."""

from __future__ import annotations

import math
from collections import defaultdict


ROWS = [
    ("u1", "v1", 1, 0.90),
    ("u1", "v2", 0, 0.80),
    ("u1", "v3", 1, 0.20),
    ("u2", "v4", 0, 0.70),
    ("u2", "v5", 1, 0.60),
    ("u2", "v6", 0, 0.10),
]


def dcg(labels: list[int], k: int) -> float:
    return sum((2**label - 1) / math.log2(rank + 2) for rank, label in enumerate(labels[:k]))


def per_user_metrics(rows: list[tuple[str, str, int, float]], k: int = 2) -> dict[str, tuple[float, float, float]]:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for user_id, _video_id, label, score in rows:
        grouped[user_id].append((label, score))

    result = {}
    for user_id, items in grouped.items():
        ranked = [label for label, _score in sorted(items, key=lambda item: item[1], reverse=True)]
        ideal = sorted(ranked, reverse=True)
        ndcg = dcg(ranked, k) / dcg(ideal, k) if dcg(ideal, k) else 0.0
        positives = sum(ranked)
        recall = sum(ranked[:k]) / positives if positives else 0.0
        hits = 0
        precision_sum = 0.0
        for position, label in enumerate(ranked[:k], start=1):
            if label:
                hits += 1
                precision_sum += hits / position
        average_precision = precision_sum / min(positives, k) if positives else 0.0
        result[user_id] = (ndcg, recall, average_precision)
    return result


def ece(rows: list[tuple[str, str, int, float]], bins: int = 4) -> float:
    buckets = [[] for _ in range(bins)]
    for _user_id, _video_id, label, score in rows:
        index = min(int(score * bins), bins - 1)
        buckets[index].append((label, score))
    total = len(rows)
    return sum(
        len(bucket) / total * abs(
            sum(score for _label, score in bucket) / len(bucket)
            - sum(label for label, _score in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
    )


def main() -> None:
    metrics = per_user_metrics(ROWS)
    print("per-user NDCG@2 / Recall@2 / MAP@2")
    for user_id, values in metrics.items():
        print(user_id, *(f"{value:.4f}" for value in values))
    print("macro:", *(f"{sum(values[i] for values in metrics.values()) / len(metrics):.4f}" for i in range(3)))
    print(f"ECE (4 fixed bins): {ece(ROWS):.4f}")
    rescaled = [(user, video, label, score * 0.5) for user, video, label, score in ROWS]
    print(f"ECE after score rescaling (same order): {ece(rescaled):.4f}")


if __name__ == "__main__":
    main()
