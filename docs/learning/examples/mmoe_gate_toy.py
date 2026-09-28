"""A tiny MMoE forward pass using only the Python standard library."""

from __future__ import annotations

import math


EXPERTS = {
    "expert_1": [1.0, 0.0],
    "expert_2": [0.0, 1.0],
    "expert_3": [0.5, 0.5],
}

TASK_LOGITS = {
    "click": [2.0, 0.0, 1.0],
    "like": [0.0, 2.0, 1.0],
}


def softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def mix(weights: list[float], vectors: list[list[float]]) -> list[float]:
    return [sum(weight * vector[index] for weight, vector in zip(weights, vectors)) for index in range(2)]


def main() -> None:
    names = list(EXPERTS)
    vectors = list(EXPERTS.values())
    for task, logits in TASK_LOGITS.items():
        weights = softmax(logits)
        representation = mix(weights, vectors)
        print(task)
        print("  experts:", ", ".join(names))
        print("  gate:", ", ".join(f"{value:.3f}" for value in weights))
        print("  mixed representation:", ", ".join(f"{value:.3f}" for value in representation))


if __name__ == "__main__":
    main()
