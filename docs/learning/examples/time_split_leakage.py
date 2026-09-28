"""Show why a statistic computed over the whole period can leak validation data."""

from __future__ import annotations


EVENTS = [
    ("2022-04-15", 0),
    ("2022-04-15", 1),
    ("2022-04-16", 1),
    ("2022-04-16", 1),
]


def mean(values: list[int]) -> float:
    return sum(values) / len(values)


def main() -> None:
    train = [label for date, label in EVENTS if date <= "2022-04-15"]
    validation = [label for date, label in EVENTS if date >= "2022-04-16"]
    whole_period = [label for _date, label in EVENTS]
    print(f"train-only statistic: {mean(train):.3f}")
    print(f"whole-period statistic: {mean(whole_period):.3f}")
    print(f"validation outcome mean: {mean(validation):.3f}")
    print("The whole-period value uses validation outcomes and must not be used as a train feature.")


if __name__ == "__main__":
    main()
