"""Turn a tiny exposure table into grouped user candidates."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    path = Path(__file__).with_name("sample_exposures.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["user_id"]].append(row)

    for user_id, rows in grouped.items():
        labels = [int(row["label"]) for row in rows]
        print(f"{user_id}: {len(rows)} exposures, positives={sum(labels)}, videos={[row['video_id'] for row in rows]}")
        print("  candidate set for within-exposure ranking")


if __name__ == "__main__":
    main()
