"""Tests for multi-seed aggregation in the comparison script.

Run from the repo root inside env_tf:
    python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from pathlib import Path

from baselines import aggregate_seeds, parse_seeds, write_seed_metrics

TASKS = ["is_click", "is_like", "is_follow", "is_comment"]


def seed_result(click, like, follow, comment):
    return {"tasks": dict(zip(TASKS, (click, like, follow, comment)))}


class ParseSeedsTest(unittest.TestCase):
    def test_comma_separated_seeds(self):
        self.assertEqual(parse_seeds("2025, 2026,2027"), [2025, 2026, 2027])

    def test_empty_means_no_seed_list(self):
        self.assertIsNone(parse_seeds(None))
        self.assertIsNone(parse_seeds("  "))


class AggregateSeedsTest(unittest.TestCase):
    """Mean and sample standard deviation (ddof=1) across seed runs."""

    def setUp(self):
        self.runs = [
            seed_result(0.80, 0.90, 0.70, 0.60),
            seed_result(0.90, 0.90, 0.70, 0.60),
            seed_result(1.00, 0.90, 0.70, 0.60),
        ]

    def test_per_task_mean_and_std(self):
        aggregated = aggregate_seeds(self.runs, TASKS)

        self.assertAlmostEqual(aggregated["tasks_mean"]["is_click"], 0.90)
        self.assertAlmostEqual(aggregated["tasks_std"]["is_click"], 0.10)
        self.assertAlmostEqual(aggregated["tasks_mean"]["is_like"], 0.90)
        self.assertAlmostEqual(aggregated["tasks_std"]["is_like"], 0.0)

    def test_summary_averages_seeds_then_tasks(self):
        aggregated = aggregate_seeds(self.runs, TASKS)
        summary = aggregated["summary"]

        # four-task means per seed: 0.75, 0.775, 0.80
        self.assertAlmostEqual(summary["mean_all_tasks"], 0.775)
        self.assertAlmostEqual(summary["std_all_tasks"], 0.025)
        # gate-task means per seed (click, like): 0.85, 0.90, 0.95
        self.assertAlmostEqual(summary["mean_gate_tasks"], 0.90)
        self.assertAlmostEqual(summary["std_gate_tasks"], 0.05)

    def test_a_single_seed_reports_zero_spread(self):
        aggregated = aggregate_seeds(self.runs[:1], TASKS)

        self.assertAlmostEqual(aggregated["tasks_mean"]["is_click"], 0.80)
        self.assertAlmostEqual(aggregated["tasks_std"]["is_click"], 0.0)
        self.assertAlmostEqual(aggregated["summary"]["std_all_tasks"], 0.0)


class RunArtifactTest(unittest.TestCase):
    def test_each_seed_gets_an_independent_traceable_metrics_file(self):
        result = {
            "tasks": dict.fromkeys(TASKS, 0.75),
            "model": {
                "encoder": {"name": "mlp", "hyperparams": {"hidden": [64]}},
                "structure": {"name": "mmoe", "hyperparams": {"num_tasks": 4}},
                "total_params": 123,
            },
            "params": 123,
            "artifacts": {"model": "model.keras"},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "mmoe+mlp" / "seed-2025"
            write_seed_metrics(
                run_dir,
                result,
                {"evaluation_split": "val", "epochs": 1},
                "mmoe+mlp",
                2025,
            )
            payload = json.loads(
                (run_dir / "metrics.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["config"]["evaluation_split"], "val")
        self.assertEqual(payload["config"]["model_spec"], "mmoe+mlp")
        self.assertEqual(payload["config"]["seed"], 2025)
        self.assertEqual(payload["model"]["structure"]["name"], "mmoe")


if __name__ == "__main__":
    unittest.main()
