"""Unit tests for training helpers in main.py.

Run from the repo root inside env_tf:
    python -m unittest discover -s tests
"""

import unittest

from main import mean_val_auc


class MeanValAUCTest(unittest.TestCase):
    def test_mean_of_available_tasks(self):
        logs = {
            "val_output_1_auc": 0.70,
            "val_output_2_auc": 0.80,
            "val_output_3_auc": float("nan"),
        }
        self.assertAlmostEqual(mean_val_auc(logs, 4), 0.75)

    def test_returns_none_when_no_metric(self):
        self.assertIsNone(mean_val_auc({}, 4))
        self.assertIsNone(mean_val_auc({"val_output_1_auc": float("nan")}, 4))

    def test_ignores_non_numeric(self):
        logs = {"val_output_1_auc": 0.6, "val_output_2_auc": None}
        self.assertAlmostEqual(mean_val_auc(logs, 2), 0.6)


if __name__ == "__main__":
    unittest.main()
