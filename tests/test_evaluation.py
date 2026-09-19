"""Tests for user-level exposure-set ranking metrics."""

import unittest

import numpy as np

from evaluation import (
    ece,
    evaluate_task,
    gauc,
    map_at_k,
    ndcg_at_k,
    recall_at_k,
)


class RankingMetricsTest(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([1, 0, 0, 1, 0], dtype=float)
        self.scores = np.array([0.9, 0.1, 0.8, 0.2, 0.3], dtype=float)
        self.groups = np.array(["u1", "u1", "u2", "u2", "u3"])
        self.row_ids = np.arange(len(self.labels))

    def test_gauc_ignores_groups_without_both_classes(self):
        # u1 is perfectly ordered, u2 is inversely ordered, u3 has no positive.
        self.assertAlmostEqual(
            gauc(self.labels, self.scores, self.groups), 0.5, places=7
        )

    def test_user_ranking_metrics_have_explicit_no_positive_policy(self):
        # u3 has no positive and contributes zero to NDCG/MAP, while recall is
        # averaged only over u1/u2, the users with at least one positive.
        self.assertAlmostEqual(
            ndcg_at_k(self.labels, self.scores, self.groups, self.row_ids, k=1),
            (1.0 + 0.0 + 0.0) / 3.0,
        )
        self.assertAlmostEqual(
            recall_at_k(self.labels, self.scores, self.groups, self.row_ids, k=1),
            0.5,
        )
        self.assertAlmostEqual(
            map_at_k(self.labels, self.scores, self.groups, self.row_ids, k=1),
            (1.0 + 0.0 + 0.0) / 3.0,
        )

    def test_ece(self):
        labels = np.array([1, 0, 1, 0], dtype=float)
        probabilities = np.array([0.9, 0.1, 0.6, 0.4], dtype=float)
        self.assertAlmostEqual(ece(labels, probabilities, n_bins=2), 0.25)

    def test_evaluate_task_bootstrap_is_reproducible(self):
        first = evaluate_task(
            self.labels,
            self.scores,
            self.groups,
            self.row_ids,
            k=1,
            bootstrap=20,
            seed=7,
        )
        second = evaluate_task(
            self.labels,
            self.scores,
            self.groups,
            self.row_ids,
            k=1,
            bootstrap=20,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["users"], 3)
        self.assertEqual(first["gauc_users"], 2)
        self.assertIn("ece", first["bootstrap"])
        self.assertEqual(first["bootstrap"]["ece"]["groups"], 3)
        self.assertLessEqual(
            first["bootstrap"]["ece"]["lower"], first["ece"]
        )
        self.assertGreaterEqual(
            first["bootstrap"]["ece"]["upper"], first["ece"]
        )


if __name__ == "__main__":
    unittest.main()
