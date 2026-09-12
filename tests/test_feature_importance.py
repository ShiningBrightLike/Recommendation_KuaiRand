"""Unit tests for the permutation feature-importance module.

Run from the repo root inside env_tf:
    python -m unittest discover -s tests
"""

import unittest

import numpy as np
from numpy.random import SeedSequence, default_rng
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from feature_importance import (
    DECISION_PASS,
    DECISION_REJECT,
    analyze,
    build_column_descriptors,
    decide,
    parse_candidate_cols,
    shuffled_inputs,
)


class ColumnDescriptorTest(unittest.TestCase):
    def test_descriptor_ordering(self):
        specs = build_column_descriptors(["cat_a", "cat_b"], ["num_a", "num_b"])
        self.assertEqual(len(specs), 4)
        self.assertEqual(specs[0].name, "cat_a")
        self.assertEqual(specs[0].kind, "categorical")
        self.assertEqual(specs[1].input_index, 1)
        # Numeric features live in the single numeric input that follows the
        # categorical inputs.
        self.assertEqual(specs[2].kind, "numeric")
        self.assertEqual(specs[2].input_index, 2)
        self.assertEqual(specs[2].column_index, 0)
        self.assertEqual(specs[3].column_index, 1)


class ShuffleTest(unittest.TestCase):
    def test_permutation_is_deterministic_and_non_mutating(self):
        rng = default_rng(SeedSequence([2025, 0, 0]))
        numeric = rng.normal(size=(8, 2)).astype("float32")
        original = numeric.copy()
        spec = build_column_descriptors([], ["num_a", "num_b"])[0]

        a = shuffled_inputs([numeric], spec, default_rng(SeedSequence([1, 2, 3])))
        b = shuffled_inputs([numeric], spec, default_rng(SeedSequence([1, 2, 3])))
        np.testing.assert_array_equal(a[0], b[0])
        self.assertFalse(np.array_equal(a[0][:, 0], original[:, 0]))
        np.testing.assert_array_equal(numeric, original)  # inputs untouched


class DecisionTest(unittest.TestCase):
    def test_three_way_decision(self):
        self.assertEqual(decide(0.003, 0.0005, cutoff=0.001), DECISION_PASS)
        self.assertEqual(decide(0.0012, 0.0006, cutoff=0.001), "待确认")
        self.assertEqual(decide(0.0008, 0.0002, cutoff=0.001), DECISION_REJECT)


class CandidateColsTest(unittest.TestCase):
    def test_parse_comma_separated_names(self):
        self.assertEqual(parse_candidate_cols("shadow_0,new_feat_a"), ["shadow_0", "new_feat_a"])
        self.assertEqual(parse_candidate_cols(" a , b ,, c "), ["a", "b", "c"])

    def test_parse_empty_returns_none(self):
        self.assertIsNone(parse_candidate_cols(None))
        self.assertIsNone(parse_candidate_cols("   "))
        self.assertIsNone(parse_candidate_cols(",,"))


class AnalyzeTest(unittest.TestCase):
    def test_important_feature_outranks_noise(self):
        tf.keras.utils.set_random_seed(7)
        n = 3000
        rng = default_rng(42)
        x = rng.normal(size=(n, 2)).astype("float32")
        x_copy = x.copy()

        inp = Input(shape=(2,), name="numeric_input", dtype="float32")
        out_click = Dense(1, activation="sigmoid", name="output_1", use_bias=False)(inp)
        out_like = Dense(1, activation="sigmoid", name="output_2", use_bias=False)(inp)
        model = Model(inputs=[inp], outputs=[out_click, out_like])
        # x0 drives task 1; x1 is irrelevant to both tasks.
        model.layers[1].set_weights([np.array([[5.0], [0.0]], dtype="float32")])
        model.layers[2].set_weights([np.array([[0.0], [0.0]], dtype="float32")])

        y_click = (x[:, 0] > 0).astype("float32").reshape(-1, 1)
        y_like = (rng.random(n) > 0.7).astype("float32").reshape(-1, 1)
        specs = build_column_descriptors([], ["x0", "x1"])

        result = analyze(
            model=model,
            inputs=[x],
            targets=[y_click, y_like],
            task_names=["is_click", "is_like"],
            specs=specs,
            numeric_cols=["x0", "x1"],
            gate_tasks=["is_click", "is_like"],
            repeats=3,
            batch_size=1024,
            cutoff=0.001,
            verbose=False,
        )
        rows = {r["feature"]: r for r in result["features"]}
        np.testing.assert_array_equal(x, x_copy)  # analysis must not mutate inputs
        self.assertGreater(rows["x0"]["overall_mean"], 0.05)
        self.assertEqual(rows["x0"]["decision"], DECISION_PASS)
        self.assertLess(rows["x1"]["overall_mean"], 0.001)
        self.assertEqual(rows["x1"]["decision"], DECISION_REJECT)


if __name__ == "__main__":
    unittest.main()
