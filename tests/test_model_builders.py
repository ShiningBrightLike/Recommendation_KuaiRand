"""Shape/parameter tests for the ranking model builders.

Run from the repo root inside env_tf:
    python -m unittest discover -s tests
"""

import unittest

import numpy as np

from MMoE_model import (
    build_logistic_model,
    build_mmoe_model,
    build_shared_bottom_model,
    build_single_task_model,
)

CAT_COLS = ["cat_a", "cat_b"]
NUM_COLS = ["num_a", "num_b"]
VOCAB = 20
BATCH = 4


def make_inputs():
    rng = np.random.default_rng(0)
    cat = [rng.integers(0, VOCAB, size=BATCH).astype("int32") for _ in CAT_COLS]
    num = rng.normal(size=(BATCH, len(NUM_COLS))).astype("float32")
    return cat + [num]


class BuilderShapeTest(unittest.TestCase):
    def _assert_multi_output(self, model, expected_outputs):
        self.assertEqual(len(model.outputs), expected_outputs)
        preds = model.predict(make_inputs(), verbose=0)
        if expected_outputs == 1:
            preds = [preds]
        self.assertEqual(len(preds), expected_outputs)
        for pred in preds:
            self.assertEqual(pred.shape, (BATCH, 1))
            self.assertTrue(np.all(pred >= 0) and np.all(pred <= 1))

    def test_mmoe_outputs(self):
        model = build_mmoe_model(CAT_COLS, NUM_COLS, VOCAB, num_tasks=4)
        self._assert_multi_output(model, 4)

    def test_shared_bottom_outputs(self):
        model = build_shared_bottom_model(CAT_COLS, NUM_COLS, VOCAB, num_tasks=4)
        self._assert_multi_output(model, 4)

    def test_single_task_output(self):
        model = build_single_task_model(CAT_COLS, NUM_COLS, VOCAB)
        self._assert_multi_output(model, 1)

    def test_logistic_outputs_and_params(self):
        model = build_logistic_model(CAT_COLS, NUM_COLS, VOCAB, num_tasks=4)
        self._assert_multi_output(model, 4)
        # Embedding(VOCAB, 1) + 4 tasks x (4 weights + 1 bias)
        self.assertEqual(model.count_params(), VOCAB + 4 * (len(CAT_COLS) + len(NUM_COLS) + 1))


if __name__ == "__main__":
    unittest.main()
