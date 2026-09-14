"""Shape/parameter tests for the ranking model builders.

Run from the repo root inside env_tf:
    python -m unittest discover -s tests
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from models import (
    available_encoders,
    available_models,
    build_model,
    build_logistic_model,
    build_single_task_model,
    create_model,
    custom_objects,
)

CAT_COLS = ["cat_a", "cat_b"]
NUM_COLS = ["num_a", "num_b"]
VOCAB = 20
BATCH = 4
REPO_ROOT = Path(__file__).resolve().parents[1]

# Runs in a fresh interpreter: reloads the saved run and re-predicts, so the
# test covers what a new process (a scoring job, feature importance, ...) does.
RELOAD_IN_CHILD = """
import sys
import numpy as np
import tensorflow as tf
from models import custom_objects

model_path, inputs_path, output_path = sys.argv[1:4]
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects())
with np.load(inputs_path) as data:
    inputs = [data[f"arr_{i}"] for i in range(int(data["n"]))]
predictions = model.predict(inputs, verbose=0)
np.savez(output_path, *[np.asarray(pred) for pred in predictions])
"""


def make_inputs():
    rng = np.random.default_rng(0)
    cat = [rng.integers(0, VOCAB, size=BATCH).astype("int32") for _ in CAT_COLS]
    num = rng.normal(size=(BATCH, len(NUM_COLS))).astype("float32")
    return cat + [num]


def build_small_mmoe_model(num_tasks=2):
    """A tiny MMoE model at the default encoder, for shape/serialization tests."""
    model, _ = build_model(
        encoder="mlp",
        structure="mmoe",
        categorical_cols=CAT_COLS,
        numeric_cols=NUM_COLS,
        cat_vocab_size=VOCAB,
        num_tasks=num_tasks,
        num_experts=2,
        units=8,
        tower_units=4,
    )
    return model


class BuilderShapeTest(unittest.TestCase):
    def _assert_multi_output(self, model, expected_outputs):
        assert_multi_output(self, model, expected_outputs)

    def test_mmoe_outputs(self):
        model, _ = build_model(
            encoder="mlp",
            structure="mmoe",
            categorical_cols=CAT_COLS,
            numeric_cols=NUM_COLS,
            cat_vocab_size=VOCAB,
            num_tasks=4,
            num_experts=2,
            units=8,
            tower_units=4,
        )
        self._assert_multi_output(model, 4)

    def test_shared_bottom_outputs(self):
        model, _ = build_model(
            encoder="mlp",
            structure="shared_bottom",
            categorical_cols=CAT_COLS,
            numeric_cols=NUM_COLS,
            cat_vocab_size=VOCAB,
            num_tasks=4,
            bottom_units=8,
            tower_units=4,
        )
        self._assert_multi_output(model, 4)

    def test_single_task_output(self):
        model = build_single_task_model(CAT_COLS, NUM_COLS, VOCAB)
        self._assert_multi_output(model, 1)

    def test_logistic_outputs_and_params(self):
        model = build_logistic_model(CAT_COLS, NUM_COLS, VOCAB, num_tasks=4)
        self._assert_multi_output(model, 4)
        # Embedding(VOCAB, 1) + 4 tasks x (4 weights + 1 bias)
        self.assertEqual(model.count_params(), VOCAB + 4 * (len(CAT_COLS) + len(NUM_COLS) + 1))


REGISTRY_KWARGS = {
    "mmoe": {"num_tasks": 4, "num_experts": 2, "units": 8, "tower_units": 4},
    "shared_bottom": {"num_tasks": 4, "bottom_units": 8, "tower_units": 4},
    "single_task": {"units": 8, "tower_units": 4},
    "logistic": {"num_tasks": 4},
}
REGISTRY_OUTPUTS = {"mmoe": 4, "shared_bottom": 4, "single_task": 1, "logistic": 4}


def assert_multi_output(test_case, model, expected_outputs):
    """Shared shape/probability assertions for a built ranking model."""
    test_case.assertEqual(len(model.outputs), expected_outputs)
    preds = model.predict(make_inputs(), verbose=0)
    if expected_outputs == 1:
        preds = [preds]
    test_case.assertEqual(len(preds), expected_outputs)
    for pred in preds:
        test_case.assertEqual(pred.shape, (BATCH, 1))
        test_case.assertTrue(np.all(pred >= 0) and np.all(pred <= 1))


class FeatureEncoderAxisTest(unittest.TestCase):
    """The encoder axis is selectable and carries its own run metadata."""

    def test_default_encoder_builds_the_mmoe_model_with_run_metadata(self):
        model, axes = build_model(
            encoder="mlp",
            structure="mmoe",
            categorical_cols=CAT_COLS,
            numeric_cols=NUM_COLS,
            cat_vocab_size=VOCAB,
            num_tasks=4,
            num_experts=2,
            units=8,
            tower_units=4,
        )

        assert_multi_output(self, model, 4)
        self.assertEqual(axes["encoder"]["name"], "mlp")
        self.assertEqual(axes["encoder"]["hyperparams"]["hidden"], [64])
        self.assertEqual(axes["encoder"]["output_dim"], 64)
        # Dense(64) over (2 categorical x 8 embedding dims + 2 numeric) = 18 dims
        self.assertEqual(axes["encoder"]["params"], 18 * 64 + 64)
        self.assertEqual(axes["structure"]["name"], "mmoe")
        self.assertEqual(axes["structure"]["hyperparams"]["num_experts"], 2)
        self.assertEqual(axes["total_params"], model.count_params())

    def test_dcn_encoder_concatenates_low_rank_cross_and_deep(self):
        """DCN-v2: `num_cross_layers` low-rank crosses in parallel with a deep branch.

        Input width is 2 categorical x 8 embedding dims + 2 numeric = 18.
        Parameters: cross layers 2 x (2 x 18 x 64 for U,V + 18 for the bias)
        = 4644, deep branch 18 x 64 + 64 = 1216, so 5860 in total; the output is
        the cross output (18) concatenated with the deep branch (64).
        """
        model, axes = build_model(
            encoder="dcn",
            structure="mmoe",
            categorical_cols=CAT_COLS,
            numeric_cols=NUM_COLS,
            cat_vocab_size=VOCAB,
            num_tasks=4,
            num_experts=2,
            units=8,
            tower_units=4,
        )

        assert_multi_output(self, model, 4)
        self.assertEqual(axes["encoder"]["name"], "dcn")
        self.assertEqual(
            axes["encoder"]["hyperparams"],
            {"num_cross_layers": 2, "rank": 64, "deep_units": 64},
        )
        self.assertEqual(axes["encoder"]["output_dim"], 18 + 64)
        self.assertEqual(axes["encoder"]["params"], 5860)

    def test_encoder_overrides_replace_the_module_defaults(self):
        model, axes = build_model(
            encoder="dcn",
            structure="mmoe",
            categorical_cols=CAT_COLS,
            numeric_cols=NUM_COLS,
            cat_vocab_size=VOCAB,
            num_tasks=4,
            num_experts=2,
            units=8,
            tower_units=4,
            encoder_overrides={"num_cross_layers": 1, "rank": 8, "deep_units": 32},
        )

        assert_multi_output(self, model, 4)
        self.assertEqual(axes["encoder"]["hyperparams"]["num_cross_layers"], 1)
        self.assertEqual(axes["encoder"]["hyperparams"]["rank"], 8)
        # 1 x (2 x 18 x 8 + 18) cross layer + (18 x 32 + 32) deep branch
        self.assertEqual(axes["encoder"]["params"], 306 + 608)
        self.assertEqual(axes["encoder"]["output_dim"], 18 + 32)

    def test_senet_encoder_reweights_each_field(self):
        """SENet keeps the input width and gates each field, not each dimension.

        Four fields (2 categorical, 2 numeric) at reduction 2 give a bottleneck
        of 2 units: 2 x 4 x 2 excitation weights + 2 + 4 bias terms = 22.
        """
        model, axes = build_model(
            encoder="senet",
            structure="mmoe",
            categorical_cols=CAT_COLS,
            numeric_cols=NUM_COLS,
            cat_vocab_size=VOCAB,
            num_tasks=4,
            num_experts=2,
            units=8,
            tower_units=4,
        )

        assert_multi_output(self, model, 4)
        self.assertEqual(axes["encoder"]["name"], "senet")
        self.assertEqual(axes["encoder"]["hyperparams"], {"reduction": 2})
        self.assertEqual(axes["encoder"]["output_dim"], 18)
        self.assertEqual(axes["encoder"]["params"], 22)

    def test_senet_field_count_follows_the_schema(self):
        """One extra numeric feature means one more field, and a wider gate."""
        _, axes = build_model(
            encoder="senet",
            structure="shared_bottom",
            categorical_cols=CAT_COLS,
            numeric_cols=NUM_COLS + ["extra_num"],
            cat_vocab_size=VOCAB,
            num_tasks=2,
            bottom_units=8,
            tower_units=4,
        )

        self.assertEqual(axes["encoder"]["output_dim"], 2 * 8 + 3)
        # 5 fields at reduction 2 -> bottleneck 2: 2 x 5 x 2 + 2 + 5 = 27
        self.assertEqual(axes["encoder"]["params"], 27)


class RegistryTest(unittest.TestCase):
    def test_registry_lists_the_documented_models(self):
        self.assertEqual(
            available_models(), ["logistic", "mmoe", "shared_bottom", "single_task"]
        )

    def test_create_model_dispatches_by_name(self):
        for name, expected_outputs in REGISTRY_OUTPUTS.items():
            with self.subTest(model=name):
                model = create_model(
                    name,
                    categorical_cols=CAT_COLS,
                    numeric_cols=NUM_COLS,
                    cat_vocab_size=VOCAB,
                    **REGISTRY_KWARGS[name],
                )
                self.assertEqual(len(model.outputs), expected_outputs)

    def test_create_model_rejects_unknown_names(self):
        with self.assertRaises(ValueError):
            create_model(
                "no_such_model",
                categorical_cols=CAT_COLS,
                numeric_cols=NUM_COLS,
                cat_vocab_size=VOCAB,
            )


class SerializationTest(unittest.TestCase):
    """A saved run must reload through the package's own load entry point."""

    def assert_reloads_identically(self, model):
        inputs = make_inputs()
        expected = model.predict(inputs, verbose=0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.keras"
            model.save(model_path)
            restored = tf.keras.models.load_model(
                str(model_path), custom_objects=custom_objects()
            )

        actual = restored.predict(inputs, verbose=0)
        self.assertEqual(len(actual), len(expected))
        for restored_pred, original_pred in zip(actual, expected):
            np.testing.assert_allclose(restored_pred, original_pred, rtol=1e-6, atol=1e-6)

    def test_saved_model_reloads_and_predicts_identically(self):
        self.assert_reloads_identically(build_small_mmoe_model())

    def test_every_encoder_model_reloads_and_predicts_identically(self):
        for encoder in available_encoders():
            with self.subTest(encoder=encoder):
                model, _ = build_model(
                    encoder=encoder,
                    structure="mmoe",
                    categorical_cols=CAT_COLS,
                    numeric_cols=NUM_COLS,
                    cat_vocab_size=VOCAB,
                    num_tasks=2,
                    num_experts=2,
                    units=8,
                    tower_units=4,
                )
                self.assert_reloads_identically(model)

    def test_saved_model_reloads_and_predicts_in_a_fresh_process(self):
        model = build_small_mmoe_model()
        inputs = make_inputs()
        expected = model.predict(inputs, verbose=0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            model_path = tmp / "model.keras"
            inputs_path = tmp / "inputs.npz"
            output_path = tmp / "predictions.npz"
            model.save(model_path)
            np.savez(
                inputs_path,
                n=len(inputs),
                **{f"arr_{i}": np.asarray(arr) for i, arr in enumerate(inputs)},
            )

            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    RELOAD_IN_CHILD,
                    str(model_path),
                    str(inputs_path),
                    str(output_path),
                ],
                cwd=str(REPO_ROOT),
                check=True,
                capture_output=True,
            )

            with np.load(output_path) as data:
                actual = [data[f"arr_{i}"] for i in range(len(expected))]

        for restored_pred, original_pred in zip(actual, expected):
            np.testing.assert_allclose(restored_pred, original_pred, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
