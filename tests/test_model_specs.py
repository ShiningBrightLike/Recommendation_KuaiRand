"""Tests for the `structure+encoder` specs the comparison script accepts.

Run from the repo root inside env_tf:
    python -m unittest discover -s tests
"""

import unittest

from baselines import evaluation_split_for, format_model_spec, parse_model_spec
from models import available_encoders, available_structures, structure_hyperparams
from models.__main__ import build_all_models


class ParseModelSpecTest(unittest.TestCase):
    def test_bare_structure_uses_the_default_encoder(self):
        self.assertEqual(parse_model_spec("mmoe"), ("mmoe", "mlp"))

    def test_structure_with_encoder(self):
        self.assertEqual(parse_model_spec("shared_bottom+mlp"), ("shared_bottom", "mlp"))

    def test_surrounding_whitespace_is_ignored(self):
        self.assertEqual(parse_model_spec(" mmoe + mlp "), ("mmoe", "mlp"))

    def test_logistic_keeps_no_encoder(self):
        self.assertEqual(parse_model_spec("logistic"), ("logistic", None))

    def test_logistic_rejects_an_encoder(self):
        with self.assertRaises(ValueError):
            parse_model_spec("logistic+mlp")

    def test_unknown_structure_and_encoder_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_model_spec("no_such_structure")
        with self.assertRaises(ValueError):
            parse_model_spec("mmoe+no_such_encoder")

    def test_labels_always_name_the_encoder(self):
        # A bare structure name resolves to the default encoder, and the label
        # keeps it, so a run never hides which encoder produced it.
        self.assertEqual(format_model_spec(parse_model_spec("mmoe")), "mmoe+mlp")
        self.assertEqual(
            format_model_spec(parse_model_spec("shared_bottom+mlp")), "shared_bottom+mlp"
        )
        self.assertEqual(format_model_spec(parse_model_spec("logistic")), "logistic")

    def test_every_registered_encoder_can_be_compared(self):
        for encoder in available_encoders():
            with self.subTest(encoder=encoder):
                self.assertEqual(parse_model_spec(f"mmoe+{encoder}"), ("mmoe", encoder))


class StructureParamsCoverageTest(unittest.TestCase):
    """Every structure owns effective defaults; callers do not maintain a second map."""

    def test_every_structure_exposes_effective_defaults(self):
        for name in available_structures():
            with self.subTest(structure=name):
                params = structure_hyperparams(name, num_tasks=4)
                self.assertEqual(params["num_tasks"], 4)
                self.assertGreater(len(params), 1)


class EvaluationDisciplineTest(unittest.TestCase):
    def test_comparisons_use_the_feature_decision_set(self):
        self.assertEqual(
            evaluation_split_for([("mmoe", "mlp"), ("mmoe", "dcn")]),
            "val",
        )

    def test_final_confirmation_requires_one_locked_model_spec(self):
        self.assertEqual(
            evaluation_split_for([("mmoe", "mlp")], final_test=True),
            "test",
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluation_split_for(
                [("mmoe", "mlp"), ("mmoe", "dcn")], final_test=True
            )
        with self.assertRaisesRegex(ValueError, "exactly one seed"):
            evaluation_split_for(
                [("mmoe", "mlp")], final_test=True, num_seeds=3
            )

    def test_existing_test_metrics_cannot_enter_a_validation_comparison(self):
        with self.assertRaisesRegex(ValueError, "final confirmation"):
            evaluation_split_for([], has_existing_test_run=True)


class ModuleSelfCheckTest(unittest.TestCase):
    def test_self_check_includes_baselines_and_all_axis_combinations(self):
        models = build_all_models()
        expected = {"logistic"}
        expected.update(f"single_task+{encoder}" for encoder in available_encoders())
        expected.update(
            f"{structure}+{encoder}"
            for structure in available_structures()
            for encoder in available_encoders()
        )
        self.assertEqual(set(models), expected)


if __name__ == "__main__":
    unittest.main()
