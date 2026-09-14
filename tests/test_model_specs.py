"""Tests for the `structure+encoder` specs the comparison script accepts.

Run from the repo root inside env_tf:
    python -m unittest discover -s tests
"""

import unittest

import config as C
from baselines import format_model_spec, parse_model_spec
from models import available_encoders, available_structures


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
    """Every structure must have hyper-parameters in config, or runs break late."""

    def test_every_structure_has_config_params(self):
        missing = [name for name in available_structures() if name not in C.STRUCTURE_PARAMS]
        self.assertEqual(missing, [], f"structures missing config.STRUCTURE_PARAMS: {missing}")


if __name__ == "__main__":
    unittest.main()
