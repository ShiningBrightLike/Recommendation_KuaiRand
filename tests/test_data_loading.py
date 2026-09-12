"""Unit tests for shared data-loading helpers.

Run from the repo root inside env_tf:
    python -m unittest discover -s tests
"""

import unittest

import config as C
from data_loading import filter_features, parse_name_list, video_statistic_cols


class ParseNameListTest(unittest.TestCase):
    def test_parses_and_strips(self):
        self.assertEqual(parse_name_list(" a, b ,,c "), ["a", "b", "c"])
        self.assertIsNone(parse_name_list(None))
        self.assertIsNone(parse_name_list("  ,,"))


class FilterFeaturesTest(unittest.TestCase):
    def test_removes_from_both_lists(self):
        cat, num, dropped = filter_features(["a", "b"], ["x", "y"], {"b", "x"})
        self.assertEqual(cat, ["a"])
        self.assertEqual(num, ["y"])
        self.assertEqual(dropped, ["b", "x"])

    def test_unknown_feature_raises(self):
        with self.assertRaises(ValueError):
            filter_features(["a"], ["x"], {"nope"})

    def test_removing_all_categorical_raises(self):
        with self.assertRaises(ValueError):
            filter_features(["a"], ["x"], {"a"})

    def test_removing_all_numeric_raises(self):
        with self.assertRaises(ValueError):
            filter_features(["a"], ["x"], {"x"})


class VideoStatisticColsTest(unittest.TestCase):
    def test_columns_exclude_video_id_and_belong_to_numeric(self):
        cols = video_statistic_cols()
        self.assertGreater(len(cols), 40)
        self.assertNotIn("video_id", cols)
        self.assertTrue(set(cols) <= set(C.NUMERIC_COLS))


if __name__ == "__main__":
    unittest.main()
