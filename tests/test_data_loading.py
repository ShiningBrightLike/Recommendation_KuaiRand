"""Unit tests for shared data-loading helpers.

Run from the repo root inside env_tf:
    python -m unittest discover -s tests
"""

import unittest

import pandas as pd

import config as C
from data_loading import filter_features, load_split_ids, parse_name_list, video_statistic_cols
from data_process import _build_id_sidecar


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


class EvaluationSidecarTest(unittest.TestCase):
    def test_build_id_sidecar_preserves_raw_identifiers_and_order(self):
        frame = pd.DataFrame(
            {
                "user_id": [8, 8, 3],
                "video_id": [10, 11, 12],
                "date": [20220408, 20220408, 20220409],
            }
        )
        sidecar = _build_id_sidecar(frame)
        self.assertEqual(sidecar.columns.tolist(), ["row_id", "user_id", "video_id", "date"])
        self.assertEqual(sidecar["row_id"].tolist(), [0, 1, 2])
        self.assertEqual(sidecar["user_id"].tolist(), [8, 8, 3])
        self.assertEqual(sidecar["date"].tolist(), ["20220408", "20220408", "20220409"])

    @unittest.skipUnless(
        (C.PROCESSED_DIR / C.ID_SPLIT_FILES["val"]).exists(),
        "processed validation sidecar is not available in this checkout",
    )
    def test_processed_sidecars_are_available_and_contiguous(self):
        sidecar = load_split_ids("val", max_rows=3)
        self.assertEqual(sidecar["row_id"].tolist(), [0, 1, 2])
        self.assertEqual(
            sidecar.columns.tolist(), ["row_id", "user_id", "video_id", "date"]
        )


if __name__ == "__main__":
    unittest.main()
