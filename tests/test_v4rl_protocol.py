"""V4RL: части разбиения не делят кадры ground truth; позитивы обучения — центр интервала."""

import unittest

from tgseqloc.data.v4rl import separate_by_ground_truth
from tgseqloc.training.trainer import central_positives


class SeparateByGroundTruthTests(unittest.TestCase):
    def test_boundary_and_revisit_queries_are_dropped(self) -> None:
        split = {
            "positives": {0: [0, 1, 2], 1: [2, 3, 4], 2: [4, 5], 3: [6, 7], 4: [7, 8], 5: [0, 1], 6: [20, 21]},
            "train_query_indices": [0, 1],
            "validation_query_indices": [2, 3],
            "test_query_indices": [4, 5, 6],
        }
        out = separate_by_ground_truth(split)
        self.assertEqual(out["validation_query_indices"], [3])        # 2 делит кадр 4 с обучением
        self.assertEqual(out["excluded_validation_indices"], [2])
        self.assertEqual(out["test_query_indices"], [6])              # 4 делит 7 с валидацией, 5 — повторный проход
        self.assertEqual(out["excluded_test_indices"], [4, 5])
        self.assertEqual(split["test_query_indices"], [4, 5, 6])      # вход не меняется

    def test_an_empty_test_is_an_error(self) -> None:
        with self.assertRaises(RuntimeError):
            separate_by_ground_truth({"positives": {0: [1], 1: [1]}, "train_query_indices": [0],
                                      "validation_query_indices": [], "test_query_indices": [1]})


class CentralPositivesTests(unittest.TestCase):
    def test_central_half_of_the_interval(self) -> None:
        out = central_positives({0: [10, 11, 12, 13, 14, 15, 16, 17], 1: [5]}, [0, 1, 2], 0.5)
        self.assertEqual(out[0], [12, 13, 14, 15])
        self.assertEqual(out[1], [5])
        self.assertNotIn(2, out)

    def test_at_least_one_frame(self) -> None:
        self.assertEqual(central_positives({0: [1, 2, 3]}, [0], 0.1)[0], [2])
