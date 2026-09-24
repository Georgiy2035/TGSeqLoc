"""V4RL: части разбиения не делят кадры ground truth; позитивы обучения — центр интервала."""

import unittest

from tgseqloc.data.v4rl import build_block_split, separate_by_ground_truth
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


class BlockSplitTests(unittest.TestCase):
    @staticmethod
    def route(count: int, width: int = 3) -> dict:
        """A straight walk: query i sees reference frames i..i+width-1."""
        return {i: list(range(i, i + width)) for i in range(count)}

    def test_parts_share_no_frame_and_the_test_stays_whole(self) -> None:
        positives = self.route(100)
        out = build_block_split(100, positives, test_ratio=0.2, validation_ratio=0.2, validation_centres=[0.3, 0.7])
        self.assertEqual(out["test_query_indices"], list(range(80, 100)))
        self.assertEqual(out["validation_blocks"], [[20, 27], [52, 59]])
        frames = lambda part: {f for i in out[part] for f in positives[i]}
        self.assertFalse(frames("train_query_indices") & frames("validation_query_indices"))
        self.assertFalse(frames("train_query_indices") & frames("test_query_indices"))
        self.assertFalse(frames("validation_query_indices") & frames("test_query_indices"))
        # training pays for the separation: the queries next to each block are gone
        self.assertIn(18, out["excluded_train_indices"])
        self.assertIn(28, out["excluded_train_indices"])
        self.assertIn(78, out["excluded_train_indices"])
        self.assertEqual(out["excluded_test_indices"], [])

    def test_validation_touching_the_test_is_dropped(self) -> None:
        out = build_block_split(100, self.route(100), test_ratio=0.2, validation_ratio=0.1, validation_centres=[0.95])
        self.assertEqual(out["excluded_validation_indices"], [78, 79])
        self.assertEqual(out["validation_query_indices"], [72, 73, 74, 75, 76, 77])

    def test_overlapping_blocks_are_an_error(self) -> None:
        with self.assertRaises(ValueError):
            build_block_split(100, self.route(100), test_ratio=0.2, validation_ratio=0.5, validation_centres=[0.4, 0.5])


class CentralPositivesTests(unittest.TestCase):
    def test_central_half_of_the_interval(self) -> None:
        out = central_positives({0: [10, 11, 12, 13, 14, 15, 16, 17], 1: [5]}, [0, 1, 2], 0.5)
        self.assertEqual(out[0], [12, 13, 14, 15])
        self.assertEqual(out[1], [5])
        self.assertNotIn(2, out)

    def test_at_least_one_frame(self) -> None:
        self.assertEqual(central_positives({0: [1, 2, 3]}, [0], 0.1)[0], [2])
