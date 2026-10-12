"""Буферная зона: кадры чуть дальше радиуса позитивов не должны становиться негативами."""

import unittest

import numpy as np

from tgseqloc.training.mining import mine_hard_negatives
from tgseqloc.training.trainer import negative_exclusions


def embeddings():
    # запрос ближе всего к кадру 0, затем 1, затем 2, затем 3
    database = np.array([[1.0, 0.0], [0.95, 0.31], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    return database, query


class MinerTests(unittest.TestCase):
    def test_excluded_frames_are_skipped_for_the_next_hardest(self) -> None:
        database, query = embeddings()
        mined = mine_hard_negatives(database, query, [7], {7: [0]}, 1, exclusions={7: [1]})
        self.assertEqual(mined[7], [2])

    def test_without_exclusions_the_nearest_non_positive_is_taken(self) -> None:
        database, query = embeddings()
        self.assertEqual(mine_hard_negatives(database, query, [7], {7: [0]}, 1)[7], [1])

    def test_no_exclusions_argument_changes_nothing(self) -> None:
        database, query = embeddings()
        self.assertEqual(mine_hard_negatives(database, query, [7], {7: [0]}, 2, seed=3),
                         mine_hard_negatives(database, query, [7], {7: [0]}, 2, seed=3, exclusions=None))


class ExclusionTests(unittest.TestCase):
    SPLIT = {
        "query_positions": {"0": [0.0, 0.0], "1": [500.0, 0.0]},
        "database_positions": {str(i): [float(i * 10), 0.0] for i in range(60)},
    }

    def test_everything_within_the_buffer_is_listed(self) -> None:
        excluded = negative_exclusions(self.SPLIT, [0], 50.0)
        self.assertEqual(excluded[0], [0, 1, 2, 3, 4, 5])
        self.assertNotIn(1, excluded)

    def test_positions_are_required(self) -> None:
        with self.assertRaises(ValueError):
            negative_exclusions({}, [0], 50.0)
