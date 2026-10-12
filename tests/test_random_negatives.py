"""Случайные негативы: не позитивы, не исключённые кадры, воспроизводимы по сиду."""

import unittest

import numpy as np

from tgseqloc.training.mining import random_negatives


class RandomNegativesTests(unittest.TestCase):
    def test_skips_positives_and_exclusions(self) -> None:
        db = np.zeros((20, 4)); q = np.zeros((2, 4))
        out = random_negatives(db, q, [0, 1], {0: [0, 1, 2], 1: [5]}, 6, seed=3, exclusions={0: [3, 4]})
        self.assertEqual(len(out[0]), 6)
        self.assertFalse(set(out[0]) & {0, 1, 2, 3, 4})
        self.assertNotIn(5, out[1])

    def test_same_seed_same_choice(self) -> None:
        db = np.zeros((50, 4)); q = np.zeros((1, 4))
        self.assertEqual(random_negatives(db, q, [0], {0: [1]}, 4, seed=7), random_negatives(db, q, [0], {0: [1]}, 4, seed=7))

    def test_everything_positive_gives_empty(self) -> None:
        self.assertEqual(random_negatives(np.zeros((2, 4)), np.zeros((1, 4)), [0], {0: [0, 1]}, 3), {0: []})
