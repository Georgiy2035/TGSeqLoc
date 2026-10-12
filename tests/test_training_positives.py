"""Позитивы для обучения ближе радиуса разметки; промежуточные кадры не участвуют вовсе."""

import unittest
from pathlib import Path

from tgseqloc.training.trainer import TripletGraphDataset, training_positives

SPLIT = {
    "query_positions": {"0": [0.0, 0.0], "1": [1000.0, 0.0]},
    "database_positions": {str(i): [float(i * 5), 0.0] for i in range(12)},   # 0, 5, …, 55 м
    "positives": {"0": [0, 1, 2, 3, 4, 5]},                                   # до 25 м
}


class TrainingPositivesTests(unittest.TestCase):
    def test_only_close_ground_truth_positives_are_kept(self) -> None:
        self.assertEqual(training_positives(SPLIT, [0, 1], 10.0), {0: [0, 1, 2]})

    def test_positions_are_required(self) -> None:
        with self.assertRaises(ValueError):
            training_positives({"positives": {}}, [0], 10.0)


def dataset(**extra):
    ds = TripletGraphDataset([Path(f"db{i}") for i in range(12)], [Path("q0")], [0],
                             {0: [0, 1, 2]}, 20, **extra)
    ds._load = lambda path: path
    return ds


class IgnoredFramesTests(unittest.TestCase):
    def test_ignored_frames_are_never_negatives(self) -> None:
        item = dataset(ignored={0: [0, 1, 2, 3, 4, 5]})[0]
        negatives = {int(p.name[2:]) for p in item[2:]}
        self.assertTrue(negatives.isdisjoint({0, 1, 2, 3, 4, 5}))
        self.assertIn(int(item[1].name[2:]), {0, 1, 2})

    def test_without_ignored_frames_nothing_changes(self) -> None:
        self.assertEqual(dataset()[0], dataset(ignored=None)[0])

    def test_hard_negatives_inside_the_ignored_band_are_dropped(self) -> None:
        ds = dataset(ignored={0: [3, 4, 5]})
        ds.set_hard_negatives({0: [4, 9]})
        negatives = {int(p.name[2:]) for p in ds[0][2:]}
        self.assertEqual(negatives, {9})
