"""Тройки внутри одной камеры: майнинг и случайные негативы не выходят за камеру запроса."""

import unittest

import numpy as np

from tgseqloc.training.mining import mine_hard_negatives, random_negatives
from tgseqloc.training.trainer import frame_group, mine_within_groups


class SameCameraTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.db = rng.standard_normal((40, 8)).astype(np.float32)
        self.db_groups = ["", "mono_left"] * 20
        self.q = rng.standard_normal((6, 8)).astype(np.float32)
        self.q_indices = [0, 1, 2, 3, 4, 5]
        self.q_groups = {0: "", 1: "mono_left", 2: "", 3: "mono_left", 4: "", 5: "mono_left"}
        self.positives = {i: [2 * i, 2 * i + 1] for i in self.q_indices}

    def test_mined_negatives_stay_in_the_query_camera(self) -> None:
        for miner in (mine_hard_negatives, random_negatives):
            mined = mine_within_groups(miner, self.db, self.q, self.q_indices, self.positives, 3, 16, 7,
                                       database_groups=self.db_groups, query_groups=self.q_groups)
            for query, items in mined.items():
                self.assertEqual(len(items), 3, miner.__name__)
                self.assertTrue(all(self.db_groups[i] == self.q_groups[query] for i in items), miner.__name__)
                self.assertFalse(set(items) & set(self.positives[query]), miner.__name__)

    def test_one_group_is_the_plain_miner(self) -> None:
        groups = [""] * 40
        mined = mine_within_groups(mine_hard_negatives, self.db, self.q, self.q_indices, self.positives, 3, 16, 7,
                                   database_groups=groups, query_groups={i: "" for i in self.q_indices})
        plain = mine_hard_negatives(self.db, self.q, self.q_indices, self.positives, 3, 16, 7)
        self.assertEqual(mined, plain)

    def test_exclusions_are_translated(self) -> None:
        excl = {i: [j for j in range(40) if j % 4 == 1] for i in self.q_indices}
        mined = mine_within_groups(mine_hard_negatives, self.db, self.q, self.q_indices, self.positives, 3, 40, 7,
                                   database_groups=self.db_groups, query_groups=self.q_groups, exclusions=excl)
        for query, items in mined.items():
            self.assertFalse(set(items) & set(excl[query]))

    def test_frame_group(self) -> None:
        self.assertEqual(frame_group("base/1439388279982387.pt"), "")
        self.assertEqual(frame_group("query/mono_rear-1441186653310472.pt"), "mono_rear")
