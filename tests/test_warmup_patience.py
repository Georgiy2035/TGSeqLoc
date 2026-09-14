"""Ранняя остановка не считает эпохи прогрева."""

import unittest

from tgseqloc.training.trainer import epochs_without_improvement


def stop_epoch(improvements, patience, warmup):
    count = 0
    for epoch, improved in enumerate(improvements, start=1):
        count = epochs_without_improvement(count, improved, epoch, warmup)
        if not improved and count >= patience:
            return epoch
    return None


class WarmupPatienceTests(unittest.TestCase):
    def test_without_warmup_it_is_the_usual_counter(self) -> None:
        self.assertEqual(stop_epoch([True, False, False, False], patience=3, warmup=0), 4)

    def test_warmup_epochs_do_not_count(self) -> None:
        run = [True] + [False] * 12
        self.assertEqual(stop_epoch(run, patience=3, warmup=10), 13)

    def test_improvement_resets(self) -> None:
        self.assertEqual(stop_epoch([True, False, True, False, False], patience=2, warmup=0), 5)
