"""Коррекция распознанного текста перед кодированием."""

from __future__ import annotations

import unittest

from tgseqloc.preparation.spelling import (
    SymSpell,
    _deletes,
    edit_distance,
    from_corpus,
)


class DistanceTests(unittest.TestCase):
    def test_counts_substitutions_insertions_and_deletions(self) -> None:
        self.assertEqual(edit_distance("boucherie", "boucherie"), 0)
        self.assertEqual(edit_distance("boucherie", "bouchebie"), 1)   # замена
        self.assertEqual(edit_distance("boucherie", "boucheri"), 1)    # удаление
        self.assertEqual(edit_distance("boucherie", "boucheriex"), 1)  # вставка
        self.assertEqual(edit_distance("abc", ""), 3)

    def test_adjacent_transposition_costs_one(self) -> None:
        """OCR swaps neighbours often; charging two edits would hide the fix."""

        self.assertEqual(edit_distance("confiserie", "confiserei"), 1)

    def test_deletion_set_grows_with_distance(self) -> None:
        self.assertIn("bc", _deletes("abc", 1))
        self.assertIn("a", _deletes("abc", 2))
        self.assertNotIn("a", _deletes("abc", 1))


class CorrectionTests(unittest.TestCase):
    def speller(self, **params) -> SymSpell:
        return SymSpell(**params).build(
            [("boucherie", 10), ("confiserie", 8), ("parking", 5), ("peclard", 3)]
        )

    def test_a_known_word_is_left_alone(self) -> None:
        self.assertEqual(self.speller().correct("BOUCHERIE"), "BOUCHERIE")

    def test_one_wrong_character_is_repaired(self) -> None:
        self.assertEqual(self.speller().correct("BOUCHEBIE"), "BOUCHERIE")

    def test_case_pattern_is_preserved(self) -> None:
        speller = self.speller()
        self.assertEqual(speller.correct("bouchebie"), "boucherie")
        self.assertEqual(speller.correct("Bouchebie"), "Boucherie")
        self.assertEqual(speller.correct("BOUCHEBIE"), "BOUCHERIE")

    def test_a_word_beyond_the_radius_is_untouched(self) -> None:
        self.assertEqual(self.speller().correct("XYZWVUT"), "XYZWVUT")

    def test_short_strings_are_not_corrected(self) -> None:
        """Below four characters nearly every entry is in range; that is guessing."""

        speller = self.speller(min_length=4)
        self.assertEqual(speller.correct("PAR"), "PAR")

    def test_ties_are_broken_by_frequency_not_iteration_order(self) -> None:
        speller = SymSpell().build([("cafe", 100), ("cafo", 1)])
        self.assertEqual(speller.correct("cafx"), "cafe")

    def test_correct_all_preserves_order_and_length(self) -> None:
        speller = self.speller()
        result = speller.correct_all(["BOUCHEBIE", "unknown", "PARKING"])
        self.assertEqual(result, ["BOUCHERIE", "unknown", "PARKING"])


class CorpusDictionaryTests(unittest.TestCase):
    def test_proper_nouns_survive_because_the_corpus_defines_them(self) -> None:
        """A general dictionary would correct PECLARD away; the corpus keeps it."""

        speller = from_corpus(["PECLARD"] * 5 + ["PECLARD"] * 2 + ["BOUCHERIE"] * 3)
        self.assertEqual(speller.correct("PECLARD"), "PECLARD")
        self.assertEqual(speller.correct("PEGLARD"), "PECLARD")

    def test_rare_readings_stay_out_of_the_vocabulary(self) -> None:
        speller = from_corpus(["cafe"] * 5 + ["cxfe"], min_count=2)
        self.assertEqual(speller.correct("cxfe"), "cafe")

    def test_identity_changes_with_the_vocabulary(self) -> None:
        first = from_corpus(["cafe"] * 3).cache_identity
        second = from_corpus(["cafe"] * 3 + ["parking"] * 3).cache_identity
        self.assertNotEqual(first, second)

    def test_identity_is_stable_for_the_same_vocabulary(self) -> None:
        self.assertEqual(
            from_corpus(["cafe"] * 3).cache_identity,
            from_corpus(["cafe"] * 3).cache_identity,
        )


if __name__ == "__main__":
    unittest.main()
