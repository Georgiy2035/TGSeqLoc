"""Reranking of text-graph candidates by image descriptors, and sequence pool fusion."""

from __future__ import annotations

import unittest

import numpy as np

from tgseqloc.evaluation.reranking import align_descriptors, rerank_sequence

# Five database frames on the unit circle; query descriptors pick one of them.
ANGLES = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
DATABASE = np.stack([np.cos(ANGLES), np.sin(ANGLES)], axis=1)


def query(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle)])


class RerankTests(unittest.TestCase):
    def test_candidates_are_ordered_by_image_similarity(self) -> None:
        ordered = rerank_sequence([[0, 1, 2]], [query(1.0)], DATABASE)
        self.assertEqual(ordered.tolist(), [[2, 1, 0]])

    def test_only_candidates_are_ranked(self) -> None:
        """The image descriptor never searches the database beyond the candidates."""

        ordered = rerank_sequence([[0, 1]], [query(2.0)], DATABASE)
        self.assertEqual(ordered.tolist(), [[1, 0]])

    def test_sequence_pool_brings_candidates_of_earlier_frames(self) -> None:
        candidates = [[4, 3], [0, 1]]
        queries = [query(2.0), query(1.9)]
        single = rerank_sequence(candidates, queries, DATABASE, frames=1)
        fused = rerank_sequence(candidates, queries, DATABASE, frames=2, depth=2)
        self.assertEqual(single[1].tolist(), [1, 0])
        self.assertEqual(fused[1].tolist(), [4, 3])

    def test_fusion_is_causal(self) -> None:
        candidates = [[0, 1], [2, 3], [4, 0]]
        queries = [query(0.0), query(1.0), query(2.0)]
        full = rerank_sequence(candidates, queries, DATABASE, frames=3, depth=3)
        prefix = rerank_sequence(candidates[:2], queries[:2], DATABASE, frames=3, depth=3)
        self.assertEqual(full[:2].tolist(), prefix.tolist())

    def test_short_pool_is_padded(self) -> None:
        ordered = rerank_sequence([[0, 1]], [query(0.0)], DATABASE, depth=4)
        self.assertEqual(ordered.tolist(), [[0, 1, -1, -1]])

    def test_misaligned_inputs_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            rerank_sequence([[0, 1]], [query(0.0), query(1.0)], DATABASE)
        with self.assertRaises(ValueError):
            rerank_sequence([[0, 1]], [query(0.0)], DATABASE, frames=0)


class AlignTests(unittest.TestCase):
    def test_rows_follow_the_prepared_paths(self) -> None:
        images = {"seq1": {"000000_1": np.ones(2), "000001_2": np.zeros(2)}}
        rows = align_descriptors(["seq1/000001_2.pt", "seq1/000000_1.pt"], images)
        self.assertEqual(rows.tolist(), [[0.0, 0.0], [1.0, 1.0]])

    def test_missing_frame_is_named(self) -> None:
        with self.assertRaisesRegex(KeyError, "seq2/000000_1"):
            align_descriptors(["seq2/000000_1.pt"], {"seq2": {}})


if __name__ == "__main__":
    unittest.main()
