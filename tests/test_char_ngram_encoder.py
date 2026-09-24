"""Character n-gram encoder: shape, determinism and behaviour under OCR noise."""

from __future__ import annotations

import unittest

import torch

from tgseqloc.preparation.text import (
    CharNGramEncoder,
    build_char_ngram_encoder,
    validate_embeddings,
)


def similarity(encoder: CharNGramEncoder, left: str, right: str) -> float:
    vectors = encoder.encode([left, right])
    return float(vectors[0] @ vectors[1])


class ShapeTests(unittest.TestCase):
    def test_encoding_is_normalized_and_correctly_shaped(self) -> None:
        encoder = CharNGramEncoder(embedding_dim=64)
        vectors = encoder.encode(["BOUCHERIE", "CONFISERIE"])
        self.assertEqual(vectors.shape, (2, 64))
        self.assertEqual(vectors.dtype, torch.float32)
        for norm in vectors.norm(dim=1).tolist():
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_empty_input_keeps_the_declared_width(self) -> None:
        encoder = CharNGramEncoder(embedding_dim=32)
        self.assertEqual(encoder.encode([]).shape, (0, 32))

    def test_string_too_short_for_any_ngram_encodes_to_zeros(self) -> None:
        """Normalizing such a row would divide by nothing, so it stays zero."""

        encoder = CharNGramEncoder(embedding_dim=16, min_n=5, max_n=6)
        vectors = encoder.encode(["ab"])
        self.assertEqual(vectors.shape, (1, 16))
        self.assertEqual(float(vectors.norm()), 0.0)

    def test_output_satisfies_the_shared_encoder_contract(self) -> None:
        encoder = CharNGramEncoder(embedding_dim=32)
        texts = ["STARBUCKS", "", "Meteg"]
        validate_embeddings(encoder, texts, encoder.encode(texts))

    def test_rejects_impossible_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "embedding_dim must be positive"):
            CharNGramEncoder(embedding_dim=0)
        with self.assertRaisesRegex(ValueError, "min_n <= max_n"):
            CharNGramEncoder(min_n=5, max_n=3)


class NoiseTests(unittest.TestCase):
    """The reason this encoder exists: OCR on these datasets runs at 30-50% CER."""

    def setUp(self) -> None:
        self.encoder = CharNGramEncoder(embedding_dim=1024)

    def test_a_single_wrong_character_keeps_most_of_the_similarity(self) -> None:
        self.assertAlmostEqual(similarity(self.encoder, "BOUCHERIE", "BOUCHERIE"), 1.0, places=5)
        self.assertGreater(similarity(self.encoder, "BOUCHERIE", "BOUCHEBIE"), 0.5)

    def test_an_error_near_an_edge_costs_less_than_one_in_the_middle(self) -> None:
        """Fewer n-grams span a character at the boundary, so fewer are lost."""

        middle = similarity(self.encoder, "BOUCHERIE", "BOUCHEBIE")
        edge = similarity(self.encoder, "BOUCHERIE", "BOUCHERIF")
        self.assertGreater(edge, middle)

    def test_similarity_falls_monotonically_as_corruption_grows(self) -> None:
        readings = ["PECLARD", "PECLABD", "PEGLABD", "PEGLA8D", "P3GLA8D"]
        scores = [similarity(self.encoder, "PECLARD", value) for value in readings]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_longer_strings_survive_the_same_number_of_errors_better(self) -> None:
        """Worth knowing for the datasets: short signs are the hard case.

        Two errors leave a thirteen-character string well above a seven-
        character one, because a smaller share of its n-grams is touched.
        """

        short = similarity(self.encoder, "PECLARD", "PEGLABD")
        long = similarity(self.encoder, "YVES COIFFURE", "YVE5 C0IFFURE")
        self.assertGreater(long, short)

    def test_a_truncated_reading_still_matches_its_source(self) -> None:
        self.assertGreater(similarity(self.encoder, "CONFISERIE", "CONFISER"), 0.5)

    def test_unrelated_strings_share_nothing(self) -> None:
        self.assertLess(similarity(self.encoder, "BOUCHERIE", "PARKING"), 0.2)

    def test_case_is_ignored_by_default(self) -> None:
        self.assertAlmostEqual(similarity(self.encoder, "Starbucks", "STARBUCKS"), 1.0, places=5)
        cased = CharNGramEncoder(embedding_dim=1024, lowercase=False)
        self.assertLess(similarity(cased, "Starbucks", "STARBUCKS"), 1.0)


class DeterminismTests(unittest.TestCase):
    def test_bucketing_does_not_depend_on_the_process(self) -> None:
        """Python's hash() is salted per process, which would silently change
        embeddings between runs and invalidate every cached graph."""

        import subprocess
        import sys

        script = (
            "import sys; sys.path.insert(0, 'src');"
            "from tgseqloc.preparation.text import CharNGramEncoder;"
            "print(CharNGramEncoder(embedding_dim=32)._features('BOUCHERIE'))"
        )
        outputs = {
            subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, check=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            ).stdout
            for seed in ("0", "1", "12345")
        }
        self.assertEqual(len(outputs), 1)

    def test_identity_tracks_every_setting_that_changes_output(self) -> None:
        base = CharNGramEncoder().cache_identity
        self.assertNotEqual(base, CharNGramEncoder(embedding_dim=128).cache_identity)
        self.assertNotEqual(base, CharNGramEncoder(min_n=2).cache_identity)
        self.assertNotEqual(base, CharNGramEncoder(lowercase=False).cache_identity)


class FactoryTests(unittest.TestCase):
    def test_unknown_params_are_named(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown char_ngram params: nonsense"):
            build_char_ngram_encoder(nonsense=1)

    def test_both_encoders_are_registered(self) -> None:
        from tgseqloc.components import register_builtin_components

        available = register_builtin_components().available("encoder")
        self.assertIn("char_ngram", available)
        self.assertIn("multilingual_e5", available)


if __name__ == "__main__":
    unittest.main()
