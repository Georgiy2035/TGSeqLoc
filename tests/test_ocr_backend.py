"""PaddleOCR backend: normalization, filtering and identity.

The engine itself is not exercised here; what matters for correctness of the
graph is how a recognized polygon becomes a normalized detection, and that is
pure logic.
"""

from __future__ import annotations

import unittest

from tgseqloc.inference.ocr import PaddleOCRv5, _make_detection, build_paddleocr_v5

WIDTH, HEIGHT = 640, 320
QUAD = [[64.0, 32.0], [192.0, 32.0], [192.0, 64.0], [64.0, 64.0]]


def make(text="shop", score=0.9, polygon=None, threshold=0.0, noops=frozenset()):
    return _make_detection(
        text, score, polygon if polygon is not None else QUAD,
        WIDTH, HEIGHT, threshold=threshold, noops=set(noops),
    )


class DetectionTests(unittest.TestCase):
    def test_polygon_becomes_a_normalized_axis_aligned_box(self) -> None:
        detection = make()
        self.assertEqual(detection.box, (0.1, 0.1, 0.3, 0.2))
        self.assertEqual(detection.text, "shop")
        self.assertEqual(detection.confidence, 0.9)

    def test_rotated_polygon_uses_its_extent(self) -> None:
        rotated = [[64.0, 48.0], [128.0, 32.0], [192.0, 48.0], [128.0, 64.0]]
        self.assertEqual(make(polygon=rotated).box, (0.1, 0.1, 0.3, 0.2))

    def test_coordinates_outside_the_frame_are_clipped(self) -> None:
        overflowing = [[-20.0, -10.0], [700.0, -10.0], [700.0, 400.0], [-20.0, 400.0]]
        self.assertEqual(make(polygon=overflowing).box, (0.0, 0.0, 1.0, 1.0))

    def test_empty_and_noop_strings_are_dropped(self) -> None:
        self.assertIsNone(make(text="   "))
        self.assertIsNone(make(text="No Text", noops={"no text"}))

    def test_confidence_below_the_threshold_is_dropped(self) -> None:
        self.assertIsNone(make(score=0.2, threshold=0.5))
        self.assertIsNotNone(make(score=0.5, threshold=0.5))

    def test_polygon_that_collapses_after_clipping_is_dropped(self) -> None:
        """Clipping can leave a zero-area box, which carries no location."""

        offscreen = [[-50.0, 10.0], [-10.0, 10.0], [-10.0, 40.0], [-50.0, 40.0]]
        self.assertIsNone(make(polygon=offscreen))

    def test_degenerate_polygon_is_dropped(self) -> None:
        self.assertIsNone(make(polygon=[[10.0], [20.0]]))
        self.assertIsNone(make(polygon=[]))


class BackendTests(unittest.TestCase):
    def test_predict_before_load_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "call load"):
            PaddleOCRv5().predict(["frame.png"])

    def test_cache_identity_covers_settings_that_change_output(self) -> None:
        identity = PaddleOCRv5().cache_identity
        for key in (
            "backend",
            "text_detection_model",
            "text_recognition_model",
            "use_textline_orientation",
            "confidence_threshold",
        ):
            self.assertIn(key, identity)

    def test_cache_identity_changes_with_the_recognizer(self) -> None:
        default = PaddleOCRv5().cache_identity
        other = PaddleOCRv5(text_recognition_model="other_rec").cache_identity
        self.assertNotEqual(default, other)

    def test_identity_ignores_declaration_order_of_noop_texts(self) -> None:
        first = PaddleOCRv5(noop_texts=("a", "b")).cache_identity
        second = PaddleOCRv5(noop_texts=("b", "a")).cache_identity
        self.assertEqual(first, second)

    def test_factory_rejects_unknown_params_by_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown paddleocr_v5 params: nonsense"):
            build_paddleocr_v5(nonsense=1)

    def test_factory_declares_its_weight_requirement(self) -> None:
        self.assertEqual(
            build_paddleocr_v5.weight_requirements(), ("paddleocr_v5_eslav",)
        )

    def test_backend_is_registered_under_the_ocr_kind(self) -> None:
        from tgseqloc.components import register_builtin_components

        self.assertIn("paddleocr_v5", register_builtin_components().available("ocr"))


if __name__ == "__main__":
    unittest.main()
