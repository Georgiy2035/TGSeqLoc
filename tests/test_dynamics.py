"""Dynamic-text scoring: IoA against the union of dynamic masks."""

from __future__ import annotations

import unittest

import numpy as np

from tgseqloc.data.formats import FrameMasks, FrameText, TextDetection
from tgseqloc.inference.dynamics import MaskIoA, build_mask_ioa
from tgseqloc.inference.segmentation import encode_mask

WIDTH, HEIGHT = 100, 100


def text(*boxes: tuple[float, float, float, float]) -> FrameText:
    return FrameText(
        detections=tuple(
            TextDetection(box=box, text=f"t{index}", confidence=0.9)
            for index, box in enumerate(boxes)
        ),
        image_size=(WIDTH, HEIGHT),
    )


def masks(*rectangles: tuple[int, int, int, int]) -> FrameMasks:
    """Build masks from pixel rectangles given as (y1, y2, x1, x2)."""

    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for y1, y2, x1, x2 in rectangles:
        mask[y1:y2, x1:x2] = True
    return FrameMasks(dynamic_rle=encode_mask(mask), image_size=(WIDTH, HEIGHT))


class ScoreTests(unittest.TestCase):
    def test_box_fully_inside_a_mask_scores_one(self) -> None:
        scores = MaskIoA().score(text((0.2, 0.2, 0.3, 0.3)), masks((10, 50, 10, 50)))
        self.assertEqual(scores, [1.0])

    def test_box_entirely_outside_scores_zero(self) -> None:
        scores = MaskIoA().score(text((0.6, 0.6, 0.7, 0.7)), masks((10, 50, 10, 50)))
        self.assertEqual(scores, [0.0])

    def test_half_covered_box_scores_one_half(self) -> None:
        # Box spans x 20..40, mask covers x 20..30 of it.
        scores = MaskIoA().score(text((0.2, 0.2, 0.4, 0.4)), masks((0, 100, 0, 30)))
        self.assertAlmostEqual(scores[0], 0.5, places=2)

    def test_small_text_on_a_large_object_scores_high(self) -> None:
        """The reason for IoA over IoU: their IoU here is negligible."""

        tiny = text((0.48, 0.48, 0.52, 0.52))
        covering = masks((0, 100, 0, 100))
        self.assertEqual(MaskIoA().score(tiny, covering), [1.0])

    def test_scores_follow_detection_order(self) -> None:
        frame = text((0.0, 0.0, 0.1, 0.1), (0.9, 0.9, 1.0, 1.0))
        scores = MaskIoA().score(frame, masks((0, 20, 0, 20)))
        self.assertEqual(scores, [1.0, 0.0])

    def test_empty_mask_means_nothing_is_dynamic(self) -> None:
        """A segmenter that ran and found nothing differs from one that did not."""

        frame = text((0.2, 0.2, 0.3, 0.3))
        empty = FrameMasks(dynamic_rle=None, image_size=(WIDTH, HEIGHT))
        self.assertEqual(MaskIoA().score(frame, empty), [0.0])

    def test_frame_without_text_scores_nothing(self) -> None:
        self.assertEqual(
            MaskIoA().score(FrameText(image_size=(WIDTH, HEIGHT)), masks((0, 10, 0, 10))),
            [],
        )

    def test_mismatched_frames_are_refused(self) -> None:
        other = FrameMasks(dynamic_rle=None, image_size=(64, 64))
        with self.assertRaisesRegex(ValueError, "different frames"):
            MaskIoA().score(text((0.2, 0.2, 0.3, 0.3)), other)


class ThresholdTests(unittest.TestCase):
    def test_threshold_decides_classification_not_the_score(self) -> None:
        frame = text((0.2, 0.2, 0.4, 0.4))
        half = masks((0, 100, 0, 30))
        self.assertEqual(MaskIoA(threshold=0.4).classify(frame, half), [True])
        self.assertEqual(MaskIoA(threshold=0.6).classify(frame, half), [False])

    def test_apply_records_scores_and_drops_dynamic_text(self) -> None:
        frame = text((0.0, 0.0, 0.1, 0.1), (0.9, 0.9, 1.0, 1.0))
        kept = MaskIoA(threshold=0.5).apply(frame, masks((0, 20, 0, 20)))
        self.assertEqual(kept.texts, ["t1"])
        self.assertEqual(kept.detections[0].dynamic_score, 0.0)

    def test_apply_keeps_everything_when_nothing_is_covered(self) -> None:
        frame = text((0.5, 0.5, 0.6, 0.6))
        kept = MaskIoA().apply(frame, FrameMasks(image_size=(WIDTH, HEIGHT)))
        self.assertEqual(len(kept), 1)

    def test_invalid_threshold_is_refused(self) -> None:
        for value in (-0.1, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                    MaskIoA(threshold=value)

    def test_identity_tracks_the_threshold(self) -> None:
        self.assertNotEqual(
            MaskIoA(threshold=0.3).cache_identity, MaskIoA(threshold=0.7).cache_identity
        )


class FactoryTests(unittest.TestCase):
    def test_unknown_params_are_named(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown mask_ioa params: nonsense"):
            build_mask_ioa(nonsense=1)

    def test_backend_is_registered(self) -> None:
        from tgseqloc.components import register_builtin_components

        self.assertIn(
            "mask_ioa", register_builtin_components().available("text_dynamics")
        )


if __name__ == "__main__":
    unittest.main()
