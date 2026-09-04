"""Canonical frame formats: validation and the dynamics attribute."""

from __future__ import annotations

import unittest

from tgseqloc.data.formats import FrameMasks, FrameText, TextDetection


def detection(box=(0.1, 0.1, 0.4, 0.4), text="shop", confidence=0.9, **kwargs):
    return TextDetection(box=box, text=text, confidence=confidence, **kwargs)


class TextDetectionTests(unittest.TestCase):
    def test_box_must_be_normalized_and_non_degenerate(self) -> None:
        for box, expected in (
            ((0.1, 0.1, 1.4, 0.4), "normalized"),
            ((-0.1, 0.1, 0.4, 0.4), "normalized"),
            ((0.4, 0.1, 0.4, 0.4), "non-degenerate"),
            ((0.1, 0.4, 0.4, 0.4), "non-degenerate"),
            ((0.1, 0.1, 0.4), "four values"),
        ):
            with self.subTest(box=box):
                with self.assertRaisesRegex(ValueError, expected):
                    detection(box=box)

    def test_dynamic_score_defaults_to_unknown_and_is_bounded(self) -> None:
        self.assertIsNone(detection().dynamic_score)
        self.assertEqual(detection(dynamic_score=0.25).dynamic_score, 0.25)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            detection(dynamic_score=1.5)

    def test_area_is_normalized(self) -> None:
        self.assertAlmostEqual(detection(box=(0.0, 0.0, 0.5, 0.5)).area, 0.25)


class FrameTextTests(unittest.TestCase):
    def test_empty_frame_is_valid(self) -> None:
        frame = FrameText(image_size=(640, 480))
        self.assertEqual(len(frame), 0)
        self.assertEqual(frame.boxes, [])
        self.assertEqual(frame.texts, [])

    def test_image_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "image_size must be positive"):
            FrameText(image_size=(0, 480))

    def test_boxes_and_texts_stay_aligned(self) -> None:
        frame = FrameText(
            detections=(detection(text="a"), detection(box=(0.5, 0.5, 0.6, 0.6), text="b")),
            image_size=(640, 480),
        )
        self.assertEqual(frame.texts, ["a", "b"])
        self.assertEqual(len(frame.boxes), len(frame.texts))

    def test_select_keeps_flagged_detections_only(self) -> None:
        frame = FrameText(
            detections=(detection(text="keep"), detection(box=(0.5, 0.5, 0.6, 0.6), text="drop")),
            image_size=(640, 480),
        )
        self.assertEqual(frame.select([True, False]).texts, ["keep"])
        with self.assertRaisesRegex(ValueError, "1 flags for 2 detections"):
            frame.select([True])

    def test_with_dynamic_scores_requires_one_score_per_detection(self) -> None:
        frame = FrameText(detections=(detection(),), image_size=(640, 480))
        self.assertEqual(frame.with_dynamic_scores([0.75]).detections[0].dynamic_score, 0.75)
        with self.assertRaisesRegex(ValueError, "2 scores for 1 detections"):
            frame.with_dynamic_scores([0.1, 0.2])

    def test_originals_are_not_mutated(self) -> None:
        frame = FrameText(detections=(detection(),), image_size=(640, 480))
        frame.with_dynamic_scores([0.5])
        self.assertIsNone(frame.detections[0].dynamic_score)


class FrameMasksTests(unittest.TestCase):
    def test_absent_mask_differs_from_stage_not_run(self) -> None:
        masks = FrameMasks(image_size=(640, 480))
        self.assertTrue(masks.is_empty)
        self.assertFalse(FrameMasks(dynamic_rle={"size": [480, 640]}, image_size=(640, 480)).is_empty)

    def test_image_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "image_size must be positive"):
            FrameMasks(image_size=(640, -1))


if __name__ == "__main__":
    unittest.main()
