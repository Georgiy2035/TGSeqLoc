"""Text annotation: recognizer outputs, junk strings and dynamic scores."""

from __future__ import annotations

import importlib.util
import unittest

from tgseqloc.annotation.ocr import parse_word_predictions, predictions_to_frame_text
from tgseqloc.data.formats import (
    FrameMasks,
    FrameText,
    TextDetection,
    frame_text_from_json,
    frame_text_to_json,
    is_junk_text,
)

HAS_PYCOCOTOOLS = importlib.util.find_spec("pycocotools") is not None
WIDTH, HEIGHT = 100, 100


class PredictionsTests(unittest.TestCase):
    def test_pixel_box_becomes_a_normalized_box(self) -> None:
        frame = predictions_to_frame_text(
            [{"bbox": [64, 32, 128, 32], "text": " shop ", "confidence": 0.9}], 640, 320
        )
        self.assertEqual(frame.detections[0].box, (0.1, 0.1, 0.3, 0.2))
        self.assertEqual(frame.texts, ["shop"])
        self.assertEqual(frame.image_size, (640, 320))

    def test_polygon_uses_its_extent(self) -> None:
        polygon = [[64, 48], [128, 32], [192, 48], [128, 64]]
        frame = predictions_to_frame_text([{"bbox": polygon, "text": "shop"}], 640, 320)
        self.assertEqual(frame.detections[0].box, (0.1, 0.1, 0.3, 0.2))

    def test_missing_confidence_is_stored_as_one(self) -> None:
        """Vision-language models report no confidence."""

        frame = predictions_to_frame_text([{"bbox": [0, 0, 10, 10], "text": "cafe", "confidence": None}], 100, 100)
        self.assertEqual(frame.detections[0].confidence, 1.0)

    def test_empty_and_no_text_readings_are_dropped(self) -> None:
        rows = [{"bbox": [0, 0, 10, 10], "text": value} for value in ("", "   ", "No Text", "<no_text>")]
        self.assertEqual(len(predictions_to_frame_text(rows, 100, 100)), 0)

    def test_box_outside_the_frame_is_dropped(self) -> None:
        frame = predictions_to_frame_text([{"bbox": [-50, 10, 40, 30], "text": "shop"}], 100, 100)
        self.assertEqual(len(frame), 0)

    def test_invalid_frame_size_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            predictions_to_frame_text([], 0, 100)


class WordSpottingTests(unittest.TestCase):
    def test_boxes_are_converted_from_the_0_1000_scale(self) -> None:
        predictions = parse_word_predictions('[{"bbox": [100, 100, 300, 200], "text": "shop"}]', 1000, 500)
        self.assertEqual(predictions, [{"bbox": [100.0, 50.0, 200.0, 50.0], "text": "shop", "confidence": None}])

    def test_markdown_fence_and_trailing_comma_are_tolerated(self) -> None:
        raw = '```json\n[{"bbox": [0, 0, 500, 500], "text": "cafe"},]\n```'
        self.assertEqual([p["text"] for p in parse_word_predictions(raw, 100, 100)], ["cafe"])

    def test_box_given_as_a_string_is_decoded(self) -> None:
        raw = '[{"bbox": "<box>0 0 500 500</box>", "text": "cafe"}]'
        self.assertEqual(parse_word_predictions(raw, 100, 100)[0]["bbox"], [0.0, 0.0, 50.0, 50.0])

    def test_empty_array_means_no_text(self) -> None:
        self.assertEqual(parse_word_predictions("[]", 100, 100), [])

    def test_answers_that_are_not_word_arrays_are_errors(self) -> None:
        for raw, message in (
            ("there is no text", "did not return a JSON array"),
            ("[] but also this", "accompanied by other output"),
            ('[{"bbox": [0, 0, 1200, 10], "text": "x"}]', "outside 0-1000"),
            ('[{"bbox": [0, 0], "text": "x"}]', "no valid bbox and text"),
        ):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, message):
                    parse_word_predictions(raw, 100, 100)


class JunkTextTests(unittest.TestCase):
    def test_short_and_digit_only_strings_are_junk(self) -> None:
        for value in ("", "ab", " a1 ", "2024", "0000"):
            with self.subTest(value=value):
                self.assertTrue(is_junk_text(value))

    def test_words_and_mixed_strings_are_kept(self) -> None:
        for value in ("abc", "cafe", "12a4", "24/7"):
            with self.subTest(value=value):
                self.assertFalse(is_junk_text(value))

    def test_digit_rule_can_be_disabled(self) -> None:
        self.assertFalse(is_junk_text("2024", drop_numeric=False))


class JsonTests(unittest.TestCase):
    def test_round_trip_keeps_dynamic_scores(self) -> None:
        frame = FrameText(
            detections=(TextDetection(box=(0.1, 0.2, 0.3, 0.4), text="shop", confidence=1.0, dynamic_score=0.25),),
            image_size=(640, 480),
            model_identity={"ocr": "qwen3_vl_4b"},
        )
        payload = frame_text_to_json(frame, frame_id="000001")
        self.assertEqual(payload["frame_id"], "000001")
        self.assertEqual(frame_text_from_json(payload), frame)


def _text(*boxes: tuple[float, float, float, float]) -> FrameText:
    return FrameText(
        detections=tuple(TextDetection(box=box, text=f"t{i}", confidence=1.0) for i, box in enumerate(boxes)),
        image_size=(WIDTH, HEIGHT),
    )


def _masks(*rectangles: tuple[int, int, int, int]) -> FrameMasks:
    """Union mask of pixel rectangles ``(y1, y2, x1, x2)``."""

    import numpy as np

    from tgseqloc.annotation.segmentation import encode_mask

    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for y1, y2, x1, x2 in rectangles:
        mask[y1:y2, x1:x2] = True
    return FrameMasks(dynamic_rle=encode_mask(mask), image_size=(WIDTH, HEIGHT))


@unittest.skipUnless(HAS_PYCOCOTOOLS, "pycocotools is required for masks")
class DynamicScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        from tgseqloc.annotation.dynamics import dynamic_scores

        self.score = dynamic_scores

    def test_box_inside_a_mask_scores_one_and_outside_zero(self) -> None:
        frame = _text((0.2, 0.2, 0.3, 0.3), (0.6, 0.6, 0.7, 0.7))
        self.assertEqual(self.score(frame, _masks((10, 50, 10, 50))), [1.0, 0.0])

    def test_half_covered_box_scores_one_half(self) -> None:
        self.assertEqual(self.score(_text((0.2, 0.2, 0.4, 0.4)), _masks((0, 100, 0, 30))), [0.5])

    def test_small_text_on_a_large_object_scores_one(self) -> None:
        """Intersection over the box area: the IoU of the two would be tiny."""

        self.assertEqual(self.score(_text((0.48, 0.48, 0.52, 0.52)), _masks((0, 100, 0, 100))), [1.0])

    def test_empty_mask_means_nothing_is_dynamic(self) -> None:
        empty = FrameMasks(dynamic_rle=None, image_size=(WIDTH, HEIGHT))
        self.assertEqual(self.score(_text((0.2, 0.2, 0.3, 0.3)), empty), [0.0])

    def test_frame_without_text_scores_nothing(self) -> None:
        self.assertEqual(self.score(FrameText(image_size=(WIDTH, HEIGHT)), _masks((0, 10, 0, 10))), [])

    def test_masks_of_another_frame_are_refused(self) -> None:
        other = FrameMasks(dynamic_rle=None, image_size=(64, 64))
        with self.assertRaisesRegex(ValueError, "different frames"):
            self.score(_text((0.2, 0.2, 0.3, 0.3)), other)


if __name__ == "__main__":
    unittest.main()
