"""Cached model stages: reuse, resumability and isolation between them."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tgseqloc.data.formats import FrameMasks, FrameText, TextDetection
from tgseqloc.inference.segmentation import encode_mask
from tgseqloc.stages import (
    frame_masks_from_json,
    frame_masks_to_json,
    frame_text_from_json,
    frame_text_to_json,
    read_stage,
    run_stage,
    stage_status,
)


class FakeOCR:
    """Counts how often it actually ran, which is what the cache should cut."""

    def __init__(self, identity: str = "v1") -> None:
        self.identity = identity
        self.loads = 0
        self.calls = 0

    @property
    def cache_identity(self) -> dict:
        return {"backend": "fake_ocr", "identity": self.identity}

    def load(self) -> None:
        self.loads += 1

    def predict_one(self, image_path) -> FrameText:
        self.calls += 1
        return FrameText(
            detections=(TextDetection(box=(0.1, 0.1, 0.4, 0.4), text=Path(image_path).stem,
                                      confidence=0.9),),
            image_size=(64, 32),
            model_identity=self.cache_identity,
        )


class FailingOCR(FakeOCR):
    def predict_one(self, image_path):
        raise RuntimeError("model exploded")


class CodecTests(unittest.TestCase):
    def test_frame_text_survives_a_round_trip(self) -> None:
        frame = FrameText(
            detections=(
                TextDetection(box=(0.1, 0.1, 0.4, 0.4), text="CAFE", confidence=0.8),
                TextDetection(box=(0.5, 0.5, 0.6, 0.6), text="BUS", confidence=0.7,
                              dynamic_score=0.9),
            ),
            image_size=(640, 480),
            model_identity={"backend": "paddleocr_v5"},
        )
        restored = frame_text_from_json(frame_text_to_json(frame))
        self.assertEqual(restored.texts, frame.texts)
        self.assertEqual(restored.boxes, frame.boxes)
        self.assertEqual(restored.image_size, frame.image_size)
        self.assertEqual(restored.detections[1].dynamic_score, 0.9)
        self.assertIsNone(restored.detections[0].dynamic_score)

    def test_frame_masks_survive_a_round_trip(self) -> None:
        mask = np.zeros((32, 64), dtype=bool)
        mask[4:20, 8:40] = True
        masks = FrameMasks(dynamic_rle=encode_mask(mask), image_size=(64, 32),
                           model_identity={"backend": "yolo_seg"})
        restored = frame_masks_from_json(frame_masks_to_json(masks))
        self.assertEqual(restored.image_size, (64, 32))
        self.assertEqual(restored.dynamic_rle, masks.dynamic_rle)

    def test_empty_mask_stays_empty_through_json(self) -> None:
        masks = FrameMasks(dynamic_rle=None, image_size=(64, 32))
        self.assertTrue(frame_masks_from_json(frame_masks_to_json(masks)).is_empty)


class RunStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "stages" / "ocr"
        self.frames = [
            ("seq1", f"{index:06d}", Path(self.temporary.name) / f"{index:06d}.png")
            for index in range(4)
        ]

    def test_first_run_computes_every_frame(self) -> None:
        model = FakeOCR()
        result = run_stage("ocr", model, self.frames, self.root)
        self.assertEqual((result.written, result.reused, result.failed), (4, 0, 0))
        self.assertTrue(result.complete)
        self.assertEqual(model.calls, 4)

    def test_second_run_reuses_everything_and_never_loads_the_model(self) -> None:
        """A fully cached stage must not pay for loading a model at all."""

        run_stage("ocr", FakeOCR(), self.frames, self.root)
        model = FakeOCR()
        result = run_stage("ocr", model, self.frames, self.root)
        self.assertEqual((result.written, result.reused), (0, 4))
        self.assertEqual(model.calls, 0)
        self.assertEqual(model.loads, 0)

    def test_interrupted_run_only_covers_what_is_missing(self) -> None:
        run_stage("ocr", FakeOCR(), self.frames[:2], self.root)
        model = FakeOCR()
        result = run_stage("ocr", model, self.frames, self.root)
        self.assertEqual((result.written, result.reused), (2, 2))
        self.assertEqual(model.calls, 2)

    def test_changing_the_model_invalidates_the_cache(self) -> None:
        run_stage("ocr", FakeOCR(identity="v1"), self.frames, self.root)
        result = run_stage("ocr", FakeOCR(identity="v2"), self.frames, self.root)
        self.assertEqual((result.written, result.reused), (4, 0))

    def test_rebuild_ignores_a_valid_cache(self) -> None:
        run_stage("ocr", FakeOCR(), self.frames, self.root)
        result = run_stage("ocr", FakeOCR(), self.frames, self.root, rebuild=True)
        self.assertEqual((result.written, result.reused), (4, 0))

    def test_one_failing_frame_does_not_end_the_run(self) -> None:
        result = run_stage("ocr", FailingOCR(), self.frames, self.root)
        self.assertEqual(result.failed, 4)
        self.assertFalse(result.complete)

    def test_a_failed_frame_reports_its_error_when_read(self) -> None:
        run_stage("ocr", FailingOCR(), self.frames[:1], self.root)
        with self.assertRaisesRegex(RuntimeError, "model exploded"):
            read_stage("ocr", self.root, "seq1", "000000")

    def test_reading_an_absent_artifact_says_which_stage_to_run(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "tgseqloc stage ocr"):
            read_stage("ocr", self.root, "seq1", "999999")

    def test_written_artifact_reads_back_as_the_same_value(self) -> None:
        run_stage("ocr", FakeOCR(), self.frames, self.root)
        frame = read_stage("ocr", self.root, "seq1", "000002")
        self.assertEqual(frame.texts, ["000002"])
        self.assertEqual(frame.image_size, (64, 32))

    def test_status_counts_without_reading_artifacts(self) -> None:
        run_stage("ocr", FakeOCR(), self.frames[:3], self.root)
        self.assertEqual(
            stage_status(self.root, self.frames), {"present": 3, "missing": 1}
        )

    def test_unknown_stage_is_named(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown stage 'nonsense'"):
            run_stage("nonsense", FakeOCR(), self.frames, self.root)


if __name__ == "__main__":
    unittest.main()
