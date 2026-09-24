"""Dynamic-object segmentation: class tables, masks and back-end selection."""

from __future__ import annotations

import importlib.util
import unittest

from tgseqloc.annotation.segmentation import (
    DYNAMIC_CLASSES,
    SEGMENTERS,
    YoloESeg,
    YoloSeg,
    build_segmenter,
    dynamic_indices,
)

HAS_PYCOCOTOOLS = importlib.util.find_spec("pycocotools") is not None


class ClassTests(unittest.TestCase):
    def test_thirteen_coco_classes_are_dynamic(self) -> None:
        self.assertEqual(len(set(DYNAMIC_CLASSES)), 13)

    def test_aliases_of_checkpoint_label_tables_are_recognized(self) -> None:
        names = {0: "person", 3: "motorbike", 5: "tree", 62: "tvmonitor", 67: "mobile phone"}
        self.assertEqual(dynamic_indices(names), {0, 3, 62, 67})


class BackendTests(unittest.TestCase):
    def test_paper_segmenters_and_their_weights(self) -> None:
        self.assertEqual(
            {name: (cls, weights) for name, (cls, weights) in SEGMENTERS.items()},
            {
                "yolo11x_seg": (YoloSeg, "yolo11x-seg.pt"),
                "yolo11m_seg": (YoloSeg, "yolo11m-seg.pt"),
                "yoloe_11l_seg": (YoloESeg, "yoloe-11l-seg.pt"),
            },
        )

    def test_segmenter_is_built_without_loading_weights(self) -> None:
        segmenter = build_segmenter("yolo11x_seg")
        self.assertEqual(segmenter.confidence, 0.25)
        self.assertIsNone(segmenter._model)
        with self.assertRaisesRegex(RuntimeError, "call load"):
            segmenter.predict(["frame.png"])

    def test_unknown_segmenter_is_named(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown segmenter 'sam'"):
            build_segmenter("sam")


@unittest.skipUnless(HAS_PYCOCOTOOLS, "pycocotools is required for masks")
class MaskTests(unittest.TestCase):
    def test_mask_survives_a_json_round_trip(self) -> None:
        import json

        import numpy as np

        from tgseqloc.annotation.segmentation import decode_mask, encode_mask

        mask = np.zeros((40, 60), dtype=bool)
        mask[5:20, 10:50] = True
        rle = json.loads(json.dumps(encode_mask(mask)))
        self.assertTrue((decode_mask(rle, (40, 60)) == mask).all())

    def test_empty_mask_is_stored_as_none(self) -> None:
        import numpy as np

        from tgseqloc.annotation.segmentation import decode_mask, encode_mask

        self.assertIsNone(encode_mask(np.zeros((4, 4), dtype=bool)))
        self.assertFalse(decode_mask(None, (4, 4)).any())


if __name__ == "__main__":
    unittest.main()
