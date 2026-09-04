"""Segmentation backend: dynamic-class mapping and mask encoding."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tgseqloc.inference.segmentation import (
    DYNAMIC_CLASSES,
    YoloSeg,
    _fit,
    build_yolo_seg,
    canonical,
    decode_mask,
    dynamic_indices,
    encode_mask,
    is_dynamic,
)


class DynamicClassTests(unittest.TestCase):
    def test_aliases_map_to_the_canonical_coco_name(self) -> None:
        """Checkpoints disagree on names; without this, classes vanish silently."""

        for alias, expected in (
            ("motorbike", "motorcycle"),
            ("tvmonitor", "tv"),
            ("monitor", "tv"),
            ("Screen", "tv"),
            ("  CAR  ", "car"),
        ):
            with self.subTest(alias=alias):
                self.assertEqual(canonical(alias), expected)

    def test_both_dynamic_groups_are_recognized(self) -> None:
        self.assertTrue(is_dynamic("person"))      # ephemeral
        self.assertTrue(is_dynamic("tv"))          # volatile content
        self.assertTrue(is_dynamic("motorbike"))   # via alias
        self.assertFalse(is_dynamic("building"))
        self.assertFalse(is_dynamic("traffic light"))

    def test_indices_follow_the_models_label_table(self) -> None:
        names = {0: "person", 1: "building", 2: "motorbike", 3: "sky"}
        self.assertEqual(dynamic_indices(names), {0, 2})

    def test_class_list_can_be_narrowed(self) -> None:
        names = {0: "person", 1: "car"}
        self.assertEqual(dynamic_indices(names, ("car",)), {1})


class MaskCodecTests(unittest.TestCase):
    def test_round_trip_preserves_the_mask(self) -> None:
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:20, 15:35] = True
        restored = decode_mask(encode_mask(mask), mask.shape)
        self.assertTrue(np.array_equal(mask, restored))

    def test_empty_mask_encodes_to_nothing(self) -> None:
        """Absent is distinct from present-and-empty for the stage that reads it."""

        self.assertIsNone(encode_mask(np.zeros((10, 10), dtype=bool)))
        self.assertFalse(decode_mask(None, (10, 10)).any())

    def test_encoded_payload_survives_json(self) -> None:
        import json

        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 2:5] = True
        payload = json.loads(json.dumps(encode_mask(mask)))
        self.assertTrue(np.array_equal(decode_mask(payload, mask.shape), mask))

    def test_resize_keeps_the_mask_binary(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True
        resized = _fit(mask, 20, 20)
        self.assertEqual(resized.shape, (20, 20))
        self.assertEqual(resized.dtype, np.dtype(bool))
        self.assertEqual(set(np.unique(resized).tolist()), {False, True})


class BackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.weights = Path(self.temporary.name) / "yolo11x-seg.pt"

    def test_missing_weights_say_how_to_get_them(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "weights sync"):
            YoloSeg(weights_path=self.weights).load()

    def test_predict_before_load_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "call load"):
            YoloSeg(weights_path=self.weights).predict(["frame.png"])

    def test_cache_identity_tracks_settings_that_change_masks(self) -> None:
        base = YoloSeg(weights_path=self.weights).cache_identity
        self.assertEqual(base["weights"], "yolo11x-seg.pt")
        self.assertNotEqual(
            base, YoloSeg(weights_path=self.weights, confidence=0.5).cache_identity
        )
        self.assertNotEqual(
            base,
            YoloSeg(weights_path=self.weights, dynamic_classes=("person",)).cache_identity,
        )

    def test_identity_ignores_class_declaration_order(self) -> None:
        first = YoloSeg(weights_path=self.weights, dynamic_classes=("car", "person"))
        second = YoloSeg(weights_path=self.weights, dynamic_classes=("person", "car"))
        self.assertEqual(first.cache_identity, second.cache_identity)

    def test_factory_rejects_unknown_params(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown yolo_seg params: nonsense"):
            build_yolo_seg(weights_path=self.weights, nonsense=1)

    def test_factory_declares_its_weight_requirement(self) -> None:
        self.assertEqual(build_yolo_seg.weight_requirements(), ("yolo11x_seg",))

    def test_backend_is_registered_under_the_segmenter_kind(self) -> None:
        from tgseqloc.components import register_builtin_components

        self.assertIn("yolo_seg", register_builtin_components().available("segmenter"))

    def test_default_classes_cover_both_groups(self) -> None:
        self.assertIn("person", DYNAMIC_CLASSES)
        self.assertIn("tv", DYNAMIC_CLASSES)


if __name__ == "__main__":
    unittest.main()
