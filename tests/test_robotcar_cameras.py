"""Несколько камер RobotCar как дополнительные кадры тех же проездов."""

import tempfile
import unittest
from pathlib import Path

from tgseqloc.data.robotcar import (
    camera_frame_stem,
    discover_camera_records,
    extend_fold_assignment,
    frame_camera,
)


def make_camera(root: Path, camera: str, stems_by_role: dict) -> dict:
    for role, (traversal, stems) in stems_by_role.items():
        (root / "graphs" / camera / role).mkdir(parents=True, exist_ok=True)
        (root / "ocr" / camera / traversal).mkdir(parents=True, exist_ok=True)
        for stem in stems:
            (root / "ocr" / camera / traversal / stem).mkdir(parents=True, exist_ok=True)
            (root / "graphs" / camera / role / f"{stem}.json").write_text("{}", encoding="utf-8")
    return {
        "ocr_root_template": str(root / "ocr" / "{camera}" / "{traversal}"),
        "scene_graph_root_template": str(root / "graphs" / "{camera}" / "{sequence}"),
    }


class CameraDiscoveryTests(unittest.TestCase):
    def test_primary_keeps_names_and_indices_others_follow_prefixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traversals = {"base": "t1", "query": "t2"}
            front = make_camera(root, "front", {"base": ("t1", ["100", "200"]), "query": ("t2", ["300"])})
            left = make_camera(root, "left", {"base": ("t1", ["150"]), "query": ("t2", ["310"])})
            right = make_camera(root, "right", {"base": ("t1", ["150"]), "query": ("t2", [])})
            records = discover_camera_records({"front": front, "left": left, "right": right}, traversals, primary="front")
            base = [r for r in records if r.sequence == "base"]
            self.assertEqual([r.stem for r in base], ["100", "200", "left-150", "right-150"])
            self.assertEqual([r.index for r in base], [0, 1, 2, 3])
            self.assertEqual([r.timestamp for r in base], [100, 200, 150, 150])
            query = [r for r in records if r.sequence == "query"]
            self.assertEqual([(r.stem, r.index) for r in query], [("300", 0), ("left-310", 1)])
            self.assertTrue(str(base[2].ocr_path).endswith("left/t1/150/paddleocr_v5.json"))

    def test_each_camera_reads_its_own_frame_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            front = make_camera(root, "front", {"base": ("t1", ["100", "200"])})
            left = make_camera(root, "left", {"base": ("t1", ["150", "250"])})
            (root / "left.txt").write_text("250\n", encoding="utf-8")
            left["frame_list_path"] = str(root / "left.txt")
            records = discover_camera_records({"front": front, "left": left}, {"base": "t1"}, primary="front")
            self.assertEqual([r.stem for r in records], ["100", "200", "left-250"])

    def test_primary_must_be_configured(self) -> None:
        with self.assertRaises(ValueError):
            discover_camera_records({"left": {}}, {"base": "t1"}, primary="front")

    def test_stem_round_trip(self) -> None:
        self.assertEqual(camera_frame_stem("front", 5, "front"), "5")
        self.assertEqual(frame_camera(camera_frame_stem("mono_left", 5, "front"), "front"), "mono_left")
        self.assertEqual(frame_camera("5", "front"), "front")


class FoldExtensionTests(unittest.TestCase):
    ASSIGNMENT = {"folds": 1, "assignment": {"0": {
        "train": ["1000000", "1500000"], "validation": ["5000000"], "test": ["9000000"]}}}

    def test_frames_join_the_part_of_the_nearest_primary_frame(self) -> None:
        frames = [("left-1100000", 1_100_000), ("rear-5200000", 5_200_000), ("right-8800000", 8_800_000)]
        out = extend_fold_assignment(self.ASSIGNMENT, frames, 300_000)["assignment"]["0"]
        self.assertIn("left-1100000", out["train"])
        self.assertIn("rear-5200000", out["validation"])
        self.assertIn("right-8800000", out["test"])

    def test_a_frame_far_from_every_primary_frame_joins_no_part(self) -> None:
        out = extend_fold_assignment(self.ASSIGNMENT, [("left-3000000", 3_000_000)], 300_000)["assignment"]["0"]
        self.assertFalse(any("left-3000000" in stems for stems in out.values()))

    def test_input_is_not_modified(self) -> None:
        extend_fold_assignment(self.ASSIGNMENT, [("left-1100000", 1_100_000)], 300_000)
        self.assertEqual(self.ASSIGNMENT["assignment"]["0"]["train"], ["1000000", "1500000"])


if __name__ == "__main__":
    unittest.main()
