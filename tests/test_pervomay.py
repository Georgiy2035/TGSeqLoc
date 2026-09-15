"""Первомайская: позы, 3D ground truth, разбиение только для оценки, чужой словарь классов."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tgseqloc.data.pervomay import (
    build_evaluation_split,
    discover_pervomay_records,
    load_poses,
    parse_timestamp_ns,
)
from tgseqloc.data.robotcar import build_radius_positives
from tgseqloc.data.v4rl import load_vocabularies
from tgseqloc.preparation.pervomay import process_pervomay
from tgseqloc.preparation.v4rl import load_prepared_graph

from tests.helpers import FakeEncoder, write_json

HEADER = "image,timestamp,tx,ty,tz,qx,qy,qz,qw\n"


def make_walk(root: Path, walk: str, stems_positions: dict, *, graphs=None, ocr=None) -> None:
    rows = [HEADER]
    for stem, (x, y, z) in stems_positions.items():
        rows.append(f"{stem}.jpg,1787823716.{int(stem):09d},{x},{y},{z},0,0,0,1\n")
    (root / "walks" / walk).mkdir(parents=True, exist_ok=True)
    (root / "walks" / walk / "poses.csv").write_text("".join(rows), encoding="utf-8")
    for stem in stems_positions:
        (root / "walks" / walk / f"{stem}.jpg").write_bytes(f"{walk}-{stem}".encode())
    for stem in graphs if graphs is not None else stems_positions:
        write_json(root / "graphs" / walk / f"{stem}.json", {
            "nodes": [
                {"id": 1, "class_name": "building", "bbox": [0.1, 0.1, 0.6, 0.9]},
                {"id": 2, "class_name": "lamp", "bbox": [0.7, 0.2, 0.8, 0.9]},
            ],
            "links": [{"source": 2, "target": 1, "label": "next to"}],
        })
    for stem in ocr if ocr is not None else stems_positions:
        write_json(root / "ocr" / walk / stem / "paddleocr_v5.json", {
            "image_width": 100, "image_height": 50,
            "predictions": [{"text": f"гастроном {stem}", "confidence": 0.9, "bbox": [15, 10, 20, 10]}],
        })


def settings(root: Path, **extra) -> dict:
    return {
        "dataset": "pervomay",
        "dataset_root": root / "walks",
        "sequences": ("base", "query"),
        "traversals": {"base": "28", "query": "27"},
        "ocr_root_template": str(root / "ocr" / "{traversal}"),
        "scene_graph_root_template": str(root / "graphs" / "{traversal}"),
        "poses_path_template": str(root / "walks" / "{traversal}" / "poses.csv"),
        "image_path_template": str(root / "walks" / "{traversal}"),
        "gt_path": root / "walks" / "28" / "poses.csv",
        "output_root": root / "prepared",
        "prepared_root": root / "prepared",
        "reference_sequence": "base",
        "query_sequence": "query",
        "gt_radius_m": 25.0,
        "frame_batch_size": 2,
        "device": "cpu",
        **extra,
    }


class PoseTests(unittest.TestCase):
    def test_timestamp_keeps_nanoseconds(self) -> None:
        """В float 19 значащих цифр не помещаются."""

        self.assertEqual(parse_timestamp_ns("1787823716.199182034"), 1787823716199182034)
        self.assertEqual(parse_timestamp_ns("5"), 5_000_000_000)
        self.assertEqual(parse_timestamp_ns("1.5"), 1_500_000_000)
        with self.assertRaises(ValueError):
            parse_timestamp_ns("1e9")

    def test_poses_are_keyed_by_image_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_walk(root, "27", {"000001": (1.0, 2.0, 3.0)})
            poses = load_poses(root / "walks" / "27" / "poses.csv")
            self.assertEqual(poses["000001"], (1787823716000000001, (1.0, 2.0, 3.0)))

    def test_repeated_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "poses.csv"
            path.write_text(HEADER + "1.jpg,1.0,0,0,0,0,0,0,1\n1.jpg,2.0,0,0,0,0,0,0,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_poses(path)


class DiscoveryTests(unittest.TestCase):
    def test_frames_need_text_and_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            positions = {"000001": (0, 0, 0), "000011": (1, 0, 0), "000021": (2, 0, 0)}
            make_walk(root, "27", positions, graphs=["000001", "000011"], ocr=["000011", "000021"])
            records = discover_pervomay_records(
                str(root / "ocr" / "{traversal}"), str(root / "graphs" / "{traversal}"),
                str(root / "walks" / "{traversal}" / "poses.csv"), {"query": "27"},
            )
            self.assertEqual([(r.stem, r.index) for r in records], [("000011", 0)])
            self.assertEqual(records[0].timestamp, 1787823716000000011)

    def test_frame_without_pose_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_walk(root, "27", {"000001": (0, 0, 0)})
            write_json(root / "graphs" / "27" / "000011.json", {"nodes": []})
            (root / "ocr" / "27" / "000011").mkdir(parents=True)
            with self.assertRaises(ValueError):
                discover_pervomay_records(
                    str(root / "ocr" / "{traversal}"), str(root / "graphs" / "{traversal}"),
                    str(root / "walks" / "{traversal}" / "poses.csv"), {"query": "27"},
                )


class GroundTruthTests(unittest.TestCase):
    def test_radius_counts_the_third_axis(self) -> None:
        positives = build_radius_positives({7: (0.0, 0.0, 0.0)}, {0: (0.0, 3.0, 20.0), 1: (0.0, 0.0, 30.0)}, 25.0)
        self.assertEqual(positives, {7: [0]})

    def test_evaluation_split_has_no_training_part(self) -> None:
        split = build_evaluation_split({0: (0, 0, 0), 1: (9, 9, 9), 2: (5, 5, 5)}, {0: [3], 2: [1, 4]})
        self.assertEqual(split["test_query_indices"], [0, 2])
        self.assertEqual(split["train_query_indices"], [])
        self.assertEqual(split["validation_query_indices"], [])


class PreparationTests(unittest.TestCase):
    def test_prepares_graphs_and_a_metric_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_walk(root, "28", {"000001": (0, 0, 0), "000011": (100, 0, 0)})
            make_walk(root, "27", {"000001": (0, 0, 10), "000011": (0, 0, 40), "000021": (100, 0, 1)})
            manifest = process_pervomay(settings(root), encoder=FakeEncoder())
            self.assertTrue(manifest["output_complete"])
            self.assertEqual(manifest["frame_count"], 5)
            split = json.loads((root / "prepared" / "pervomay" / "mappings" / "temporal_split.json").read_text())
            self.assertEqual(split["positives"], {"0": [0], "2": [1]})
            self.assertEqual(split["test_query_indices"], [0, 2])
            self.assertEqual(split["queries_without_positives"], 1)
            self.assertEqual(split["query_positions"]["1"], [0.0, 0.0, 40.0])
            self.assertEqual(split["database_paths"], ["base/000001.pt", "base/000011.pt"])
            ground_truth = json.loads((root / "prepared" / "pervomay" / "mappings" / "gt_radius.json").read_text())
            self.assertEqual(sorted(ground_truth["poses"]), ["base", "query"])

    def test_vocabulary_comes_from_another_manifest(self) -> None:
        """Класс, которого нет в чужом словаре, читается как unknown."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_walk(root, "28", {"000001": (0, 0, 0)})
            make_walk(root, "27", {"000001": (0, 0, 1)})
            source = root / "source_manifest.json"
            write_json(source, {
                "node_class_to_idx": {"unknown": 0, "building": 1, "tree": 2},
                "edge_label_to_idx": {"unknown": 0, "beside": 1, "next to": 2},
            })
            manifest = process_pervomay(settings(root, vocabulary_manifest=str(source)), encoder=FakeEncoder())
            self.assertEqual(manifest["node_class_to_idx"], {"unknown": 0, "building": 1, "tree": 2})
            graph = load_prepared_graph(root / "prepared" / "pervomay" / "query" / "000001.pt")
            self.assertEqual(graph.node_class[:2].tolist(), [1, 0])

    def test_vocabulary_manifest_must_start_with_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            write_json(path, {"node_class_to_idx": {"building": 0}, "edge_label_to_idx": {"unknown": 0}})
            with self.assertRaises(ValueError):
                load_vocabularies(path)
            write_json(path, {"node_class_to_idx": {"unknown": 0, "a": 2}, "edge_label_to_idx": {"unknown": 0}})
            with self.assertRaises(ValueError):
                load_vocabularies(path)


if __name__ == "__main__":
    unittest.main()
