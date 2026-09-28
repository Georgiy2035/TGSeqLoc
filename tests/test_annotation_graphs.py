"""Scene-graph annotation: relation rules, graph layout and acceptance checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tgseqloc.annotation.checks import check_sequence
from tgseqloc.annotation.graphs import (
    MASK2FORMER_DYNAMIC_CLASSES,
    STREET_VOCABULARY,
    build_graph,
    relation_label,
)
from tgseqloc.annotation.segmentation import DYNAMIC_CLASSES


class RelationTests(unittest.TestCase):
    def test_each_rule(self) -> None:
        cases = {
            "in": ((0.2, 0.2, 0.3, 0.3), (0.1, 0.1, 0.5, 0.5)),
            "standing on": ((0.1, 0.1, 0.3, 0.4), (0.0, 0.42, 0.5, 0.9)),
            "beside": ((0.1, 0.1, 0.3, 0.3), (0.2, 0.2, 0.5, 0.5)),
            "above": ((0.1, 0.0, 0.2, 0.1), (0.6, 0.5, 0.7, 0.6)),
            "next to": ((0.0, 0.5, 0.1, 0.6), (0.8, 0.5, 0.9, 0.6)),
        }
        for label, (source, target) in cases.items():
            with self.subTest(label=label):
                self.assertEqual(relation_label(source, target), label)

    def test_thresholds_are_strict(self) -> None:
        # Exactly 80 % of the source inside the target is not yet "in".
        self.assertEqual(relation_label((0.0, 0.0, 1.0, 0.5), (0.0, 0.0, 0.8, 0.5)), "beside")
        # "standing on" needs the bottom of the source within 0.05 of the target's top.
        self.assertEqual(relation_label((0.0, 0.5, 0.2, 0.6), (0.1, 0.56, 0.3, 0.9)), "standing on")
        self.assertEqual(relation_label((0.0, 0.5, 0.2, 0.6), (0.1, 0.66, 0.3, 0.9)), "above")


class BuildGraphTests(unittest.TestCase):
    OBJECTS = [
        ("building", (0.0, 0.0, 0.4, 0.6)),
        ("sign", (0.1, 0.1, 0.2, 0.2)),
        ("tree", (0.8, 0.2, 0.9, 0.7)),
    ]

    def test_every_object_links_to_its_nearest_object(self) -> None:
        graph = build_graph(self.OBJECTS, {"scene_name": "000001"})
        self.assertEqual([node["id"] for node in graph["nodes"]], [1, 2, 3])
        self.assertEqual(
            [(link["source"], link["target"], link["label"]) for link in graph["links"]],
            [(1, 2, "beside"), (2, 1, "in"), (3, 1, "next to")],
        )

    def test_neighbours_sets_the_number_of_relations(self) -> None:
        self.assertEqual(len(build_graph(self.OBJECTS, {}, neighbours=2)["links"]), 6)
        self.assertEqual(build_graph(self.OBJECTS, {}, neighbours=0)["links"], [])
        self.assertEqual(build_graph(self.OBJECTS[:1], {})["links"], [])


class VocabularyTests(unittest.TestCase):
    def test_grounding_dino_is_queried_with_twenty_static_classes(self) -> None:
        self.assertEqual(len(set(STREET_VOCABULARY)), 20)
        self.assertFalse(set(STREET_VOCABULARY) & MASK2FORMER_DYNAMIC_CLASSES)

    def test_mask2former_drops_every_class_the_text_filter_treats_as_dynamic(self) -> None:
        self.assertLessEqual(set(DYNAMIC_CLASSES), MASK2FORMER_DYNAMIC_CLASSES)


class CheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.text_root, self.graph_root = root / "text" / "seq1", root / "graphs" / "seq1"
        self.text_root.mkdir(parents=True)
        self.graph_root.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, stem: str, *, graph: dict | None = None, text: dict | None = None) -> None:
        if graph is not None:
            (self.graph_root / f"{stem}.json").write_text(json.dumps(graph), encoding="utf-8")
        if text is not None:
            (self.text_root / f"{stem}.json").write_text(json.dumps(text), encoding="utf-8")

    @staticmethod
    def text(stem: str, box=(0.1, 0.1, 0.2, 0.2)) -> dict:
        return {"frame_id": stem, "image_size": [640, 480], "detections": [{"box": list(box), "text": "shop"}]}

    def test_consistent_annotation_is_accepted(self) -> None:
        graph = build_graph(BuildGraphTests.OBJECTS, {"scene_name": "000001"})
        self.write("000001", graph=graph, text=self.text("000001"))
        report = check_sequence(self.text_root, self.graph_root)
        self.assertTrue(report.accepted, report.summary())
        self.assertEqual((report.frames, report.nodes, report.strings), (1, 3, 1))

    def test_each_defect_is_reported(self) -> None:
        dynamic = build_graph([("car", (0.1, 0.1, 0.2, 0.2))], {"scene_name": "000002"})
        dangling = build_graph([("tree", (0.1, 0.1, 0.2, 0.2))], {"scene_name": "000003"})
        dangling["links"].append({"source": 1, "target": 7, "label": "next to"})
        self.write("000001", graph=build_graph([], {"scene_name": "000009"}), text=self.text("000001"))
        self.write("000002", graph=dynamic, text=self.text("000002", box=(0.3, 0.3, 0.2, 0.4)))
        self.write("000003", graph=dangling)
        report = check_sequence(self.text_root, self.graph_root)
        self.assertFalse(report.accepted)
        self.assertEqual(len(report.binding_errors), 1)
        self.assertEqual(len(report.dynamic_nodes), 1)
        self.assertEqual(len(report.degenerate_boxes), 1)
        self.assertEqual(len(report.dangling_edges), 1)
        self.assertEqual(len(report.missing_text), 1)
        self.assertEqual(report.empty_graphs, 1)


if __name__ == "__main__":
    unittest.main()
