"""Конвертация сцен-графов RobotCar из формата генератора."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tgseqloc.data.robotcar_graphs import (
    build_frame_index,
    convert_split,
    parse_scene_graph_text,
)

GOOD = """<answer>
      obj[3]{id,name,x1,y1,x2,y2}:
        1,tree,0,0,320,240
        2,car,320,240,640,480
        3,wall,0,0,640,480
      rel[2]{subj,pred,obj}:
        2,on,3
        1,next-to,2
</answer>"""

DEGENERATE = "<answer>\n obj[1]{id[1]{subj[1]{subj[2]{" + "subj[3]{" * 60 + "\n</answer>"


class ParseTests(unittest.TestCase):
    def test_boxes_are_normalized_by_the_generator_image_size(self) -> None:
        nodes, _ = parse_scene_graph_text(GOOD)
        self.assertEqual(nodes[0]["data"]["bbox_2d"]["xyxy"], [0.0, 0.0, 0.5, 0.5])
        self.assertEqual(nodes[1]["data"]["bbox_2d"]["xyxy"], [0.5, 0.5, 1.0, 1.0])

    def test_classes_and_relations_survive(self) -> None:
        nodes, links = parse_scene_graph_text(GOOD)
        self.assertEqual([n["data"]["class_name"] for n in nodes], ["tree", "car", "wall"])
        self.assertEqual([l["label"] for l in links], ["on", "next-to"])

    def test_degenerate_output_is_rejected_not_half_parsed(self) -> None:
        """0.9 % базового сплита — повтор скобок вместо таблицы."""

        self.assertIsNone(parse_scene_graph_text(DEGENERATE))

    def test_missing_object_table_is_rejected(self) -> None:
        self.assertIsNone(parse_scene_graph_text("<answer>\n</answer>"))
        self.assertIsNone(parse_scene_graph_text(""))

    def test_zero_area_box_is_dropped(self) -> None:
        """Бокс нулевой площади не несёт положения, его геометрия бессмысленна."""

        text = "obj[2]{id,name,x1,y1,x2,y2}:\n  1,adult,10,10,10,50\n  2,car,0,0,640,480\n"
        nodes, _ = parse_scene_graph_text(text)
        self.assertEqual([n["data"]["class_name"] for n in nodes], ["car"])

    def test_relation_to_a_dropped_object_is_dropped(self) -> None:
        text = "obj[1]{id,name,x1,y1,x2,y2}:\n  1,car,0,0,64,48\nrel[1]{subj,pred,obj}:\n  1,on,9\n"
        nodes, links = parse_scene_graph_text(text)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(links, [])

    def test_coordinates_are_clamped_to_the_frame(self) -> None:
        text = "obj[1]{id,name,x1,y1,x2,y2}:\n  1,sky,-20,-5,700,500\n"
        nodes, _ = parse_scene_graph_text(text)
        self.assertEqual(nodes[0]["data"]["bbox_2d"]["xyxy"], [0.0, 0.0, 1.0, 1.0])


class ConversionTests(unittest.TestCase):
    def _tree(self, root: Path) -> tuple[Path, dict]:
        chunk = root / "chunks" / "rc-base-centre_c000"
        chunk.mkdir(parents=True)
        (chunk / "frame_index.json").write_text(
            json.dumps({"frames": {"000000": "1439388259735205.png",
                                   "000001": "1439388260235138.png"}}),
            encoding="utf-8",
        )
        jsonl = root / "base.jsonl"
        jsonl.write_text(
            "\n".join(
                json.dumps({"images": [str(chunk / f"{local}.jpg")], "predict": predict})
                for local, predict in (("000000", GOOD), ("000001", DEGENERATE))
            ),
            encoding="utf-8",
        )
        return jsonl, build_frame_index([root / "chunks"])

    def test_frames_are_named_by_their_original_timestamp(self) -> None:
        """Генератор нумерует кадры внутри чанка; остальной пайплайн — таймстампом."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl, index = self._tree(root)
            out = root / "graphs"
            stats = convert_split(jsonl, index, out)
            self.assertEqual([p.name for p in sorted(out.glob("*.json"))],
                             ["1439388259735205.json"])
            self.assertEqual(stats.written, 1)
            self.assertEqual(stats.degenerate, 1)
            self.assertEqual(stats.records, 2)

    def test_written_graph_matches_the_pipeline_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl, index = self._tree(root)
            out = root / "graphs"
            convert_split(jsonl, index, out)
            payload = json.loads((out / "1439388259735205.json").read_text())
            self.assertEqual(set(payload) >= {"nodes", "links"}, True)
            node = payload["nodes"][0]
            self.assertIn("xyxy", node["data"]["bbox_2d"])
            self.assertIn("class_name", node["data"])

    def test_unknown_chunk_is_counted_not_guessed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl, _ = self._tree(root)
            stats = convert_split(jsonl, {}, root / "graphs")
            self.assertEqual(stats.written, 0)
            self.assertEqual(stats.unknown_frame, 2)


if __name__ == "__main__":
    unittest.main()
