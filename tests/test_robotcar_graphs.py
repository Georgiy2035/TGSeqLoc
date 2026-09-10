"""Конвертация сцен-графов RobotCar из формата генератора."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
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

# Второе поколение генератора: цвет и материал между именем и боксом.
NEW_LAYOUT = """<answer>
obj[3]{id,name,color,material,x1,y1,x2,y2}:
  1,wall,brown,brick,0,0,320,240
  2,ivy,green,organic,320,240,640,480
  3,house,white,brick,0,0,640,480
rel[2]{subj,pred,obj}:
  2,attached-to,1
  3,behind,1
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
        self.assertIsNone(parse_scene_graph_text(None))

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


class LayoutTests(unittest.TestCase):
    """Колонки читаются из заголовка таблицы, а не подразумеваются."""

    def test_the_second_generation_layout_is_read(self) -> None:
        nodes, links = parse_scene_graph_text(NEW_LAYOUT)
        self.assertEqual([n["data"]["class_name"] for n in nodes], ["wall", "ivy", "house"])
        self.assertEqual(nodes[0]["data"]["bbox_2d"]["xyxy"], [0.0, 0.0, 0.5, 0.5])
        self.assertEqual([l["label"] for l in links], ["attached-to", "behind"])

    def test_extra_columns_are_kept_as_attributes(self) -> None:
        nodes, _ = parse_scene_graph_text(NEW_LAYOUT)
        self.assertEqual(nodes[0]["data"]["attributes"],
                         {"color": "brown", "material": "brick"})

    def test_the_first_layout_carries_no_attributes_key(self) -> None:
        """Файл первого поколения должен остаться побайтно тем же, что до правки."""

        nodes, _ = parse_scene_graph_text(GOOD)
        self.assertTrue(all("attributes" not in n["data"] for n in nodes))

    def test_columns_are_located_by_name_not_position(self) -> None:
        text = "obj[1]{name,x1,y1,x2,y2,id,color}:\n  wall,0,0,320,240,7,red\n"
        nodes, _ = parse_scene_graph_text(text)
        self.assertEqual(nodes[0]["id"], 7)
        self.assertEqual(nodes[0]["data"]["class_name"], "wall")
        self.assertEqual(nodes[0]["data"]["attributes"], {"color": "red"})

    def test_a_table_without_a_required_column_is_rejected(self) -> None:
        self.assertIsNone(parse_scene_graph_text("obj[1]{id,name,x1,y1}:\n  1,wall,0,0\n"))

    def test_a_row_with_the_wrong_column_count_is_dropped(self) -> None:
        counters: Counter = Counter()
        text = ("obj[2]{id,name,color,material,x1,y1,x2,y2}:\n"
                "  1,wall,brown,brick,0,0,320,240\n  2,sign,red,0,0,64,48\n")
        nodes, _ = parse_scene_graph_text(text, counters=counters)
        self.assertEqual([n["data"]["class_name"] for n in nodes], ["wall"])
        self.assertEqual(counters["dropped_boxes"], 1)


class CounterTests(unittest.TestCase):
    """Счётчики потерь раньше не увеличивались вовсе и в отчёте всегда были нулями."""

    def test_zero_area_and_duplicate_boxes_are_counted(self) -> None:
        counters: Counter = Counter()
        text = ("obj[3]{id,name,x1,y1,x2,y2}:\n"
                "  1,adult,10,10,10,50\n  2,car,0,0,64,48\n  2,car,0,0,64,48\n")
        parse_scene_graph_text(text, counters=counters)
        self.assertEqual(counters["dropped_boxes"], 2)

    def test_a_relation_to_a_missing_object_is_counted(self) -> None:
        counters: Counter = Counter()
        text = ("obj[1]{id,name,x1,y1,x2,y2}:\n  1,car,0,0,64,48\n"
                "rel[2]{subj,pred,obj}:\n  1,on,9\n  1,on,1\n")
        _, links = parse_scene_graph_text(text, counters=counters)
        self.assertEqual(len(links), 1)
        self.assertEqual(counters["dropped_relations"], 1)

    def test_the_layout_is_recorded(self) -> None:
        counters: Counter = Counter()
        parse_scene_graph_text(NEW_LAYOUT, counters=counters)
        self.assertEqual(counters["layout:id,name,color,material,x1,y1,x2,y2"], 1)


class ConversionTests(unittest.TestCase):
    def _tree(self, root: Path, predicts=None) -> tuple[Path, dict]:
        predicts = predicts or (("000000", GOOD), ("000001", DEGENERATE))
        chunk = root / "chunks" / "rc-base-centre_c000"
        chunk.mkdir(parents=True)
        (chunk / "frame_index.json").write_text(
            json.dumps({"frames": {"000000": "1439388259735205.png",
                                   "000001": "1439388260235138.png",
                                   "000002": "1439388260735071.png"}}),
            encoding="utf-8",
        )
        jsonl = root / "base.jsonl"
        jsonl.write_text(
            "\n".join(
                json.dumps({"images": [str(chunk / f"{local}.jpg")], "predict": predict})
                for local, predict in predicts
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

    def test_both_layouts_and_a_failed_completion_in_one_file(self) -> None:
        """Упавшая генерация хранит null; раньше это обрывало всю конвертацию."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl, index = self._tree(root, (("000000", GOOD), ("000001", NEW_LAYOUT),
                                             ("000002", None)))
            out = root / "graphs"
            stats = convert_split(jsonl, index, out)
            self.assertEqual(stats.written, 2)
            self.assertEqual(stats.unparsed, 1)
            self.assertEqual(set(stats.layouts), {"id,name,x1,y1,x2,y2",
                                                  "id,name,color,material,x1,y1,x2,y2"})
            fresh = json.loads((out / "1439388260235138.json").read_text())
            self.assertEqual(fresh["nodes"][1]["data"]["attributes"]["material"], "organic")

    def test_losses_reach_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = ("obj[2]{id,name,x1,y1,x2,y2}:\n  1,adult,10,10,10,50\n  2,car,0,0,64,48\n"
                    "rel[1]{subj,pred,obj}:\n  1,on,2\n")
            jsonl, index = self._tree(root, (("000000", text),))
            stats = convert_split(jsonl, index, root / "graphs")
            self.assertEqual(stats.as_dict()["dropped_boxes"], 1)
            self.assertEqual(stats.as_dict()["dropped_relations"], 1)


if __name__ == "__main__":
    unittest.main()
