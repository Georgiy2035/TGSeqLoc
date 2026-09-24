from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tgseqloc.data.formats import FrameRecord, frame_text_from_json
from tgseqloc.data.v4rl import (
    build_vocabularies,
    build_gt_mapping,
    build_temporal_split,
    discover_v4rl_records,
    parse_scene_graph,
)
from tgseqloc.preparation.v4rl import (
    build_preprocess_fingerprint,
    load_prepared_graph,
    process_v4rl,
)
from tgseqloc.training import Trainer

from tests.helpers import FakeEncoder, make_v4rl_tree, text_annotation


class IdentityEncoder(FakeEncoder):
    def __init__(self, identity: str) -> None:
        super().__init__()
        self.cache_identity = {"weights": identity}


class AlternateIdentityEncoder(IdentityEncoder):
    pass


class MissingIdentityEncoder(FakeEncoder):
    cache_identity = None


class StatefulEncoderFactory:
    def __call__(self, **kwargs):
        return FakeEncoder()


class V4RLSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings, _ = make_v4rl_tree(self.root, frame_count=4)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def discover(self):
        return discover_v4rl_records(
            str(self.settings["text_root_template"]),
            str(self.settings["scene_graph_root_template"]),
            self.settings["sequences"],
        )

    def test_discovery_is_ordered_and_pairs_annotations(self) -> None:
        records = self.discover()
        self.assertEqual(len(records), 8)
        self.assertEqual(
            [(record.sequence, record.index) for record in records],
            [("seq1", index) for index in range(4)]
            + [("seq2", index) for index in range(4)],
        )
        self.assertTrue(all(record.text_path.is_file() for record in records))
        self.assertTrue(all(record.graph_path.is_file() for record in records))
        self.assertEqual(records[0].text_path.name, records[0].graph_path.name)
        records[0].text_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing text annotation"):
            self.discover()

    def test_text_and_scene_graph_annotations_are_read(self) -> None:
        record = self.discover()[0]
        frame_text = frame_text_from_json(json.loads(record.text_path.read_text()))
        self.assertEqual(frame_text.texts, ["shop 0"])
        self.assertEqual(frame_text.boxes, [[0.1, 0.1, 0.4, 0.4]])
        self.assertEqual([d.dynamic_score for d in frame_text.detections], [0.0])

        nodes, edges, dropped = parse_scene_graph(
            record.graph_path, {"unknown": 0, "building": 1}
        )
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["class_idx"], 1)
        self.assertEqual(edges, [(0, 0, "self")])
        self.assertEqual(dropped, 1)

    def test_temporal_split_can_produce_all_three_nonempty_parts(self) -> None:
        split = build_temporal_split(
            10,
            {index: [index] for index in range(10)},
            test_ratio=0.2,
            validation_ratio=0.25,
        )
        self.assertEqual(split["train_query_indices"], list(range(6)))
        self.assertEqual(split["validation_query_indices"], [6, 7])
        self.assertEqual(split["test_query_indices"], [8, 9])

    def test_zero_test_ratio_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 < test < 1"):
            build_temporal_split(10, {0: [0]}, test_ratio=0.0)

    def test_gt_empty_field_reports_line_number(self) -> None:
        records = self.discover()
        gt_path = self.root / "malformed.txt"
        valid = (
            f"{records[4].timestamp},"
            f"{records[0].timestamp},{records[0].timestamp}"
        )
        gt_path.write_text(valid + "\n1,,3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"line 2: empty CSV field"):
            build_gt_mapping(gt_path, records, tolerance_ns=1_000)


class V4RLPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings, _ = make_v4rl_tree(self.root, frame_count=4)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_prepare_reuse_rebuild_require_existing_and_source_mutation(self) -> None:
        encoder = FakeEncoder()
        first = process_v4rl(self.settings, encoder=encoder)
        self.assertEqual(first["written_count"], 8)
        self.assertEqual(first["skipped_count"], 0)
        self.assertTrue(first["output_complete"])
        self.assertEqual(first["validated_output_count"], 8)

        graph_root = Path(self.settings["output_root"]) / "v4rl" / "seq1"
        graph_path = sorted(graph_root.glob("*.pt"))[0]
        graph = load_prepared_graph(graph_path)
        self.assertEqual(tuple(graph.x.shape), (2, 4))
        self.assertEqual(tuple(graph.text_emb.shape), (2, encoder.embedding_dim))
        self.assertEqual(graph.text_strings, ["shop 0"])
        original_mtime = graph_path.stat().st_mtime_ns

        reused = process_v4rl(self.settings, encoder=FakeEncoder())
        self.assertEqual(reused["written_count"], 0)
        self.assertEqual(reused["skipped_count"], 8)
        self.assertEqual(graph_path.stat().st_mtime_ns, original_mtime)

        required_settings = dict(self.settings, cache_policy="require_existing")
        required = process_v4rl(required_settings, encoder=FakeEncoder())
        self.assertEqual(required["skipped_count"], 8)

        rebuilt = process_v4rl(
            dict(self.settings, cache_policy="rebuild"), encoder=FakeEncoder()
        )
        self.assertEqual(rebuilt["written_count"], 8)
        self.assertEqual(rebuilt["skipped_count"], 0)

        text_path = discover_v4rl_records(
            str(self.settings["text_root_template"]),
            str(self.settings["scene_graph_root_template"]),
            self.settings["sequences"],
        )[0].text_path
        payload = json.loads(text_path.read_text(encoding="utf-8"))
        payload["detections"][0]["text"] = "mutated source"
        text_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "require_existing.*incompatible"):
            process_v4rl(required_settings, encoder=FakeEncoder())
        refreshed = process_v4rl(self.settings, encoder=FakeEncoder())
        self.assertEqual(refreshed["written_count"], 1)
        self.assertEqual(refreshed["skipped_count"], 7)
        self.assertNotEqual(
            refreshed["preprocess_fingerprint"], first["preprocess_fingerprint"]
        )

    def test_require_existing_rejects_absent_outputs(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "require_existing.*absent"):
            process_v4rl(
                dict(self.settings, cache_policy="require_existing"),
                encoder=FakeEncoder(),
            )

    def test_invalid_dataset_and_cache_policy_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported dataset"):
            process_v4rl(dict(self.settings, dataset="other"), encoder=FakeEncoder())
        with self.assertRaisesRegex(ValueError, "Invalid V4RL cache policy"):
            process_v4rl(
                dict(self.settings, cache_policy="invalid"), encoder=FakeEncoder()
            )

    def test_encoder_cache_identity_and_class_invalidate_outputs(self) -> None:
        first = process_v4rl(self.settings, encoder=IdentityEncoder("alpha"))
        self.assertEqual(first["written_count"], 8)

        reused = process_v4rl(self.settings, encoder=IdentityEncoder("alpha"))
        self.assertEqual(reused["written_count"], 0)
        self.assertEqual(reused["skipped_count"], 8)

        identity_changed = process_v4rl(
            self.settings, encoder=IdentityEncoder("beta")
        )
        self.assertEqual(identity_changed["written_count"], 8)
        self.assertNotEqual(
            identity_changed["global_preprocess_fingerprint"],
            first["global_preprocess_fingerprint"],
        )

        class_changed = process_v4rl(
            self.settings, encoder=AlternateIdentityEncoder("beta")
        )
        self.assertEqual(class_changed["written_count"], 8)
        self.assertNotEqual(
            class_changed["global_preprocess_fingerprint"],
            identity_changed["global_preprocess_fingerprint"],
        )

    def test_stateful_encoder_requires_identity_and_tampering_rebuilds(self) -> None:
        with self.assertRaisesRegex(ValueError, "cache_identity"):
            process_v4rl(self.settings, encoder=MissingIdentityEncoder())
        with self.assertRaisesRegex(ValueError, "cache_identity"):
            process_v4rl(
                self.settings,
                text_encoder_factory=StatefulEncoderFactory(),
            )

        first = process_v4rl(self.settings, encoder=FakeEncoder())
        graph_record = first["graph_records"][0]
        graph_path = Path(self.settings["output_root"]) / "v4rl" / graph_record["path"]
        graph = load_prepared_graph(graph_path)
        graph.text_strings = ["tampered"]
        torch.save(graph, graph_path)

        repaired = process_v4rl(self.settings, encoder=FakeEncoder())
        self.assertEqual(repaired["written_count"], 1)
        self.assertEqual(repaired["skipped_count"], 7)

        manifest_path = Path(self.settings["output_root"]) / "v4rl" / "manifest.json"
        manifest_path.unlink()
        graph = load_prepared_graph(graph_path)
        graph.text_strings = ["tampered without manifest"]
        torch.save(graph, graph_path)
        rebuilt = process_v4rl(self.settings, encoder=FakeEncoder())
        self.assertEqual(rebuilt["written_count"], 8)

        manifest_path.write_text("[]", encoding="utf-8")
        rebuilt_from_invalid_manifest = process_v4rl(
            self.settings, encoder=FakeEncoder()
        )
        self.assertEqual(rebuilt_from_invalid_manifest["written_count"], 8)

    def test_stale_graph_is_reported_but_excluded_from_generated_split(self) -> None:
        process_v4rl(self.settings, encoder=FakeEncoder())
        data_root = Path(self.settings["output_root"]) / "v4rl"
        stale = data_root / "seq1" / "999999_stale.pt"
        source = sorted((data_root / "seq1").glob("*.pt"))[0]
        stale.write_bytes(source.read_bytes())

        manifest = process_v4rl(self.settings, encoder=FakeEncoder())
        self.assertEqual(manifest["stale_graph_paths"], ["seq1/999999_stale.pt"])
        split_path = data_root / "mappings" / "temporal_split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        selected = split["database_paths"] + split["query_paths"]
        self.assertNotIn("seq1/999999_stale.pt", selected)

        trainer = Trainer(self.settings, data_root, manifest=manifest, split=split)
        self.assertNotIn(stale, trainer.database_paths)
        self.assertNotIn(stale, trainer.query_paths)


class TextFilterTests(unittest.TestCase):
    """The junk and dynamic-text filters applied while reading annotations."""

    TEXTS = ["1", "-", "0", "50", "50%", "GIVE WAY", "ab", "xyz"]

    def frame(self, scores=None):
        scores = scores or [0.0] * len(self.TEXTS)
        return frame_text_from_json(
            text_annotation(
                [
                    (text, (0.01 * i, 0.01 * i, 0.01 * i + 0.05, 0.01 * i + 0.05), score)
                    for i, (text, score) in enumerate(zip(self.TEXTS, scores))
                ]
            )
        )

    def test_junk_filter_is_off_without_settings(self) -> None:
        from tgseqloc.preparation.v4rl import _drop_junk_text

        self.assertEqual(list(_drop_junk_text({}, self.frame()).texts), self.TEXTS)

    def test_short_and_digit_only_strings_are_dropped(self) -> None:
        from tgseqloc.preparation.v4rl import _drop_junk_text

        kept = _drop_junk_text({"text_min_length": 3, "text_drop_numeric": True}, self.frame())
        self.assertEqual(list(kept.texts), ["50%", "GIVE WAY", "xyz"])
        self.assertEqual(len(kept.boxes), len(kept.texts))

    def test_the_digit_rule_alone(self) -> None:
        from tgseqloc.preparation.v4rl import _drop_junk_text

        kept = _drop_junk_text({"text_drop_numeric": True}, self.frame())
        self.assertEqual(list(kept.texts), ["-", "50%", "GIVE WAY", "ab", "xyz"])

    def test_dynamic_text_is_dropped_at_the_threshold(self) -> None:
        from tgseqloc.preparation.v4rl import _drop_dynamic_text

        scores = [0.0, 0.49, 0.5, 1.0, 0.0, 0.7, 0.2, 0.0]
        kept = _drop_dynamic_text({"dynamic_text_threshold": 0.5}, self.frame(scores))
        self.assertEqual(list(kept.texts), ["1", "-", "50%", "ab", "xyz"])

    def test_dynamic_filter_is_off_without_a_threshold(self) -> None:
        from tgseqloc.preparation.v4rl import _drop_dynamic_text

        kept = _drop_dynamic_text({}, self.frame([1.0] * len(self.TEXTS)))
        self.assertEqual(list(kept.texts), self.TEXTS)

    def test_dynamic_filter_needs_scores(self) -> None:
        from tgseqloc.preparation.v4rl import _drop_dynamic_text

        frame = frame_text_from_json(text_annotation([("SHOP", (0.1, 0.1, 0.2, 0.2), None)]))
        with self.assertRaisesRegex(ValueError, "dynamic_score"):
            _drop_dynamic_text({"dynamic_text_threshold": 0.5}, frame)

    def test_fingerprint_changes_with_the_filters(self) -> None:
        base = {"dataset": "v4rl", "sequences": ["seq1"]}
        plain = build_preprocess_fingerprint(base, {}, {}, 64)
        off = build_preprocess_fingerprint({**base, "text_min_length": 0, "text_drop_numeric": False}, {}, {}, 64)
        on = build_preprocess_fingerprint({**base, "text_min_length": 3, "text_drop_numeric": True}, {}, {}, 64)
        dynamic = build_preprocess_fingerprint({**base, "dynamic_text_threshold": 0.5}, {}, {}, 64)
        self.assertEqual(plain, off)
        self.assertNotEqual(plain, on)
        self.assertNotEqual(plain, dynamic)


class DynamicNodeClassTests(unittest.TestCase):
    """Dynamic object classes can be removed from the graph by name."""

    GRAPH = {
        "nodes": [
            {"id": 1, "data": {"class_name": "wall", "bbox_2d": {"xyxy": [0, 0, 1, 1]}}},
            {"id": 2, "data": {"class_name": "car", "bbox_2d": {"xyxy": [0.2, 0.2, 0.6, 0.6]}}},
            {"id": 3, "data": {"class_name": "tree", "bbox_2d": {"xyxy": [0.5, 0, 0.9, 0.5]}}},
        ],
        "links": [
            {"source": 2, "target": 1, "label": "on"},
            {"source": 3, "target": 1, "label": "next-to"},
        ],
    }

    def _graph(self, root: Path) -> Path:
        path = root / "graph.json"
        path.write_text(json.dumps(self.GRAPH), encoding="utf-8")
        return path

    def test_without_a_filter_every_node_survives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nodes, edges, _ = parse_scene_graph(
                self._graph(Path(tmp)), {"unknown": 0, "wall": 1, "car": 2, "tree": 3}
            )
            self.assertEqual([n["class_name"] for n in nodes], ["wall", "car", "tree"])
            self.assertEqual(len(edges), 2)

    def test_named_classes_are_removed_with_their_edges(self) -> None:
        """An edge touching a removed node is removed with it."""

        with tempfile.TemporaryDirectory() as tmp:
            nodes, edges, dropped = parse_scene_graph(
                self._graph(Path(tmp)),
                {"unknown": 0, "wall": 1, "car": 2, "tree": 3},
                dynamic_classes=("car",),
            )
            self.assertEqual([n["class_name"] for n in nodes], ["wall", "tree"])
            self.assertEqual([label for _, _, label in edges], ["next-to"])
            self.assertEqual(dropped, 1)

    def test_matching_ignores_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nodes, _, _ = parse_scene_graph(
                self._graph(Path(tmp)),
                {"unknown": 0, "wall": 1, "car": 2, "tree": 3},
                dynamic_classes=("CAR",),
            )
            self.assertNotIn("car", [n["class_name"] for n in nodes])

    def test_vocabulary_excludes_removed_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = FrameRecord("base", 0, 1, "1", None, root / "text.json",
                                 self._graph(root))
            classes, _ = build_vocabularies([record], dynamic_classes=("car",))
            self.assertNotIn("car", classes)
            self.assertIn("wall", classes)


if __name__ == "__main__":
    unittest.main()
