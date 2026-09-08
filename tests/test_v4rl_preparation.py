from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tgseqloc.data.v4rl import (
    build_gt_mapping,
    build_temporal_split,
    discover_v4rl_records,
    parse_paddleocr,
    parse_scene_graph,
)
from tgseqloc.preparation.v4rl import (
    build_preprocess_fingerprint,
    load_prepared_graph,
    process_v4rl,
)
from tgseqloc.training import Trainer

from tests.helpers import FakeEncoder, make_v4rl_tree


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
            self.settings["dataset_root"],
            str(self.settings["ocr_root_template"]),
            str(self.settings["scene_graph_root_template"]),
            self.settings["sequences"],
            chunk_size=200,
        )

    def test_discovery_is_ordered_and_resolves_sidecars(self) -> None:
        records = self.discover()
        self.assertEqual(len(records), 8)
        self.assertEqual(
            [(record.sequence, record.index) for record in records],
            [("seq1", index) for index in range(4)]
            + [("seq2", index) for index in range(4)],
        )
        self.assertTrue(all(record.ocr_path.is_file() for record in records))
        self.assertTrue(all(record.graph_path.is_file() for record in records))
        records[0].ocr_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing OCR"):
            self.discover()

    def test_ocr_and_scene_graph_parsing_filters_and_normalizes(self) -> None:
        record = self.discover()[0]
        frame_text = parse_paddleocr(record.ocr_path, 0.5)
        self.assertEqual(frame_text.texts, ["shop 0"])
        self.assertEqual(frame_text.boxes, [[0.1, 0.1, 0.4, 0.4]])
        # Confidence used to be discarded by the parser; the dynamics stage
        # needs it to survive all the way to the graph.
        self.assertEqual([d.confidence for d in frame_text.detections], [0.95])
        self.assertIsNone(frame_text.detections[0].dynamic_score)

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

        ocr_path = discover_v4rl_records(
            self.settings["dataset_root"],
            str(self.settings["ocr_root_template"]),
            str(self.settings["scene_graph_root_template"]),
            self.settings["sequences"],
        )[0].ocr_path
        payload = json.loads(ocr_path.read_text(encoding="utf-8"))
        payload["predictions"][0]["text"] = "mutated source"
        ocr_path.write_text(json.dumps(payload), encoding="utf-8")

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


if __name__ == "__main__":
    unittest.main()


class OcrFileNameTests(unittest.TestCase):
    """Выбор распознавателя именем файла, а не отдельным деревом."""

    def _dataset(self, root: Path, names: tuple[str, ...]) -> None:
        frames = root / "seq1_frames"
        frames.mkdir(parents=True)
        (frames / "000000_1465238011340417397.png").write_bytes(b"")
        for name in names:
            frame_dir = root / "ocr" / "seq1" / "000000_1465238011340417397"
            frame_dir.mkdir(parents=True, exist_ok=True)
            (frame_dir / name).write_text(
                json.dumps({"image_width": 752, "image_height": 480, "predictions": []}),
                encoding="utf-8",
            )

    def test_default_reads_the_paddleocr_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._dataset(root, ("paddleocr_v5.json",))
            records = discover_v4rl_records(
                root, str(root / "ocr" / "{sequence}"), str(root / "graphs" / "{sequence}"),
                ("seq1",), require_inputs=False,
            )
            self.assertTrue(str(records[0].ocr_path).endswith("paddleocr_v5.json"))

    def test_another_recognizer_is_selected_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._dataset(root, ("paddleocr_v5.json", "qwen3_vl_4b.json"))
            records = discover_v4rl_records(
                root, str(root / "ocr" / "{sequence}"), str(root / "graphs" / "{sequence}"),
                ("seq1",), require_inputs=False, ocr_file_name="qwen3_vl_4b.json",
            )
            self.assertTrue(str(records[0].ocr_path).endswith("qwen3_vl_4b.json"))

    def test_fingerprint_separates_recognizers(self) -> None:
        """Иначе прогон на Qwen переиспользовал бы графы, собранные на Paddle."""

        base = {"dataset": "v4rl", "sequences": ["seq1"], "chunk_size": 200}
        first = build_preprocess_fingerprint(
            {**base, "ocr_file_name": "paddleocr_v5.json"}, {}, {}, 64
        )
        second = build_preprocess_fingerprint(
            {**base, "ocr_file_name": "qwen3_vl_4b.json"}, {}, {}, 64
        )
        self.assertNotEqual(first, second)
