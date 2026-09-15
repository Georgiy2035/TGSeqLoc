"""Оценка чекпойнта на чужих данных: что модель может прочитать и что на своих данных ничего не меняется."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tgseqloc.pipeline import PipelineRunner
from tgseqloc.training.transfer import transfer_incompatibilities

from tests.helpers import make_app_config


def manifest(**changes) -> dict:
    base = {
        "node_class_to_idx": {"unknown": 0, "building": 1},
        "edge_label_to_idx": {"unknown": 0, "near": 1},
        "node_feature_dim": 4,
        "edge_attr_dim": 10,
        "text_embedding_dim": 64,
        "schema_version": 3,
        "uses_text_nodes": True,
        "connection_strategy": "overlap_nearest",
        "connection_k": 1,
        "backend_identities": {"implementations": {"encoder": {"cache_identity": {"n": [3, 5]}}}},
    }
    base.update(changes)
    return base


class CompatibilityTests(unittest.TestCase):
    def test_same_contract_is_readable(self) -> None:
        self.assertEqual(transfer_incompatibilities(manifest(), manifest(), {"use_text_nodes": True}), [])

    def test_other_vocabulary_is_not(self) -> None:
        other = manifest(node_class_to_idx={"unknown": 0, "tree": 1})
        self.assertEqual(
            transfer_incompatibilities(manifest(), other, {"use_text_nodes": True}),
            ["node_class_to_idx differs"],
        )

    def test_other_string_encoder_is_not(self) -> None:
        other = manifest(backend_identities={"implementations": {"encoder": {"cache_identity": {"n": [2]}}}})
        self.assertEqual(
            transfer_incompatibilities(manifest(), other, {"use_text_nodes": True}),
            ["the string encoder differs"],
        )

    def test_text_model_reads_graphs_without_text(self) -> None:
        """Контроль «текст удалён»: веса с текстом, графы без текста."""

        target = manifest(uses_text_nodes=False, connection_k=3)
        self.assertEqual(transfer_incompatibilities(manifest(), target, {"use_text_nodes": True}), [])

    def test_model_without_text_cannot_read_text(self) -> None:
        source = manifest(uses_text_nodes=False)
        problems = transfer_incompatibilities(source, manifest(), {"use_text_nodes": False})
        self.assertIn("a model without text cannot read graphs with text", problems)


class SameDataTests(unittest.TestCase):
    def test_transfer_on_own_data_reproduces_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, encoder = make_app_config(root, frame_count=10)
            config.retrieval.recall_values = [1, 5]
            runner = PipelineRunner(config)
            runner.prepare(encoder=encoder)
            training = runner.train()
            expected = runner.evaluate("best")
            expected_retrievals = json.loads((config.output.root / "e2e" / "test_retrievals.json").read_text())

            config.output.experiment_name = "transfer"
            result = PipelineRunner(config).transfer(Path(training["best_checkpoint"]))
            for key in ("R@1", "R@5", "num_evaluated_queries", "num_database_frames"):
                self.assertEqual(result[key], expected[key])
            retrievals = json.loads((config.output.root / "transfer" / "test_retrievals.json").read_text())
            self.assertEqual(retrievals, expected_retrievals)
            self.assertEqual(result["source_manifest_fingerprint"], result["manifest_fingerprint"])


if __name__ == "__main__":
    unittest.main()
