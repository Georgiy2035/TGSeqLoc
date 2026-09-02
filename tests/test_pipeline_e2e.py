from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tgseqloc.pipeline import PipelineRunner

from tests.helpers import make_app_config


class PipelineEndToEndTests(unittest.TestCase):
    def test_prepare_train_one_epoch_and_evaluate_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, encoder = make_app_config(root, frame_count=10)
            runner = PipelineRunner(config)

            manifest = runner.prepare(encoder=encoder)
            self.assertTrue(manifest["output_complete"])
            self.assertEqual(manifest["frame_count"], 20)
            self.assertEqual(manifest["text_encoder"], "fake-encoder")
            self.assertFalse(torch.cuda.is_initialized())

            split = json.loads(runner.split_path.read_text(encoding="utf-8"))
            self.assertTrue(split["train_query_indices"])
            self.assertTrue(split["validation_query_indices"])
            self.assertTrue(split["test_query_indices"])

            training = runner.train()
            self.assertEqual(training["last_epoch"], 1)
            self.assertEqual(len(training["history"]), 1)
            self.assertTrue(Path(training["best_checkpoint"]).is_file())
            self.assertTrue(Path(training["last_checkpoint"]).is_file())
            self.assertGreater(training["final_metrics"]["num_evaluated_queries"], 0)

            trainer = runner.trainer()
            _, nearest = trainer.evaluate_test("best", write_results=False)
            self.assertEqual(nearest.shape[1], config.retrieval.top_k)

            metrics = runner.evaluate("best")
            self.assertIn("R@1", metrics)
            self.assertEqual(
                metrics["num_evaluated_queries"],
                len(split["test_query_indices"]),
            )
            self.assertTrue((config.output.root / "e2e" / "final_metrics.json").is_file())
            self.assertTrue((config.output.root / "e2e" / "test_retrievals.json").is_file())

    def test_nondefault_sequences_flow_through_split_trainer_and_patience_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, encoder = make_app_config(
                root,
                frame_count=10,
                reference_sequence="gallery",
                query_sequence="probe",
            )
            config.training.epochs = 2
            config.training.patience = 0
            runner = PipelineRunner(config)
            runner.prepare(encoder=encoder)

            split = json.loads(runner.split_path.read_text(encoding="utf-8"))
            self.assertEqual(split["reference_sequence"], "gallery")
            self.assertEqual(split["query_sequence"], "probe")
            self.assertTrue(
                all(path.startswith("gallery/") for path in split["database_paths"])
            )
            self.assertTrue(
                all(path.startswith("probe/") for path in split["query_paths"])
            )

            trainer = runner.trainer()
            self.assertTrue(
                all(path.parent.name == "gallery" for path in trainer.database_paths)
            )
            self.assertTrue(
                all(path.parent.name == "probe" for path in trainer.query_paths)
            )
            training = trainer.train()
            self.assertEqual(training["last_epoch"], 2)
            self.assertEqual(len(training["history"]), 2)
            self.assertGreater(training["final_metrics"]["num_evaluated_queries"], 0)

    def test_manifest_sha_rejects_mutated_and_deleted_prepared_graphs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, encoder = make_app_config(root, frame_count=6)
            runner = PipelineRunner(config)
            manifest = runner.prepare(encoder=encoder)
            data_root = runner.prepared_data_root
            first = data_root / manifest["expected_graph_paths"][0]
            second = data_root / manifest["expected_graph_paths"][1]
            original = first.read_bytes()

            first.write_bytes(original + b"mutation")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                runner.train()

            first.write_bytes(original)
            second.unlink()
            with self.assertRaisesRegex(ValueError, "does not exist"):
                runner.trainer()


if __name__ == "__main__":
    unittest.main()
