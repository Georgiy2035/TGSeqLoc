from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tgseqloc.config import AppConfig, ConfigError, load_config
from tgseqloc.registry import Registry


class ConfigTests(unittest.TestCase):
    def load_text(self, text: str) -> AppConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            return load_config(path)

    def test_strict_load_converts_nested_values_and_paths(self) -> None:
        config = self.load_text(
            """
dataset:
  root: /tmp/v4rl
  sequences: [reference, query]
  reference_sequence: reference
  query_sequence: query
runtime:
  device: cpu
  num_workers: 0
model:
  dropout: 0
"""
        )
        self.assertEqual(config.dataset.root, Path("/tmp/v4rl"))
        self.assertEqual(config.dataset.sequences, ["reference", "query"])
        self.assertEqual(config.model.dropout, 0.0)
        self.assertEqual(config.runtime.num_workers, 0)

    def test_rejects_unknown_keys_and_wrong_types(self) -> None:
        cases = (
            ("mystery: true\n", "unknown key"),
            ("dataset:\n  mystery: true\n", "unknown key"),
            ("runtime:\n  num_workers: 'zero'\n", "must be an integer"),
            ("runtime:\n  device: false\n", "must be a string"),
            ("dataset:\n  root: 7\n", "must be a path string"),
            ("[]\n", "root must be a mapping"),
        )
        for yaml_text, message in cases:
            with self.subTest(yaml_text=yaml_text):
                with self.assertRaisesRegex(ConfigError, message):
                    self.load_text(yaml_text)

    def test_rejects_invalid_registered_choices_and_cross_field_values(self) -> None:
        mutations = (
            ("dataset.adapter", lambda c: setattr(c.dataset, "adapter", "other")),
            ("sources.scene_graph", lambda c: setattr(c.sources, "scene_graph", "generated")),
            ("preprocess.fusion", lambda c: setattr(c.preprocess, "fusion", "concat")),
            ("preprocess.connection_strategy", lambda c: setattr(c.preprocess, "connection_strategy", "random")),
            ("cache.policy", lambda c: setattr(c.cache, "policy", "forever")),
            ("model.graph_encoder", lambda c: setattr(c.model, "graph_encoder", "gcn")),
            ("retrieval.retriever", lambda c: setattr(c.retrieval, "retriever", "annoy")),
        )
        for field, mutate in mutations:
            with self.subTest(field=field):
                config = AppConfig()
                mutate(config)
                with self.assertRaisesRegex(ConfigError, field):
                    config.validate()

        config = AppConfig()
        config.dataset.reference_sequence = config.dataset.query_sequence
        with self.assertRaisesRegex(ConfigError, "must differ"):
            config.validate()
        config = AppConfig()
        config.training.recall_values = [5, 1]
        with self.assertRaisesRegex(ConfigError, "sorted"):
            config.validate()

    def test_rejects_zero_test_ratio_and_nonfinite_numbers(self) -> None:
        mutations = (
            ("dataset.test_ratio", lambda c: setattr(c.dataset, "test_ratio", 0.0)),
            ("dataset.test_ratio", lambda c: setattr(c.dataset, "test_ratio", float("nan"))),
            ("dataset.validation_ratio", lambda c: setattr(c.dataset, "validation_ratio", float("inf"))),
            (
                "preprocess.dynamic_text_threshold",
                lambda c: setattr(c.preprocess, "dynamic_text_threshold", float("nan")),
            ),
            ("model.dropout", lambda c: setattr(c.model, "dropout", float("nan"))),
            (
                "training.learning_rate",
                lambda c: setattr(c.training, "learning_rate", float("inf")),
            ),
            (
                "training.weight_decay",
                lambda c: setattr(c.training, "weight_decay", float("nan")),
            ),
        )
        for field, mutate in mutations:
            with self.subTest(field=field):
                config = AppConfig()
                mutate(config)
                with self.assertRaisesRegex(ConfigError, field):
                    config.validate()

        with self.assertRaisesRegex(ConfigError, "dataset.test_ratio"):
            self.load_text("dataset:\n  test_ratio: .nan\n")

    def test_allowed_backends_come_from_the_registry(self) -> None:
        """Registering a component must be the only step to make it selectable."""

        from tgseqloc.components import register_builtin_components

        available = register_builtin_components().available("graph_encoder")
        config = AppConfig()
        config.model.graph_encoder = "definitely_not_registered"
        with self.assertRaises(ConfigError) as caught:
            config.validate()
        for name in available:
            self.assertIn(name, str(caught.exception))

    def test_defaults_are_the_final_configuration_of_the_paper(self) -> None:
        config = AppConfig().validate()
        self.assertEqual(config.preprocess.text_encoder.backend, "char_ngram")
        self.assertEqual(config.preprocess.text_encoder.params, {"embedding_dim": 64})
        self.assertEqual(config.preprocess.text_min_length, 3)
        self.assertTrue(config.preprocess.text_drop_numeric)
        self.assertEqual(config.preprocess.dynamic_text_threshold, 0.5)
        self.assertEqual(config.model.text_fusion, "set")
        self.assertEqual((config.model.n_layers, config.model.heads), (1, 4))
        self.assertEqual((config.model.hidden_dim, config.model.proj_dim), (256, 256))
        self.assertEqual(config.training.margin, 0.3)
        self.assertEqual(config.training.negatives_per_query, 2)
        self.assertEqual(config.training.miner_warmup_epochs, 10)
        self.assertEqual(config.training.learning_rate, 3e-5)
        self.assertEqual(config.training.weight_decay, 1e-4)
        self.assertEqual(config.training.batch_size, 8)
        self.assertEqual(config.training.max_grad_norm, 5.0)
        self.assertEqual((config.training.epochs, config.training.patience), (30, 5))

    def test_dynamic_text_threshold_range(self) -> None:
        for value in (0.0, 1.5, -0.1):
            with self.subTest(value=value):
                config = AppConfig()
                config.preprocess.dynamic_text_threshold = value
                with self.assertRaisesRegex(ConfigError, "dynamic_text_threshold"):
                    config.validate()
        config = AppConfig()
        config.preprocess.dynamic_text_threshold = None
        config.validate()

    def test_component_params_load_from_yaml_untyped(self) -> None:
        """Params belong to the implementation, so any mapping is accepted."""

        config = self.load_text(
            "preprocess:\n"
            "  text_encoder:\n"
            "    backend: char_ngram\n"
            "    params:\n"
            "      embedding_dim: 384\n"
            "      n_values: [3, 4]\n"
        )
        self.assertEqual(
            config.preprocess.text_encoder.params, {"embedding_dim": 384, "n_values": [3, 4]}
        )

    def test_component_params_must_be_a_mapping(self) -> None:
        with self.assertRaisesRegex(ConfigError, "must be a mapping"):
            self.load_text("preprocess:\n  text_encoder:\n    params: [1, 2]\n")


class RegistryTests(unittest.TestCase):
    def test_normalized_lookup_creation_and_availability(self) -> None:
        registry = Registry()

        @registry.register("graph-encoder", "My-Factory")
        def factory(value: int = 0) -> dict[str, int]:
            return {"value": value}

        self.assertIs(registry.get("GRAPH_ENCODER", "my_factory"), factory)
        self.assertTrue(registry.contains("graph_encoder", " MY-FACTORY "))
        self.assertEqual(registry.available("graph_encoder"), ("my_factory",))
        self.assertEqual(registry.create("graph_encoder", "my-factory", 3), {"value": 3})

    def test_lookup_and_registration_errors_are_informative(self) -> None:
        registry = Registry()
        registry.add("metric", "known", lambda: None)
        with self.assertRaisesRegex(KeyError, "available: known"):
            registry.get("metric", "missing")
        with self.assertRaisesRegex(KeyError, "unknown registry kind"):
            registry.get("not-a-kind", "value")
        with self.assertRaisesRegex(KeyError, "already registered"):
            registry.add("metric", "KNOWN", lambda: None)
        replacement = lambda: "replacement"
        registry.add("metric", "known", replacement, replace=True)
        self.assertIs(registry.get("metric", "known"), replacement)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            registry.contains("metric", " ")


class TextNodeAblationTests(unittest.TestCase):
    """The no-text variant must reach the graphs, not only the model."""

    def test_flag_defaults_to_building_text_nodes(self) -> None:
        self.assertTrue(AppConfig().preprocess.use_text_nodes)

    def test_disabling_text_nodes_yields_object_only_graphs(self) -> None:
        import tempfile

        from tests.helpers import make_app_config
        from tgseqloc.pipeline import PipelineRunner

        with tempfile.TemporaryDirectory() as temporary:
            config, _ = make_app_config(Path(temporary), frame_count=4)
            config.preprocess.use_text_nodes = False
            config.model.use_text_nodes = False
            manifest = PipelineRunner(config).prepare()
            self.assertFalse(manifest["uses_text_nodes"])

            import torch

            for record in manifest["graph_records"]:
                graph = torch.load(
                    Path(config.dataset.prepared_root) / "v4rl" / record["path"],
                    map_location="cpu", weights_only=False,
                )
                self.assertEqual(int(graph.is_text.sum()), 0)
                self.assertEqual(list(graph.text_strings), [])


if __name__ == "__main__":
    unittest.main()
