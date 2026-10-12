from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tgseqloc.config import AppConfig, ComponentConfig, ConfigError, load_config
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
            ("sources.ocr", lambda c: setattr(c.sources, "ocr", "cloud")),
            ("sources.scene_graph", lambda c: setattr(c.sources, "scene_graph", "generated")),
            ("preprocess.text_filter", lambda c: setattr(c.preprocess, "text_filter", "none")),
            ("preprocess.fusion", lambda c: setattr(c.preprocess, "fusion", "concat")),
            ("preprocess.connection_strategy", lambda c: setattr(c.preprocess, "connection_strategy", "random")),
            ("cache.policy", lambda c: setattr(c.cache, "policy", "forever")),
            ("model.graph_encoder", lambda c: setattr(c.model, "graph_encoder", "gcn")),
            ("retrieval.retriever", lambda c: setattr(c.retrieval, "retriever", "annoy")),
            ("retrieval.reranker", lambda c: setattr(c.retrieval, "reranker", "unsupported")),
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
                "preprocess.ocr_confidence_threshold",
                lambda c: setattr(c.preprocess, "ocr_confidence_threshold", float("-inf")),
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

    def test_inference_stages_are_disabled_by_default(self) -> None:
        config = AppConfig()
        config.validate()
        self.assertFalse(config.segmentation.enabled)
        self.assertFalse(config.text_dynamics.enabled)
        self.assertEqual(config.segmentation.params, {})

    def test_dynamics_without_segmentation_is_rejected(self) -> None:
        config = AppConfig()
        config.text_dynamics = ComponentConfig(backend="mask_ioa")
        with self.assertRaisesRegex(ConfigError, "requires segmentation.backend"):
            config.validate()

    def test_weights_without_backend_is_rejected(self) -> None:
        config = AppConfig()
        config.segmentation = ComponentConfig(weights="yolo11x_seg")
        with self.assertRaisesRegex(ConfigError, "weights is set while"):
            config.validate()

    def test_component_params_load_from_yaml_untyped(self) -> None:
        """Params belong to the implementation, so any mapping is accepted."""

        config = self.load_text(
            "segmentation:\n"
            "  backend: null\n"
            "  params:\n"
            "    confidence: 0.4\n"
            "    classes: [person, car]\n"
        )
        self.assertEqual(
            config.segmentation.params, {"confidence": 0.4, "classes": ["person", "car"]}
        )

    def test_component_params_must_be_a_mapping(self) -> None:
        with self.assertRaisesRegex(ConfigError, "must be a mapping"):
            self.load_text("segmentation:\n  params: [1, 2]\n")


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


if __name__ == "__main__":
    unittest.main()


class TextNodeAblationTests(unittest.TestCase):
    """The "no text" arm must reach the graphs, not only the model."""

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


class ConfidenceFilterTests(unittest.TestCase):
    """Фильтр уверенности и распознаватели, которые её не сообщают."""

    def test_reported_confidence_is_compared(self) -> None:
        from tgseqloc.components import confidence_filter

        self.assertTrue(confidence_filter({"confidence": 0.9}, 0.5))
        self.assertFalse(confidence_filter({"confidence": 0.3}, 0.5))

    def test_unreported_confidence_passes(self) -> None:
        """Qwen3-VL пишет null; ноль здесь обнулил бы всю ветку сравнения."""

        from tgseqloc.components import confidence_filter

        self.assertTrue(confidence_filter({"confidence": None}, 0.5))
        self.assertTrue(confidence_filter({"text": "STARBUCKS"}, 0.5))
